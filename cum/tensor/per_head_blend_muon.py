"""
PerHeadBlendMuon — Per-head orthogonalization + iterate/temporal blending.

Combines two winning strategies:
1. Per-head slicing (Series 15): NS each head independently
2. TD(lambda) + temporal blending (Series 12): oscillation cancellation

Modes:
- 'plain': Per-head NS only (equivalent to PerHeadMuon)
- 'combined': Per-head NS + two-point iterate blend + temporal EMA
- 'td': Per-head NS + TD(lambda) all-iterate blend + temporal EMA + custom polynomial

Supports:
- n_slices per param group: split rows (for QKV weights)
- col_slices per param group: transpose, split rows, NS each, recombine, transpose back (for out_proj)
"""

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from cum.newton_schulz import newton_schulz_orthogonalize, newton_schulz_multi_resolution
from cum.cum_12v1 import bifurcation_coeffs
from cum.utils import aspect_ratio_scale


def _custom_ns_all_slice(
    G: Tensor, steps: int, ns_a: float, ns_b: float, ns_c: float, eps: float,
) -> list:
    """Run custom polynomial NS on a single slice and return ALL iterates."""
    X = G / (G.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    iterates = []
    for i in range(steps):
        A = X @ X.T
        B = ns_b * A + ns_c * (A @ A)
        X = ns_a * X + B @ X
        iterates.append(X.T.clone() if transposed else X.clone())
    return iterates


def _standard_ns_all_slice(
    G: Tensor, steps: int, eps: float,
) -> list:
    """Run standard NS on a single slice and return ALL iterates."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    iterates = []
    for i in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        iterates.append(X.T.clone() if transposed else X.clone())
    return iterates


def _standard_ns_save_at(
    G: Tensor, steps: int, save_at: int, eps: float,
) -> tuple:
    """Run standard NS on a single slice, return (final, intermediate_at_save_at)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    intermediate = None
    for i in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        if i + 1 == save_at:
            intermediate = X.T.clone() if transposed else X.clone()
    final = X.T if transposed else X
    if intermediate is None:
        intermediate = final.clone()
    return final, intermediate


def _frobenius_blend(primary: Tensor, secondary: Tensor, weight: float, eps: float) -> Tensor:
    """Blend two matrices with Frobenius norm matching."""
    if weight <= 0:
        return primary
    p_norm = primary.norm()
    s_norm = secondary.norm()
    if s_norm > eps:
        secondary_scaled = secondary * (p_norm / s_norm)
        return (1 - weight) * primary + weight * secondary_scaled
    return primary


class PerHeadBlendMuon(Optimizer):
    """
    Per-head orthogonalization with optional iterate/temporal blending.

    Param groups can specify:
        n_slices: int  — split rows into n_slices chunks (for QKV)
        col_slices: int — transpose, split rows into col_slices chunks (for out_proj)

    Constructor params control blending behavior across all groups.
    """

    def __init__(self, params, lr=0.02, beta1=0.95, ns_steps=5,
                 mode='plain', td_lambda=0.5, deriv=-1.0,
                 input_blend_beta=0.5, input_blend_alpha=0.15,
                 eps=1e-7, nesterov=True):
        # Precompute custom polynomial coefficients if needed
        ns_a = ns_b = ns_c = None
        if mode == 'td' and deriv is not None:
            ns_a, ns_b, ns_c = bifurcation_coeffs(deriv)

        defaults = dict(
            lr=lr, beta1=beta1, ns_steps=ns_steps,
            mode=mode, td_lambda=td_lambda,
            ns_a=ns_a, ns_b=ns_b, ns_c=ns_c,
            input_blend_beta=input_blend_beta, input_blend_alpha=input_blend_alpha,
            n_slices=1, col_slices=1,
            eps=eps, nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            ns_steps = group['ns_steps']
            eps = group['eps']
            nesterov = group['nesterov']
            mode = group['mode']
            n_slices = group.get('n_slices', 1)
            col_slices = group.get('col_slices', 1)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)
                state['step'] += 1

                mb = state['momentum_buffer']
                mb.mul_(beta1).add_(grad, alpha=1 - beta1)

                if nesterov:
                    u = grad + beta1 * mb
                else:
                    u = mb.clone()

                if p.ndim != 2:
                    # Non-2D: just scale
                    p.add_(u, alpha=-lr)
                    continue

                # Determine slicing strategy
                if col_slices > 1 and u.shape[1] % col_slices == 0:
                    # Column-wise slicing: transpose, split rows, process, recombine, transpose back
                    orth = self._process_col_sliced(u, col_slices, group, state)
                elif n_slices > 1 and u.shape[0] % n_slices == 0:
                    # Row-wise slicing (standard per-head)
                    orth = self._process_row_sliced(u, n_slices, group, state)
                else:
                    # No slicing: treat as single slice
                    orth = self._process_row_sliced(u, 1, group, state)

                # Scale based on per-slice dimensions for the update
                m_dim, n_dim = u.shape
                if n_slices > 1 and u.shape[0] % n_slices == 0:
                    chunk_size = m_dim // n_slices
                    scale = aspect_ratio_scale(chunk_size, n_dim)
                elif col_slices > 1 and u.shape[1] % col_slices == 0:
                    chunk_size = n_dim // col_slices
                    scale = aspect_ratio_scale(chunk_size, m_dim)
                else:
                    scale = aspect_ratio_scale(m_dim, n_dim)

                p.add_(orth, alpha=-lr * scale)

        return loss

    def _process_row_sliced(self, u: Tensor, n_slices: int, group: dict, state: dict) -> Tensor:
        """Process with row-wise slicing (standard per-head for QKV)."""
        mode = group['mode']
        ns_steps = group['ns_steps']
        eps = group['eps']

        m_dim, n_dim = u.shape

        if n_slices <= 1:
            chunks = [u]
        else:
            chunk_size = m_dim // n_slices
            chunks = list(u.split(chunk_size, dim=0))

        if mode == 'plain':
            return self._plain_ns_chunks(chunks, ns_steps, eps, dim=0)
        elif mode == 'combined':
            return self._combined_ns_chunks(chunks, ns_steps, eps, group, state, dim=0)
        elif mode == 'td':
            return self._td_ns_chunks(chunks, ns_steps, eps, group, state, dim=0)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _process_col_sliced(self, u: Tensor, col_slices: int, group: dict, state: dict) -> Tensor:
        """Process with column-wise slicing (for out_proj).
        Transpose -> split rows -> process each -> recombine -> transpose back.
        """
        u_t = u.T  # (n_dim, m_dim)
        mode = group['mode']
        ns_steps = group['ns_steps']
        eps = group['eps']

        chunk_size = u_t.shape[0] // col_slices
        chunks = list(u_t.split(chunk_size, dim=0))

        if mode == 'plain':
            result_t = self._plain_ns_chunks(chunks, ns_steps, eps, dim=0)
        elif mode == 'combined':
            result_t = self._combined_ns_chunks(chunks, ns_steps, eps, group, state, dim=0)
        elif mode == 'td':
            result_t = self._td_ns_chunks(chunks, ns_steps, eps, group, state, dim=0)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return result_t.T  # Transpose back

    def _plain_ns_chunks(self, chunks: list, ns_steps: int, eps: float, dim: int) -> Tensor:
        """Plain per-head NS: just orthogonalize each chunk independently."""
        orth_chunks = []
        for chunk in chunks:
            orth_chunk = newton_schulz_orthogonalize(chunk, steps=ns_steps, eps=eps)
            orth_chunks.append(orth_chunk)
        return torch.cat(orth_chunks, dim=dim)

    def _combined_ns_chunks(self, chunks: list, ns_steps: int, eps: float,
                            group: dict, state: dict, dim: int) -> Tensor:
        """Combined mode: per-chunk two-point blend + temporal EMA."""
        input_beta = group['input_blend_beta']
        input_alpha = group['input_blend_alpha']
        save_at = 2  # Standard save point for two-point blend
        blend_w = 0.15  # Standard blend weight

        all_orth = []
        for i, chunk in enumerate(chunks):
            final, partial = _standard_ns_save_at(chunk, ns_steps, save_at, eps)
            iterate_blended = _frobenius_blend(final, partial, blend_w, eps)
            all_orth.append(iterate_blended)

        # Concatenate all chunks
        blended = torch.cat(all_orth, dim=dim)

        # Temporal EMA across steps (on the full concatenated result)
        if 'denoised_ema' not in state:
            state['denoised_ema'] = blended.clone()
        else:
            state['denoised_ema'].mul_(input_beta).add_(blended, alpha=1 - input_beta)
        orth = _frobenius_blend(blended, state['denoised_ema'], input_alpha, eps)
        return orth

    def _td_ns_chunks(self, chunks: list, ns_steps: int, eps: float,
                      group: dict, state: dict, dim: int) -> Tensor:
        """TD(lambda) mode: per-chunk all-iterate blend + temporal EMA + custom polynomial."""
        td_lambda = group['td_lambda']
        input_beta = group['input_blend_beta']
        input_alpha = group['input_blend_alpha']
        ns_a = group['ns_a']
        ns_b = group['ns_b']
        ns_c = group['ns_c']

        # Precompute TD weights: w_k = lambda^(n-k), normalized
        raw_weights = [td_lambda ** (ns_steps - k) for k in range(1, ns_steps + 1)]
        total_w = sum(raw_weights)
        weights = [w / total_w for w in raw_weights]

        all_blended = []
        for chunk in chunks:
            # Get all NS iterates for this chunk
            if ns_a is not None:
                iterates = _custom_ns_all_slice(chunk, ns_steps, ns_a, ns_b, ns_c, eps)
            else:
                iterates = _standard_ns_all_slice(chunk, ns_steps, eps)

            final = iterates[-1]
            f_norm = final.norm()

            # Norm-match all iterates to final's norm, then weighted sum
            chunk_blended = torch.zeros_like(final)
            for k, (iterate, w) in enumerate(zip(iterates, weights)):
                i_norm = iterate.norm()
                if i_norm > eps:
                    chunk_blended.add_(iterate * (f_norm / i_norm), alpha=w)
                else:
                    chunk_blended.add_(final, alpha=w)

            all_blended.append(chunk_blended)

        # Concatenate all chunks
        blended = torch.cat(all_blended, dim=dim)

        # Temporal EMA across steps
        if 'denoised_ema' not in state:
            state['denoised_ema'] = blended.clone()
        else:
            state['denoised_ema'].mul_(input_beta).add_(blended, alpha=1 - input_beta)
        orth = _frobenius_blend(blended, state['denoised_ema'], input_alpha, eps)
        return orth
