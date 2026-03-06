"""
Tier 0a: Synthetic Quadratic Optimization
Minimize f(W) = ||AW - B||^2_F where A has condition number 100.

Sanity check: all optimizers should converge on a convex quadratic.
CUM/Muon use NS-orthogonalized updates (designed for neural nets, not quadratics),
so the bar is convergence, not beating SGD/AdamW.
Runtime: < 15 seconds per run.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import csv
import time
import math

from cum import CUM
from cum.newton_schulz import newton_schulz_orthogonalize


# ── Simple Muon baseline ──
class SimpleMuon(torch.optim.Optimizer):
    """Minimal Muon implementation for baseline comparison."""
    def __init__(self, params, lr=0.02, beta1=0.95, weight_decay=0.01, ns_steps=5):
        defaults = dict(lr=lr, beta1=beta1, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                m = state["momentum_buffer"]
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                u = g + beta1 * m

                orth = newton_schulz_orthogonalize(u, steps=ns_steps)
                scale = math.sqrt(max(1, p.shape[0] / p.shape[1]))

                p.mul_(1 - wd)
                p.add_(orth, alpha=-lr * scale)


def make_ill_conditioned_problem(n=32, condition_number=100, seed=42):
    """Create square A (n x n) with known condition number, random B.
    Square system so optimal loss = 0 and convergence is testable."""
    torch.manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(n, n))
    V, _ = torch.linalg.qr(torch.randn(n, n))
    sigmas = torch.linspace(1.0, condition_number, n)
    A = U @ torch.diag(sigmas) @ V.T
    B = torch.randn(n, n)
    return A, B


def run_optimizer(opt_name, optimizer_fn, A, B, max_steps=2000, target_loss=1.0, lr_list=None):
    """Run optimizer and return results for each LR."""
    if lr_list is None:
        lr_list = [0.005, 0.01, 0.02, 0.05, 0.1]

    results = []
    for lr in lr_list:
        torch.manual_seed(42)
        W = torch.randn(A.shape[1], B.shape[1], requires_grad=True)

        opt = optimizer_fn(lr, [W])

        steps_to_target = max_steps
        losses = []
        for step in range(max_steps):
            loss = ((A @ W - B) ** 2).sum()
            losses.append(loss.item())

            if loss.item() < target_loss:
                steps_to_target = step
                break

            if math.isnan(loss.item()):
                break

            opt.zero_grad()
            loss.backward()
            opt.step()

        results.append({
            "optimizer": opt_name,
            "lr": lr,
            "steps_to_target": steps_to_target,
            "final_loss": losses[-1],
            "losses": losses,
        })
    return results


def main():
    torch.set_num_threads(4)

    print("=" * 60)
    print("Tier 0a: Synthetic Quadratic (Ill-Conditioned)")
    print("=" * 60)

    A, B = make_ill_conditioned_problem(n=32, condition_number=100)
    print(f"A shape: {A.shape}, B shape: {B.shape}")
    print(f"A condition number: {torch.linalg.cond(A).item():.1f}")

    # Compute theoretical optimum for reference
    W_opt = torch.linalg.solve(A, B)
    optimal_loss = ((A @ W_opt - B) ** 2).sum().item()
    print(f"Optimal loss: {optimal_loss:.2e}")

    # Target: converge to reasonable loss (quadratic minimum is ~0)
    target_loss = 10.0

    # Per-optimizer LR ranges
    lr_configs = {
        "SGD+Nesterov": [0.00005, 0.0001, 0.0002, 0.0005, 0.001],
        "AdamW":        [0.001, 0.005, 0.01, 0.05, 0.1],
        "Muon":         [0.005, 0.01, 0.02, 0.05, 0.1],
        "CUM":          [0.005, 0.01, 0.02, 0.05, 0.1],
    }

    optimizers = {
        "SGD+Nesterov": lambda lr, params: torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0),
        "AdamW": lambda lr, params: torch.optim.AdamW(params, lr=lr, weight_decay=0.0),
        "Muon": lambda lr, params: SimpleMuon(params, lr=lr, ns_steps=5, weight_decay=0.0),
        "CUM": lambda lr, params: CUM(params, lr=lr, ns_steps=3, weight_decay=0.0),
    }

    all_results = []
    for name, opt_fn in optimizers.items():
        t0 = time.perf_counter()
        results = run_optimizer(name, opt_fn, A, B, max_steps=3000, target_loss=target_loss, lr_list=lr_configs[name])
        t1 = time.perf_counter()
        all_results.extend(results)

        # Pick best by steps_to_target, break ties by final_loss
        best = min(results, key=lambda r: (r["steps_to_target"], r["final_loss"]))
        print(f"\n{name}:")
        print(f"  Best LR: {best['lr']}, Steps to target: {best['steps_to_target']}, Final loss: {best['final_loss']:.6e}")
        print(f"  Time: {t1-t0:.2f}s")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "tier0a_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["optimizer", "lr", "steps_to_target", "final_loss"])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: v for k, v in r.items() if k != "losses"})

    print(f"\nResults saved to {csv_path}")

    # Pass/fail: CUM should converge and not be wildly worse than Muon.
    # On a quadratic, factored preconditioning isn't the right tool, so
    # we just check CUM converges and is within 10x of Muon.
    best_cum = min([r for r in all_results if r["optimizer"] == "CUM"], key=lambda r: r["final_loss"])
    best_muon = min([r for r in all_results if r["optimizer"] == "Muon"], key=lambda r: r["final_loss"])

    cum_converged = best_cum["final_loss"] < target_loss
    muon_converged = best_muon["final_loss"] < target_loss
    cum_matches = best_cum["final_loss"] <= 10.0 * best_muon["final_loss"] + 1e-6

    passed = cum_converged and cum_matches
    print(f"\nCUM  best loss: {best_cum['final_loss']:.6e} (lr={best_cum['lr']})")
    print(f"Muon best loss: {best_muon['final_loss']:.6e} (lr={best_muon['lr']})")
    print(f"CUM converged: {cum_converged}, Muon converged: {muon_converged}, CUM matches Muon: {cum_matches}")
    print(f"{'PASS' if passed else 'FAIL'}")

    return passed, all_results


if __name__ == "__main__":
    passed, _ = main()
    sys.exit(0 if passed else 1)
