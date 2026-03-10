"""
Series 15: Universal Muon — Kill the AdamW Crutch
Quick local benchmark on M3 CPU.

Tests:
1. Muon+AdamW (baseline) — standard split optimizer
2. Universal all — single UniversalMuon for ALL params, no AdamW
3. Universal 2D + AdamW 1D — isolate whether 1D handling is the bottleneck
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import time
import math
import numpy as np

from cum.newton_schulz import newton_schulz_orthogonalize
from cum.utils import aspect_ratio_scale
from cum.tensor import UniversalMuon
from benchmarks.models.micro_gpt import MicroGPT


# === Config ===
MODEL_CFG = dict(d_model=128, n_heads=4, n_layers=4, d_ff=512, ctx_len=256)
BATCH_SIZE = 32
TOTAL_STEPS = 2000
WARMUP_STEPS = 200
EVAL_EVERY = 250
BASE_LR = 0.02
SEED = 42


class CharDataset:
    def __init__(self, text, ctx_len=256):
        self.chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)
        self.ctx_len = ctx_len

    def get_batch(self, batch_size, split='train', train_ratio=0.9):
        n = int(len(self.data) * train_ratio)
        data = self.data[:n] if split == 'train' else self.data[n:]
        ix = torch.randint(0, len(data) - self.ctx_len - 1, (batch_size,))
        x = torch.stack([data[i:i+self.ctx_len] for i in ix])
        y = torch.stack([data[i+1:i+self.ctx_len+1] for i in ix])
        return x, y


class SimpleMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, beta1=0.95, ns_steps=5):
        defaults = dict(lr=lr, beta1=beta1, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)
                m = state['momentum_buffer']
                m.mul_(group['beta1']).add_(p.grad, alpha=1 - group['beta1'])
                u = p.grad + group['beta1'] * m
                orth = newton_schulz_orthogonalize(u, steps=group['ns_steps'])
                scale = math.sqrt(max(1, p.shape[0] / p.shape[1]))
                p.add_(orth, alpha=-group['lr'] * scale)


def get_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def print_param_breakdown(model, cfg_name):
    """Print how many params are in each shape group."""
    counts = {0: 0, 1: 0, 2: 0}
    nums = {0: 0, 1: 0, 2: 0}
    for p in model.parameters():
        d = min(p.ndim, 2)
        counts[d] += 1
        nums[d] += p.numel()
    total = sum(nums.values())
    print(f'  [{cfg_name}] Param breakdown:')
    for d in [2, 1, 0]:
        label = {0: 'scalar', 1: '1D', 2: '2D'}[d]
        if counts[d] > 0:
            print(f'    {label}: {counts[d]} params, {nums[d]:,} elements ({100*nums[d]/total:.1f}%)')
    print(f'    total: {sum(counts.values())} params, {total:,} elements')


def train_one(name, optimizers, model, dataset):
    """Train with a list of optimizers. Each gets zero_grad/step called."""
    model.train()
    trajectory = []

    t0 = time.perf_counter()
    for step in range(TOTAL_STEPS):
        current_lr = get_lr(step, TOTAL_STEPS, WARMUP_STEPS, BASE_LR)
        for opt in optimizers:
            for g in opt.param_groups:
                # Scale LR for AdamW differently — it uses 3e-4 base
                if isinstance(opt, torch.optim.AdamW):
                    g['lr'] = get_lr(step, TOTAL_STEPS, WARMUP_STEPS, 3e-4)
                else:
                    g['lr'] = current_lr

        x, y = dataset.get_batch(BATCH_SIZE, split='train')
        _, loss = model(x, y)

        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        for opt in optimizers:
            opt.step()

        if step % EVAL_EVERY == 0 or step == TOTAL_STEPS - 1:
            model.eval()
            vl = []
            for _ in range(20):
                vx, vy = dataset.get_batch(BATCH_SIZE, split='val')
                with torch.no_grad():
                    _, v = model(vx, vy)
                    vl.append(v.item())
            val_loss = np.mean(vl)
            trajectory.append((step, val_loss))
            model.train()
            elapsed = time.perf_counter() - t0
            print(f'  [{name}] step {step}: val_loss={val_loss:.4f} ({elapsed:.0f}s)')

    elapsed = time.perf_counter() - t0

    model.eval()
    vl = []
    for _ in range(50):
        vx, vy = dataset.get_batch(BATCH_SIZE, split='val')
        with torch.no_grad():
            _, v = model(vx, vy)
            vl.append(v.item())
    final_val = np.mean(vl)
    return final_val, trajectory, elapsed


def make_model_and_opts(dataset, cfg):
    torch.manual_seed(SEED)
    model = MicroGPT(vocab_size=dataset.vocab_size, **MODEL_CFG)

    t = cfg['type']

    if t == 'Muon+AdamW':
        # Standard split: Muon for hidden 2D, AdamW for everything else
        hidden_2d = model.get_hidden_2d_params()
        other = model.get_other_params()
        muon = SimpleMuon(hidden_2d, lr=BASE_LR, ns_steps=5)
        adam = torch.optim.AdamW(other, lr=3e-4, weight_decay=0.01)
        optimizers = [muon, adam]

    elif t == 'Universal all':
        # Single optimizer for ALL params — no splitting
        uni = UniversalMuon(
            model.parameters(),
            lr=BASE_LR,
            beta1=0.95,
            ns_steps=5,
            scale_1d=cfg.get('scale_1d', 1.0),
        )
        optimizers = [uni]

    elif t == 'Universal 2D + AdamW 1D':
        # Split: UniversalMuon for 2D, AdamW for 1D (isolate 1D handling)
        params_2d = [p for p in model.parameters() if p.ndim == 2]
        params_1d = [p for p in model.parameters() if p.ndim < 2]
        uni = UniversalMuon(params_2d, lr=BASE_LR, ns_steps=5)
        adam = torch.optim.AdamW(params_1d, lr=3e-4, weight_decay=0.01)
        optimizers = [uni, adam]

    else:
        raise ValueError(f'Unknown config type: {t}')

    return model, optimizers


def main():
    torch.set_num_threads(4)
    print('=' * 60)
    print('Series 15: Universal Muon — Kill the AdamW Crutch')
    print('M3 CPU, 2000 steps, batch=32')
    print('=' * 60)

    data_path = os.path.join(os.path.dirname(__file__), 'data', 'tinyshakespeare.txt')
    with open(data_path, 'r') as f:
        text = f.read()
    dataset = CharDataset(text, ctx_len=MODEL_CFG['ctx_len'])
    print(f'Dataset: {len(text):,} chars, vocab={dataset.vocab_size}')

    configs = [
        {'name': 'Muon+AdamW (baseline)', 'type': 'Muon+AdamW'},
        {'name': 'Universal all', 'type': 'Universal all', 'scale_1d': 1.0},
        {'name': 'Universal 2D + AdamW 1D', 'type': 'Universal 2D + AdamW 1D'},
    ]

    results = []
    for i, cfg in enumerate(configs):
        name = cfg['name']
        print(f'\n{"=" * 60}')
        print(f'[{i+1}/{len(configs)}] {name}')
        print(f'{"=" * 60}')

        try:
            model, optimizers = make_model_and_opts(dataset, cfg)
            print_param_breakdown(model, name)
            val_loss, traj, elapsed = train_one(name, optimizers, model, dataset)
            results.append(dict(name=name, val_loss=val_loss, trajectory=traj,
                                elapsed=elapsed, error=None))
            print(f'  FINAL: {name} -> {val_loss:.4f} ({elapsed:.1f}s)')
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append(dict(name=name, val_loss=float('inf'),
                                trajectory=[], elapsed=0, error=str(e)))

    # Summary
    muon_vl = next((r['val_loss'] for r in results
                    if r['name'] == 'Muon+AdamW (baseline)'), None)

    print(f'\n\n{"=" * 60}')
    print('RESULTS SUMMARY')
    print(f'{"=" * 60}')
    print(f'| Config | Val Loss | vs Muon+AdamW | Time |')
    print(f'|--------|----------|---------------|------|')
    for r in sorted(results, key=lambda x: x['val_loss']):
        if r.get('error'):
            print(f'| {r["name"]} | FAILED | -- | {r["error"][:30]} |')
            continue
        vm = f'{r["val_loss"] - muon_vl:+.4f}' if muon_vl else '--'
        print(f'| {r["name"]} | {r["val_loss"]:.4f} | {vm} | {r["elapsed"]:.0f}s |')

    # Trajectory comparison
    print(f'\nTrajectory:')
    valid = [r for r in results if not r.get('error')]
    print(f'| Step |', end='')
    for r in valid:
        print(f' {r["name"][:20]:>20} |', end='')
    print()
    steps_to_show = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 1999]
    for step in steps_to_show:
        print(f'| {step:4d} |', end='')
        for r in valid:
            val = next((v for s, v in r['trajectory'] if s == step), None)
            if val:
                print(f' {val:20.4f} |', end='')
            else:
                print(f' {"--":>20} |', end='')
        print()


if __name__ == '__main__':
    main()
