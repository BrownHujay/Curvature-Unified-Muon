"""
Series 14: Deep Research V4 Mathematical Frameworks
Quick local benchmark on M3 CPU.

Tests:
1. Muon (baseline)
2. 8v1 combined (reference)
3. Ruiz + NS3 (basic — does pre-conditioning let us cut NS steps?)
4. Ruiz + NS3 + combined blending
5. Frame potential (aggressive η + TD(λ) + temporal)
6. Polar Express (adaptive poly per step + TD(λ) + temporal)
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
from cum import CUM8v1, CUM14v1
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


def train_one(name, main_opt, aux_opt, model, dataset):
    model.train()
    trajectory = []

    t0 = time.perf_counter()
    for step in range(TOTAL_STEPS):
        current_lr = get_lr(step, TOTAL_STEPS, WARMUP_STEPS, BASE_LR)
        if main_opt:
            for g in main_opt.param_groups:
                g['lr'] = current_lr

        x, y = dataset.get_batch(BATCH_SIZE, split='train')
        _, loss = model(x, y)

        if main_opt:
            main_opt.zero_grad()
        aux_opt.zero_grad()
        loss.backward()
        if main_opt:
            main_opt.step()
        aux_opt.step()

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

    hidden_2d = model.get_hidden_2d_params()
    other = model.get_other_params()
    aux_opt = torch.optim.AdamW(other, lr=3e-4, weight_decay=0.01)

    t = cfg['type']
    if t == 'Muon':
        main_opt = SimpleMuon(hidden_2d, lr=BASE_LR, ns_steps=5)
    elif t == '8v1':
        main_opt = CUM8v1(
            hidden_2d, lr=BASE_LR,
            method='matrix', mode='combined',
            ns_steps=5, save_at=2, blend=0.15,
            input_blend_beta=0.5, input_blend_alpha=0.15,
            total_steps=TOTAL_STEPS,
        )
    elif t == '14v1':
        main_opt = CUM14v1(
            hidden_2d, lr=BASE_LR,
            mode=cfg['mode'],
            ruiz_steps=cfg.get('ruiz_steps', 5),
            ns_steps=cfg.get('ns_steps', 3),
            frame_steps=cfg.get('frame_steps', 7),
            frame_eta=cfg.get('frame_eta', 2.5),
            frame_c=cfg.get('frame_c', 0.88),
            pe_steps=cfg.get('pe_steps', 5),
            td_lambda=cfg.get('td_lambda', 0.5),
            input_blend_beta=0.5, input_blend_alpha=0.15,
        )
    else:
        raise ValueError(f'Unknown: {t}')

    return model, main_opt, aux_opt


def main():
    torch.set_num_threads(4)
    print('=' * 60)
    print('Series 14: Deep Research V4 Mathematical Frameworks')
    print('M3 CPU, 2000 steps, batch=32')
    print('=' * 60)

    data_path = os.path.join(os.path.dirname(__file__), 'data', 'tinyshakespeare.txt')
    with open(data_path, 'r') as f:
        text = f.read()
    dataset = CharDataset(text, ctx_len=MODEL_CFG['ctx_len'])
    print(f'Dataset: {len(text):,} chars, vocab={dataset.vocab_size}')

    configs = [
        {'name': 'Muon NS=5', 'type': 'Muon'},
        {'name': '8v1 combined', 'type': '8v1'},
        {'name': 'Ruiz5+NS3 basic', 'type': '14v1', 'mode': 'ruiz_ns',
         'ruiz_steps': 5, 'ns_steps': 3},
        {'name': 'Ruiz5+NS3 combined', 'type': '14v1', 'mode': 'ruiz_ns_combined',
         'ruiz_steps': 5, 'ns_steps': 3, 'td_lambda': 0.5},
        {'name': 'Frame η=2.5 7step', 'type': '14v1', 'mode': 'frame',
         'frame_steps': 7, 'frame_eta': 2.5, 'frame_c': 0.88, 'td_lambda': 0.5},
        {'name': 'Polar Express 5step', 'type': '14v1', 'mode': 'polar_express',
         'pe_steps': 5, 'td_lambda': 0.5},
    ]

    results = []
    for i, cfg in enumerate(configs):
        name = cfg['name']
        print(f'\n{"=" * 60}')
        print(f'[{i+1}/{len(configs)}] {name}')
        print(f'{"=" * 60}')

        try:
            model, main_opt, aux_opt = make_model_and_opts(dataset, cfg)
            val_loss, traj, elapsed = train_one(name, main_opt, aux_opt, model, dataset)
            results.append(dict(name=name, val_loss=val_loss, trajectory=traj,
                                elapsed=elapsed, error=None))
            print(f'  FINAL: {name} -> {val_loss:.4f} ({elapsed:.1f}s)')
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append(dict(name=name, val_loss=float('inf'),
                                trajectory=[], elapsed=0, error=str(e)))

    # Summary
    muon_vl = next((r['val_loss'] for r in results if r['name'] == 'Muon NS=5'), None)
    combined_vl = next((r['val_loss'] for r in results if r['name'] == '8v1 combined'), None)

    print(f'\n\n{"=" * 60}')
    print('RESULTS SUMMARY')
    print(f'{"=" * 60}')
    print(f'| Config | Val Loss | vs Muon | vs Combined | Time |')
    print(f'|--------|----------|---------|-------------|------|')
    for r in sorted(results, key=lambda x: x['val_loss']):
        if r.get('error'):
            print(f'| {r["name"]} | FAILED | -- | -- | {r["error"][:30]} |')
            continue
        vm = f'{r["val_loss"] - muon_vl:+.4f}' if muon_vl else '--'
        vc = f'{r["val_loss"] - combined_vl:+.4f}' if combined_vl else '--'
        print(f'| {r["name"]} | {r["val_loss"]:.4f} | {vm} | {vc} | {r["elapsed"]:.0f}s |')

    # Trajectory comparison
    print(f'\nTrajectory:')
    print(f'| Step |', end='')
    for r in results:
        if not r.get('error'):
            print(f' {r["name"][:15]:>15} |', end='')
    print()
    steps_to_show = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 1999]
    for step in steps_to_show:
        print(f'| {step:4d} |', end='')
        for r in results:
            if r.get('error'):
                continue
            val = next((v for s, v in r['trajectory'] if s == step), None)
            if val:
                print(f' {val:15.4f} |', end='')
            else:
                print(f' {"--":>15} |', end='')
        print()


if __name__ == '__main__':
    main()
