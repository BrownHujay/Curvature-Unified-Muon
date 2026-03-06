"""
CUM v3 efficiency test: NS(3) + Soft Blend vs Muon NS(5).
Goal: beat Muon on BOTH val_loss AND wall time.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
import time
import math
import numpy as np

from cum.cum_v3 import CUMv3
from cum.newton_schulz import newton_schulz_orthogonalize
from benchmarks.models.micro_gpt import MicroGPT


class SimpleMuon(torch.optim.Optimizer):
    """Minimal Muon baseline."""
    def __init__(self, params, lr=0.02, beta1=0.95, ns_steps=5):
        defaults = dict(lr=lr, beta1=beta1, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                m = state["momentum_buffer"]
                m.mul_(group["beta1"]).add_(p.grad, alpha=1 - group["beta1"])
                u = p.grad + group["beta1"] * m
                orth = newton_schulz_orthogonalize(u, steps=group["ns_steps"])
                scale = math.sqrt(max(1, p.shape[0] / p.shape[1]))
                p.add_(orth, alpha=-group["lr"] * scale)


class CharDataset:
    def __init__(self, text, ctx_len=256):
        self.chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)
        self.ctx_len = ctx_len

    def get_batch(self, batch_size, split="train", train_ratio=0.9):
        n = int(len(self.data) * train_ratio)
        data = self.data[:n] if split == "train" else self.data[n:]
        ix = torch.randint(0, len(data) - self.ctx_len - 1, (batch_size,))
        x = torch.stack([data[i:i+self.ctx_len] for i in ix])
        y = torch.stack([data[i+1:i+self.ctx_len+1] for i in ix])
        return x, y


def get_lr_schedule(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def run_one(name, main_opt, aux_opt, model, dataset, total_steps=2000,
            batch_size=32, warmup_steps=200, base_lr=0.02, eval_every=500):
    model.train()
    t0 = time.perf_counter()

    for step in range(total_steps):
        current_lr = get_lr_schedule(step, total_steps, warmup_steps, base_lr)
        if main_opt:
            for g in main_opt.param_groups:
                g["lr"] = current_lr

        x, y = dataset.get_batch(batch_size, split="train")
        _, loss = model(x, y)

        if main_opt:
            main_opt.zero_grad()
        aux_opt.zero_grad()
        loss.backward()
        if main_opt:
            main_opt.step()
        aux_opt.step()

        if step % eval_every == 0 or step == total_steps - 1:
            model.eval()
            val_losses = []
            for _ in range(10):
                vx, vy = dataset.get_batch(batch_size, split="val")
                with torch.no_grad():
                    _, vl = model(vx, vy)
                    val_losses.append(vl.item())
            val_loss = np.mean(val_losses)
            model.train()
            elapsed = time.perf_counter() - t0
            print(f"  [{name}] step {step}: val_loss={val_loss:.4f} ({elapsed:.0f}s)")

    elapsed = time.perf_counter() - t0
    model.eval()
    val_losses = []
    for _ in range(20):
        vx, vy = dataset.get_batch(batch_size, split="val")
        with torch.no_grad():
            _, vl = model(vx, vy)
            val_losses.append(vl.item())
    final_val = np.mean(val_losses)
    return final_val, elapsed


def make_model_and_opts(dataset, opt_config):
    torch.manual_seed(42)
    model = MicroGPT(
        vocab_size=dataset.vocab_size,
        d_model=128, n_heads=4, n_layers=4, d_ff=512, ctx_len=256,
    )
    hidden_2d = model.get_hidden_2d_params()
    other = model.get_other_params()
    aux_opt = torch.optim.AdamW(other, lr=3e-4, weight_decay=0.01)

    name = opt_config["name"]
    ns_steps = opt_config.get("ns_steps", 5)

    if name.startswith("Muon"):
        main_opt = SimpleMuon(hidden_2d, lr=0.02, ns_steps=ns_steps)
    else:
        main_opt = CUMv3(
            hidden_2d, lr=0.02, ns_steps=ns_steps,
            ns_blend=opt_config.get("ns_blend", 0.0),
        )

    return model, main_opt, aux_opt


def main():
    torch.set_num_threads(4)

    print("=" * 60)
    print("CUM v3 Efficiency Test: NS(3)+Blend vs Muon NS(5)")
    print("=" * 60)

    data_path = os.path.join(os.path.dirname(__file__), "data", "tinyshakespeare.txt")
    with open(data_path, "r") as f:
        text = f.read()
    dataset = CharDataset(text, ctx_len=256)
    print(f"Dataset: {len(text):,} chars, vocab={dataset.vocab_size}")

    configs = [
        {"name": "Muon NS=5", "ns_steps": 5},
        {"name": "Muon NS=3", "ns_steps": 3},
        {"name": "CUMv3 NS3+s0.1", "ns_steps": 3, "ns_blend": 0.1},
        {"name": "CUMv3 NS3+s0.15", "ns_steps": 3, "ns_blend": 0.15},
        {"name": "CUMv3 NS4+s0.1", "ns_steps": 4, "ns_blend": 0.1},
    ]

    results = []
    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        model, main_opt, aux_opt = make_model_and_opts(dataset, cfg)
        val_loss, elapsed = run_one(
            cfg["name"], main_opt, aux_opt, model, dataset,
            total_steps=2000, eval_every=500,
        )
        results.append((cfg["name"], val_loss, elapsed))
        print(f"  {cfg['name']:25s}: val_loss={val_loss:.4f}  time={elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    muon5_loss = results[0][1]
    muon5_time = results[0][2]
    for name, val_loss, elapsed in results:
        delta_loss = val_loss - muon5_loss
        delta_time = elapsed - muon5_time
        print(f"  {name:25s}: val={val_loss:.4f} ({delta_loss:+.4f})  time={elapsed:.0f}s ({delta_time:+.0f}s)")

    # Find best CUM that beats Muon on val loss
    cum_results = [(n, v, t) for n, v, t in results if n.startswith("CUM")]
    beats_muon = [(n, v, t) for n, v, t in cum_results if v < muon5_loss]
    if beats_muon:
        best = min(beats_muon, key=lambda x: x[2])  # fastest that beats muon
        print(f"\nBEST: {best[0]} — beats Muon by {muon5_loss - best[1]:.4f} val_loss, {muon5_time - best[2]:.0f}s faster")
    else:
        best = min(cum_results, key=lambda x: x[1])
        print(f"\nNo CUM beats Muon. Closest: {best[0]} (delta={best[1] - muon5_loss:+.4f})")

    print("=" * 60)


if __name__ == "__main__":
    main()
