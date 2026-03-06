"""
Tier 0c: Micro-Transformer on TinyShakespeare
Model: d_model=128, n_heads=4, n_layers=4, d_ff=512, ctx=256 (~1.2M params)
Target: val loss < 1.50
Runtime: ~10-15 minutes per run
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
import csv
import time
import math
import numpy as np

from cum import CUM
from cum.newton_schulz import newton_schulz_orthogonalize
from cum.utils import ns_convergence_error, sv_spread
from benchmarks.models.micro_gpt import MicroGPT


class SimpleMuon(torch.optim.Optimizer):
    """Minimal Muon for baseline."""
    def __init__(self, params, lr=0.02, beta1=0.95, weight_decay=0.01, ns_steps=5):
        defaults = dict(lr=lr, beta1=beta1, weight_decay=weight_decay, ns_steps=ns_steps)
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
                p.mul_(1 - group["weight_decay"])
                p.add_(orth, alpha=-group["lr"] * scale)


class CharDataset:
    """Character-level dataset from text file."""
    def __init__(self, text, ctx_len=256):
        self.chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
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


def run_experiment(opt_name, lr, dataset, total_steps=5000, batch_size=32, seed=42,
                   log_interval=100, eval_interval=250):
    torch.manual_seed(seed)

    model = MicroGPT(
        vocab_size=dataset.vocab_size,
        d_model=128, n_heads=4, n_layers=4, d_ff=512, ctx_len=256,
    )
    print(f"  Model params: {model.count_params():,}")

    warmup_steps = 200
    hidden_2d = model.get_hidden_2d_params()
    other = model.get_other_params()

    if opt_name == "CUM":
        main_opt = CUM(hidden_2d, lr=lr, ns_steps=3, weight_decay=0.0)
        aux_opt = torch.optim.AdamW(other, lr=3e-4, weight_decay=0.01)
    elif opt_name == "Muon":
        main_opt = SimpleMuon(hidden_2d, lr=lr, ns_steps=5, weight_decay=0.0)
        aux_opt = torch.optim.AdamW(other, lr=3e-4, weight_decay=0.01)
    elif opt_name == "AdamW":
        main_opt = None
        aux_opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    elif opt_name == "SGD+Nesterov":
        main_opt = None
        aux_opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.01)

    model.train()
    metrics = []
    steps_to_target = total_steps

    for step in range(total_steps):
        current_lr = get_lr_schedule(step, total_steps, warmup_steps, lr)
        if main_opt:
            for g in main_opt.param_groups:
                g["lr"] = current_lr
        for g in aux_opt.param_groups:
            if opt_name in ("AdamW", "SGD+Nesterov"):
                g["lr"] = current_lr

        x, y = dataset.get_batch(batch_size, split="train")

        t0 = time.perf_counter()
        _, loss = model(x, y)

        if main_opt:
            main_opt.zero_grad()
        aux_opt.zero_grad()
        loss.backward()
        if main_opt:
            main_opt.step()
        aux_opt.step()
        step_time = (time.perf_counter() - t0) * 1000

        if step % eval_interval == 0 or step == total_steps - 1:
            model.eval()
            val_losses = []
            for _ in range(10):
                vx, vy = dataset.get_batch(batch_size, split="val")
                with torch.no_grad():
                    _, vl = model(vx, vy)
                    val_losses.append(vl.item())
            val_loss = np.mean(val_losses)
            model.train()

            record = {
                "step": step,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "step_time_ms": step_time,
            }
            metrics.append(record)

            if step % log_interval == 0:
                print(f"    step {step:5d} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f} | {step_time:.1f}ms/step")

            if val_loss < 1.50 and steps_to_target == total_steps:
                steps_to_target = step

    return metrics, steps_to_target


def main():
    torch.set_num_threads(4)

    print("=" * 60)
    print("Tier 0c: Micro-Transformer on TinyShakespeare")
    print("=" * 60)

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "data", "tinyshakespeare.txt")
    if not os.path.exists(data_path):
        print("Downloading TinyShakespeare...")
        from benchmarks.tier0.data.download_tinyshakespeare import download
        download()

    with open(data_path, "r") as f:
        text = f.read()
    print(f"Dataset: {len(text):,} characters")

    dataset = CharDataset(text, ctx_len=256)
    print(f"Vocab size: {dataset.vocab_size}")

    lr_configs = {
        "SGD+Nesterov": [0.01, 0.05, 0.1],
        "AdamW": [3e-4, 1e-3, 3e-3],
        "Muon": [0.01, 0.02, 0.05],
        "CUM": [0.01, 0.02, 0.05],
    }

    all_results = []
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    for opt_name, lrs in lr_configs.items():
        print(f"\n{'='*40}")
        print(f"Optimizer: {opt_name}")
        print(f"{'='*40}")

        best_val = float("inf")
        best_lr = lrs[0]

        for lr in lrs:
            print(f"\n  LR = {lr}")
            t0 = time.perf_counter()
            metrics, steps_to_target = run_experiment(opt_name, lr, dataset, total_steps=5000)
            elapsed = time.perf_counter() - t0

            final_val = metrics[-1]["val_loss"]
            print(f"  Final val_loss: {final_val:.4f}, steps_to_target: {steps_to_target}, time: {elapsed:.0f}s")

            if final_val < best_val:
                best_val = final_val
                best_lr = lr

            # Save per-run metrics
            csv_path = os.path.join(results_dir, f"tier0c_{opt_name}_lr{lr}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
                writer.writeheader()
                writer.writerows(metrics)

            all_results.append({
                "optimizer": opt_name,
                "lr": lr,
                "final_val_loss": final_val,
                "steps_to_target": steps_to_target,
            })

        print(f"\n  Best: LR={best_lr}, val_loss={best_val:.4f}")

    # Summary
    csv_path = os.path.join(results_dir, "tier0c_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["optimizer", "lr", "final_val_loss", "steps_to_target"])
        writer.writeheader()
        writer.writerows(all_results)

    # Pass/fail
    best_cum_results = [r for r in all_results if r["optimizer"] == "CUM"]
    best_muon_results = [r for r in all_results if r["optimizer"] == "Muon"]
    best_cum = min(best_cum_results, key=lambda r: r["final_val_loss"])
    best_muon = min(best_muon_results, key=lambda r: r["final_val_loss"])

    passed = best_cum["final_val_loss"] <= best_muon["final_val_loss"]
    print(f"\n{'PASS' if passed else 'FAIL'}: CUM val_loss ({best_cum['final_val_loss']:.4f}) {'<=' if passed else '>'} Muon val_loss ({best_muon['final_val_loss']:.4f})")

    return passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
