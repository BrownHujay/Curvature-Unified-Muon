"""
CUM v6-v9: Consolidated test of four new mathematical innovations.

v5 baseline: save@2 b=0.15 → val_loss=1.5077 (-0.0113 vs Muon)

New ideas being tested:
  v6: Adaptive blend (spectral divergence drives per-layer blend)
  v7: Orthogonal feedback loop (prev orth fed into NS input) + v5 blend
  v8: Multi-scale curvature (three-point blend: NS₁, NS₃, NS₅)
  v9: Dampened late-stage NS (modify iteration itself, zero extra memory)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import time
import math
import numpy as np

from cum.cum_v5 import CUMv5
from cum.cum_v6 import CUMv6
from cum.cum_v7 import CUMv7
from cum.cum_v8 import CUMv8
from cum.cum_v9 import CUMv9
from cum.cum_v10 import CUMv10
from cum.cum_v11 import CUMv11
from cum.cum_v12 import CUMv12
from cum.cum_2v1 import CUM2v1
from cum.cum_2v2 import CUM2v2
from cum.cum_3v1 import CUM3v1
from cum.cum_3v2 import CUM3v2
from cum.cum_3v3 import CUM3v3
from cum.cum_4v1 import CUM4v1
from cum.cum_4v2 import CUM4v2
from cum.cum_4v3 import CUM4v3
from cum.cum_5v1 import CUM5v1
from cum.cum_5v2 import CUM5v2
from cum.cum_5v3 import CUM5v3
from cum.cum_5v4 import CUM5v4
from cum.cum_5v5 import CUM5v5
from cum.cum_5v6 import CUM5v6
from cum.cum_5v7 import CUM5v7
from cum.newton_schulz import newton_schulz_orthogonalize
from cum.utils import aspect_ratio_scale
from benchmarks.models.micro_gpt import MicroGPT


class SimpleMuon(torch.optim.Optimizer):
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

    # Warmup the compiled model
    for _ in range(10):
        x, y = dataset.get_batch(batch_size, split="train")
        _, _ = model(x, y)

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


def make_model_and_opts(dataset, cfg):
    torch.manual_seed(42)
    model = MicroGPT(
        vocab_size=dataset.vocab_size,
        d_model=128, n_heads=4, n_layers=4, d_ff=512, ctx_len=256,
    )

    # torch.compile the model for ~20% faster fwd+bwd
    model = torch.compile(model, mode="reduce-overhead")

    hidden_2d = model.get_hidden_2d_params()
    other = model.get_other_params()
    aux_opt = torch.optim.AdamW(other, lr=3e-4, weight_decay=0.01)

    name = cfg["name"]
    opt_type = cfg.get("type", name)

    if opt_type == "Muon":
        main_opt = SimpleMuon(hidden_2d, lr=0.02, ns_steps=5)
    elif opt_type == "v5":
        main_opt = CUMv5(
            hidden_2d, lr=0.02, ns_steps=5,
            ns_save_at=cfg.get("ns_save_at", 2),
            ns_blend=cfg.get("ns_blend", 0.15),
        )
    elif opt_type == "v6":
        main_opt = CUMv6(
            hidden_2d, lr=0.02, ns_steps=5,
            ns_save_at=cfg.get("ns_save_at", 2),
            blend_scale=cfg.get("blend_scale", 0.5),
            blend_max=cfg.get("blend_max", 0.3),
            beta_blend=cfg.get("beta_blend", 0.9),
        )
    elif opt_type == "v7":
        main_opt = CUMv7(
            hidden_2d, lr=0.02, ns_steps=5,
            ns_save_at=cfg.get("ns_save_at", 2),
            ns_blend=cfg.get("ns_blend", 0.15),
            beta_feedback=cfg.get("beta_feedback", 0.1),
        )
    elif opt_type == "v8":
        main_opt = CUMv8(
            hidden_2d, lr=0.02, ns_steps=5,
            w1=cfg.get("w1", 0.05),
            w3=cfg.get("w3", 0.10),
        )
    elif opt_type == "v9":
        main_opt = CUMv9(
            hidden_2d, lr=0.02, ns_steps=5,
            dampen_after=cfg.get("dampen_after", 2),
            dampen_factor=cfg.get("dampen_factor", 0.3),
        )
    elif opt_type == "v10":
        main_opt = CUMv10(
            hidden_2d, lr=0.02, ns_steps=5,
            ns_save_at=cfg.get("ns_save_at", 2),
            ns_blend=cfg.get("ns_blend", 0.15),
            dampen_after=cfg.get("dampen_after", 2),
            dampen_factor=cfg.get("dampen_factor", 0.3),
        )
    elif opt_type == "v11":
        main_opt = CUMv11(
            hidden_2d, lr=0.02, ns_steps=5,
            ns_save_at=cfg.get("ns_save_at", 2),
            ns_blend=cfg.get("ns_blend", 0.15),
            beta2=cfg.get("beta2", 0.999),
            graft_strength=cfg.get("graft_strength", 0.3),
        )
    elif opt_type == "v12":
        main_opt = CUMv12(
            hidden_2d, lr=0.02, ns_steps=5,
            ns_save_at=cfg.get("ns_save_at", 2),
            ns_blend=cfg.get("ns_blend", 0.15),
            beta_diff=cfg.get("beta_diff", 0.9),
            alpha_diff=cfg.get("alpha_diff", 0.1),
        )
    elif opt_type == "2v1":
        main_opt = CUM2v1(
            hidden_2d, lr=0.02, ns_steps=5,
            rank=cfg.get("rank", 4),
            curv_blend=cfg.get("curv_blend", 0.15),
        )
    elif opt_type == "2v2":
        main_opt = CUM2v2(
            hidden_2d, lr=0.02,
            sv_alpha=cfg.get("sv_alpha", 0.0),
        )
    elif opt_type == "3v1":
        main_opt = CUM3v1(
            hidden_2d, lr=0.02, ns_steps=5,
            warm_steps=cfg.get("warm_steps", 3),
            inject_rate=cfg.get("inject_rate", 0.5),
        )
    elif opt_type == "3v2":
        main_opt = CUM3v2(
            hidden_2d, lr=0.02,
            ns_steps=cfg.get("ns_steps", 3),
        )
    elif opt_type == "3v3":
        main_opt = CUM3v3(
            hidden_2d, lr=0.02,
            beta_dir=cfg.get("beta_dir", 0.95),
            beta_mag=cfg.get("beta_mag", 0.95),
            ns_steps=5,
        )
    elif opt_type == "4v1":
        main_opt = CUM4v1(
            hidden_2d, lr=cfg.get("lr", 0.02),
            beta2=cfg.get("beta2", 0.999),
            precond_freq=cfg.get("precond_freq", 10),
        )
    elif opt_type == "4v2":
        main_opt = CUM4v2(
            hidden_2d, lr=cfg.get("lr", 0.02),
            top_k=cfg.get("top_k", 4),
            dampen_alpha=cfg.get("dampen_alpha", 0.5),
            power_iters=cfg.get("power_iters", 5),
        )
    elif opt_type == "4v3":
        main_opt = CUM4v3(
            hidden_2d, lr=cfg.get("lr", 0.02),
            top_k=cfg.get("top_k", 8),
            reweight_strength=cfg.get("reweight_strength", 0.5),
            subspace_freq=cfg.get("subspace_freq", 5),
        )
    elif opt_type == "5v1":
        main_opt = CUM5v1(
            hidden_2d, lr=0.02,
            ns_steps=cfg.get("ns_steps", 5),
        )
    elif opt_type == "5v2":
        main_opt = CUM5v2(
            hidden_2d, lr=0.02,
            eq_beta=cfg.get("eq_beta", 5.0),
        )
    elif opt_type == "5v3":
        main_opt = CUM5v3(
            hidden_2d, lr=0.02,
            schatten_p=cfg.get("schatten_p", 16.0),
        )
    elif opt_type == "5v4":
        main_opt = CUM5v4(
            hidden_2d, lr=0.02,
            p_min=cfg.get("p_min", 4.0),
            p_max=cfg.get("p_max", 64.0),
            cond_target=cfg.get("cond_target", 10.0),
        )
    elif opt_type == "5v5":
        main_opt = CUM5v5(
            hidden_2d, lr=0.02,
            ns_steps=cfg.get("ns_steps", 5),
        )
    elif opt_type == "5v6":
        main_opt = CUM5v6(
            hidden_2d, lr=0.02,
            mode=cfg.get("mode", "ns_blend"),
            ns_save_at=cfg.get("ns_save_at", 2),
            ns_blend=cfg.get("ns_blend", 0.15),
            tilt_eps=cfg.get("tilt_eps", 0.1),
        )
    elif opt_type == "5v7":
        main_opt = CUM5v7(
            hidden_2d, lr=0.02,
            mode=cfg.get("mode", "top_preserve"),
            ns_top_steps=cfg.get("ns_top_steps", 3),
            ns_bot_steps=cfg.get("ns_bot_steps", 5),
            split_frac=cfg.get("split_frac", 0.25),
            blend_start=cfg.get("blend_start", 0.35),
            blend_end=cfg.get("blend_end", 0.10),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    return model, main_opt, aux_opt


def main():
    torch.set_num_threads(4)

    print("=" * 60)
    print("CUM Series 5f: Novel SV Curves + Higher Blends")
    print("=" * 60)

    data_path = os.path.join(os.path.dirname(__file__), "data", "tinyshakespeare.txt")
    with open(data_path, "r") as f:
        text = f.read()
    dataset = CharDataset(text, ctx_len=256)
    print(f"Dataset: {len(text):,} chars, vocab={dataset.vocab_size}")

    configs = [
        # Current best baseline
        {"name": "5v6 s2 b=0.25", "type": "5v6", "mode": "ns_blend", "ns_save_at": 2, "ns_blend": 0.25},
        # Push blend higher
        {"name": "5v6 s2 b=0.35", "type": "5v6", "mode": "ns_blend", "ns_save_at": 2, "ns_blend": 0.35},
        # Split-spectrum: top 25% SVs get 3 iters, rest get 5
        {"name": "5v7 split top3", "type": "5v7", "mode": "top_preserve", "ns_top_steps": 3, "ns_bot_steps": 5, "split_frac": 0.25},
        # Adaptive schedule: b=0.35→0.10 cosine over training
        {"name": "5v7 schedule", "type": "5v7", "mode": "schedule", "blend_start": 0.35, "blend_end": 0.10},
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
    muon_loss = results[0][1]
    v5_loss = results[1][1]
    for name, val_loss, elapsed in results:
        delta_muon = val_loss - muon_loss
        delta_v5 = val_loss - v5_loss
        print(f"  {name:25s}: val={val_loss:.4f} (vs Muon: {delta_muon:+.4f}, vs v5: {delta_v5:+.4f})")

    # Find best among new ideas (index 2+)
    new_results = results[2:]
    if new_results:
        best = min(new_results, key=lambda x: x[1])
        improvement_muon = muon_loss - best[1]
        improvement_v5 = v5_loss - best[1]
        print(f"\nBEST NEW: {best[0]}")
        print(f"  vs Muon: -{improvement_muon:.4f}")
        print(f"  vs v5:   -{improvement_v5:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
