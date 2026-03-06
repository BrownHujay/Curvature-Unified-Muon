"""
Tier 0d: CIFAR-10 Tiny ConvNet
Conv2d(3,32,3) → Conv2d(32,64,3) → Conv2d(64,64,3) → FC(1024,10)
Target: 70% test accuracy
Runtime: ~3-5 minutes per run
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import time
import math

from cum import CUM
from cum.newton_schulz import newton_schulz_orthogonalize


class SimpleMuon(torch.optim.Optimizer):
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
                # Flatten conv weights to 2D for NS
                orig_shape = u.shape
                if u.ndim > 2:
                    u_2d = u.view(u.shape[0], -1)
                else:
                    u_2d = u
                orth_2d = newton_schulz_orthogonalize(u_2d, steps=group["ns_steps"])
                orth = orth_2d.view(orig_shape)
                scale = math.sqrt(max(1, u_2d.shape[0] / u_2d.shape[1]))
                p.mul_(1 - group["weight_decay"])
                p.add_(orth, alpha=-group["lr"] * scale)


class TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x16x16
        x = self.pool(F.relu(self.conv2(x)))  # 64x8x8
        x = self.pool(F.relu(self.conv3(x)))  # 64x4x4
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_hidden_2d(self):
        """Conv weights reshaped to 2D, plus FC weight."""
        params = []
        for m in [self.conv1, self.conv2, self.conv3, self.fc]:
            if hasattr(m, 'weight') and m.weight.ndim >= 2:
                params.append(m.weight)
        return params

    def get_other(self):
        hidden = set(id(p) for p in self.get_hidden_2d())
        return [p for p in self.parameters() if id(p) not in hidden]


class CUMConv(CUM):
    """CUM variant that handles conv weights by reshaping to 2D."""
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            sigma_max = group["sigma_max"]
            alpha_damp = group["alpha_damp"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                orig_shape = g.shape

                # Reshape to 2D if needed (conv weights)
                if g.ndim > 2:
                    g = g.view(g.shape[0], -1)
                    p_data = p.data.view(p.shape[0], -1)
                else:
                    p_data = p.data

                m_dim, n_dim = g.shape

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros(m_dim, n_dim, device=p.device, dtype=p.dtype)
                    state["row_var"] = torch.zeros(m_dim, device=p.device, dtype=p.dtype)
                    state["col_var"] = torch.zeros(n_dim, device=p.device, dtype=p.dtype)
                    v = torch.randn(n_dim, device=p.device, dtype=p.dtype)
                    state["power_iter_v"] = v / (v.norm() + 1e-7)

                state["step"] += 1
                step = state["step"]

                from cum.factored_precond import apply_factored_precond
                from cum.spectral_control import spectral_damping
                from cum.utils import aspect_ratio_scale

                mb = state["momentum_buffer"]
                mb.mul_(beta1).add_(g, alpha=1 - beta1)

                if nesterov:
                    u = g + beta1 * mb
                else:
                    u = mb.clone()

                u_precond, state["row_var"], state["col_var"] = apply_factored_precond(
                    u, g, state["row_var"], state["col_var"], beta2, step, eps,
                )

                from cum.newton_schulz import newton_schulz_orthogonalize
                orth = newton_schulz_orthogonalize(u_precond, steps=ns_steps, eps=eps)

                damping_factor, state["power_iter_v"] = spectral_damping(
                    p_data, state["power_iter_v"], sigma_max, alpha_damp,
                )

                scale = aspect_ratio_scale(m_dim, n_dim)
                p.data.mul_(1 - weight_decay)
                p.data.add_(orth.view(orig_shape), alpha=-lr * damping_factor * scale)

        return loss


def get_cifar10_data():
    from torchvision import datasets, transforms
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    data_dir = os.path.join(os.path.dirname(__file__), "data", "cifar10")
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, shuffle=False, num_workers=0)

    return train_loader, test_loader


def get_lr_schedule(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


def run_experiment(opt_name, lr, train_loader, test_loader, total_steps=3000, seed=42):
    torch.manual_seed(seed)
    model = TinyConvNet()
    warmup_steps = 150

    hidden_2d = model.get_hidden_2d()
    other = model.get_other()

    if opt_name == "CUM":
        main_opt = CUMConv(hidden_2d, lr=lr, ns_steps=3, weight_decay=0.0)
        aux_opt = torch.optim.AdamW(other, lr=3e-4, weight_decay=0.01) if other else None
    elif opt_name == "Muon":
        main_opt = SimpleMuon(hidden_2d, lr=lr, ns_steps=5, weight_decay=0.0)
        aux_opt = torch.optim.AdamW(other, lr=3e-4, weight_decay=0.01) if other else None
    elif opt_name == "AdamW":
        main_opt = None
        aux_opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    elif opt_name == "SGD+Nesterov":
        main_opt = None
        aux_opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.01)

    model.train()
    metrics = []
    step = 0
    steps_to_target = total_steps
    train_iter = iter(train_loader)

    for step in range(total_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        current_lr = get_lr_schedule(step, total_steps, warmup_steps, lr)
        if main_opt:
            for g in main_opt.param_groups:
                g["lr"] = current_lr
        if aux_opt:
            for g in aux_opt.param_groups:
                if opt_name in ("AdamW", "SGD+Nesterov"):
                    g["lr"] = current_lr

        t0 = time.perf_counter()
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        if main_opt:
            main_opt.zero_grad()
        if aux_opt:
            aux_opt.zero_grad()
        loss.backward()
        if main_opt:
            main_opt.step()
        if aux_opt:
            aux_opt.step()
        step_time = (time.perf_counter() - t0) * 1000

        if step % 200 == 0 or step == total_steps - 1:
            acc = evaluate(model, test_loader)
            metrics.append({
                "step": step, "train_loss": loss.item(),
                "test_acc": acc, "step_time_ms": step_time,
            })
            print(f"    step {step:5d} | loss {loss.item():.4f} | acc {acc:.4f} | {step_time:.1f}ms")

            if acc >= 0.70 and steps_to_target == total_steps:
                steps_to_target = step

    final_acc = evaluate(model, test_loader)
    return metrics, steps_to_target, final_acc


def main():
    torch.set_num_threads(4)

    print("=" * 60)
    print("Tier 0d: CIFAR-10 Tiny ConvNet")
    print("=" * 60)

    train_loader, test_loader = get_cifar10_data()

    lr_configs = {
        "SGD+Nesterov": [0.01, 0.05, 0.1],
        "AdamW": [3e-4, 1e-3, 3e-3],
        "Muon": [0.005, 0.01, 0.02],
        "CUM": [0.005, 0.01, 0.02],
    }

    all_results = []
    for opt_name, lrs in lr_configs.items():
        print(f"\n--- {opt_name} ---")
        for lr in lrs:
            print(f"\n  LR = {lr}")
            t0 = time.perf_counter()
            metrics, steps_to_target, final_acc = run_experiment(
                opt_name, lr, train_loader, test_loader, total_steps=3000,
            )
            elapsed = time.perf_counter() - t0
            print(f"  Final acc: {final_acc:.4f}, steps_to_70%: {steps_to_target}, time: {elapsed:.0f}s")

            all_results.append({
                "optimizer": opt_name, "lr": lr,
                "final_acc": final_acc, "steps_to_target": steps_to_target,
            })

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "tier0d_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["optimizer", "lr", "final_acc", "steps_to_target"])
        writer.writeheader()
        writer.writerows(all_results)

    best_cum = min([r for r in all_results if r["optimizer"] == "CUM"], key=lambda r: r["steps_to_target"])
    best_muon = min([r for r in all_results if r["optimizer"] == "Muon"], key=lambda r: r["steps_to_target"])

    passed = best_cum["steps_to_target"] <= best_muon["steps_to_target"]
    print(f"\n{'PASS' if passed else 'FAIL'}: CUM steps ({best_cum['steps_to_target']}) vs Muon ({best_muon['steps_to_target']})")

    return passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
