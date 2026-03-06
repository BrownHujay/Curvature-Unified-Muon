"""
Tier 0b: Tiny MLP on MNIST
Model: 784 → 128 → 64 → 10 (~110K params)
Target: 97.5% test accuracy within 2000 steps
Runtime: ~1-2 minutes per run
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


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_hidden_2d(self):
        return [self.fc1.weight, self.fc2.weight]

    def get_other(self):
        hidden = set(id(p) for p in self.get_hidden_2d())
        return [p for p in self.parameters() if id(p) not in hidden]


def get_mnist_data():
    """Load MNIST using torchvision."""
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    data_dir = os.path.join(os.path.dirname(__file__), "data", "mnist")
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=0)

    return train_loader, test_loader


def get_lr_schedule(step, total_steps, warmup_steps, base_lr):
    """Linear warmup + cosine decay."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
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


def run_experiment(opt_name, lr, seed=42):
    """Run one experiment, return metrics."""
    torch.manual_seed(seed)

    model = TinyMLP()
    train_loader, test_loader = get_mnist_data()

    total_steps = 2000
    warmup_steps = 100

    hidden_2d = model.get_hidden_2d()
    other = model.get_other()

    if opt_name == "CUM":
        cum_opt = CUM(hidden_2d, lr=lr, ns_steps=3, weight_decay=0.0)
        adam_opt = torch.optim.AdamW(other, lr=3e-4, weight_decay=0.01)
    elif opt_name == "Muon":
        cum_opt = SimpleMuon(hidden_2d, lr=lr, ns_steps=5, weight_decay=0.0)
        adam_opt = torch.optim.AdamW(other, lr=3e-4, weight_decay=0.01)
    elif opt_name == "AdamW":
        cum_opt = None
        adam_opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    elif opt_name == "SGD+Nesterov":
        cum_opt = None
        adam_opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.01)

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

        # LR schedule
        current_lr = get_lr_schedule(step, total_steps, warmup_steps, lr)
        if cum_opt:
            for g in cum_opt.param_groups:
                g["lr"] = current_lr
        for g in adam_opt.param_groups:
            if opt_name in ("AdamW", "SGD+Nesterov"):
                g["lr"] = current_lr

        # Forward + backward
        t0 = time.perf_counter()
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        if cum_opt:
            cum_opt.zero_grad()
        adam_opt.zero_grad()
        loss.backward()
        if cum_opt:
            cum_opt.step()
        adam_opt.step()
        step_time = (time.perf_counter() - t0) * 1000

        # Evaluate periodically
        if step % 100 == 0 or step == total_steps - 1:
            acc = evaluate(model, test_loader)
            metrics.append({
                "step": step,
                "train_loss": loss.item(),
                "test_acc": acc,
                "step_time_ms": step_time,
            })
            if acc >= 0.975 and steps_to_target == total_steps:
                steps_to_target = step

    final_acc = evaluate(model, test_loader)
    return metrics, steps_to_target, final_acc


def main():
    torch.set_num_threads(4)

    print("=" * 60)
    print("Tier 0b: MNIST MLP")
    print("=" * 60)

    lr_configs = {
        "SGD+Nesterov": [0.05, 0.1, 0.5],
        "AdamW": [1e-3, 3e-3, 0.01],
        "Muon": [0.01, 0.02, 0.05],
        "CUM": [0.01, 0.02, 0.05],
    }

    all_results = []
    for opt_name, lrs in lr_configs.items():
        print(f"\n--- {opt_name} ---")
        best_acc = 0
        best_lr = lrs[0]
        best_steps = 2000

        for lr in lrs:
            t0 = time.perf_counter()
            metrics, steps_to_target, final_acc = run_experiment(opt_name, lr)
            elapsed = time.perf_counter() - t0

            print(f"  LR={lr:.4f}: acc={final_acc:.4f}, steps_to_97.5%={steps_to_target}, time={elapsed:.1f}s")

            if final_acc > best_acc:
                best_acc = final_acc
                best_lr = lr
                best_steps = steps_to_target

            all_results.append({
                "optimizer": opt_name,
                "lr": lr,
                "final_acc": final_acc,
                "steps_to_target": steps_to_target,
            })

        print(f"  Best: LR={best_lr}, acc={best_acc:.4f}, steps={best_steps}")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "tier0b_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["optimizer", "lr", "final_acc", "steps_to_target"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to {csv_path}")

    # Pass/fail
    best_cum = min([r for r in all_results if r["optimizer"] == "CUM"], key=lambda r: r["steps_to_target"])
    best_muon = min([r for r in all_results if r["optimizer"] == "Muon"], key=lambda r: r["steps_to_target"])

    passed = best_cum["steps_to_target"] <= best_muon["steps_to_target"]
    print(f"\n{'PASS' if passed else 'FAIL'}: CUM steps ({best_cum['steps_to_target']}) {'<=' if passed else '>'} Muon steps ({best_muon['steps_to_target']})")

    return passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
