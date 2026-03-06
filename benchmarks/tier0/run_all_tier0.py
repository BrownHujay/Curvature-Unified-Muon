"""
Run all Tier 0 benchmarks and print pass/fail summary.
Usage: python benchmarks/tier0/run_all_tier0.py
"""

import sys
import os
import time
import ssl

# Fix SSL certs for macOS Python (torchvision dataset downloads)
try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
torch.set_num_threads(4)


def main():
    print("=" * 70)
    print("  CUM Tier 0 Benchmark Suite")
    print("  Running on CPU (Mac-compatible)")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Threads: {torch.get_num_threads()}")
    print("=" * 70)

    results = {}
    total_start = time.perf_counter()

    # Tier 0a: Synthetic Quadratic
    print("\n" + "=" * 70)
    print("  TIER 0a: Synthetic Quadratic")
    print("=" * 70)
    try:
        from benchmarks.tier0.synthetic_quadratic import main as run_0a
        results["0a"] = run_0a()
        if isinstance(results["0a"], tuple):
            results["0a"] = results["0a"][0]
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["0a"] = False

    # Tier 0b: MNIST MLP
    print("\n" + "=" * 70)
    print("  TIER 0b: MNIST MLP")
    print("=" * 70)
    try:
        from benchmarks.tier0.mnist_mlp import main as run_0b
        results["0b"] = run_0b()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["0b"] = False

    # Tier 0c: Micro-Transformer
    print("\n" + "=" * 70)
    print("  TIER 0c: Micro-Transformer on TinyShakespeare")
    print("=" * 70)
    try:
        from benchmarks.tier0.micro_transformer import main as run_0c
        results["0c"] = run_0c()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["0c"] = False

    # Tier 0d: CIFAR-10
    print("\n" + "=" * 70)
    print("  TIER 0d: CIFAR-10 Tiny ConvNet")
    print("=" * 70)
    try:
        from benchmarks.tier0.cifar10_tiny_conv import main as run_0d
        results["0d"] = run_0d()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        results["0d"] = False

    total_time = time.perf_counter() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("  TIER 0 SUMMARY")
    print("=" * 70)
    print(f"{'Test':<35} {'Result':<10}")
    print("-" * 45)
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  Tier {test:<30} {status}")
    print("-" * 45)
    all_passed = all(results.values())
    print(f"  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"  Total time: {total_time/60:.1f} minutes")

    if all_passed:
        print("\n  Ready to proceed to GPU tiers (Tier 1+)")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n  Fix failing tests before scaling up: {', '.join(failed)}")
        if "0c" in failed:
            print("  Tier 0c is the KEY GATE -- CUM must beat Muon on micro-transformer")

    # Save summary
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "tier0_summary.txt")
    with open(summary_path, "w") as f:
        for test, passed in results.items():
            f.write(f"Tier {test}: {'PASS' if passed else 'FAIL'}\n")
        f.write(f"Overall: {'PASS' if all_passed else 'FAIL'}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
