"""
Offline theoretical analysis of NS polynomial dynamics.

Computes:
1. Lyapunov exponents across derivative parameter space
2. Period-2 orbits and their stability
3. Invariant measures via Ulam's method
4. Minimax-optimal polynomial for iterated target 0.88
5. Chebyshev (QSVT) reference coefficients for sign function

All scalar math — no GPU needed.
"""

import numpy as np
from scipy.optimize import brentq, minimize, differential_evolution
from scipy.linalg import eig
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Polynomial helpers
# =============================================================================

def bifurcation_coeffs(deriv, sigma_star=0.868, c=2.0315):
    """Compute (a, b, c) from target derivative at fixed point."""
    s2 = sigma_star ** 2
    s4 = sigma_star ** 4
    a = (3 - deriv) / 2 + c * s4
    b = (deriv - 1 - 4 * c * s4) / (2 * s2)
    return a, b, c


def p(sigma, a, b, c):
    """Single NS polynomial evaluation: p(σ) = aσ + bσ³ + cσ⁵."""
    s2 = sigma * sigma
    return sigma * (a + s2 * (b + c * s2))


def p_prime(sigma, a, b, c):
    """Derivative: p'(σ) = a + 3bσ² + 5cσ⁴."""
    s2 = sigma * sigma
    return a + 3 * b * s2 + 5 * c * s2 * s2


def iterate_p(sigma, n, a, b, c):
    """Apply p n times."""
    x = sigma
    for _ in range(n):
        x = p(x, a, b, c)
    return x


# =============================================================================
# 1. Lyapunov Exponents
# =============================================================================

def compute_lyapunov(deriv, n_iter=5000, n_discard=1000, n_points=200):
    """
    Compute Lyapunov exponent for NS polynomial with given derivative.

    λ = lim (1/N) Σ log|p'(x_k)|

    λ < 0: attracting fixed point or cycle
    λ = 0: marginally stable (bifurcation boundary)
    λ > 0: chaotic
    """
    a, b, c = bifurcation_coeffs(deriv)

    # Grid of starting points in (0, 1)
    sigmas = np.linspace(0.05, 0.99, n_points)
    lyapunovs = []

    for s0 in sigmas:
        x = s0
        log_derivs = []

        for i in range(n_iter):
            dp = p_prime(x, a, b, c)
            if abs(dp) < 1e-30:
                dp = 1e-30
            if i >= n_discard:
                log_derivs.append(np.log(abs(dp)))
            x = p(x, a, b, c)

            # Check for divergence or collapse
            if abs(x) > 1e6 or np.isnan(x):
                break

        if len(log_derivs) > 100:
            lyapunovs.append(np.mean(log_derivs))

    if not lyapunovs:
        return float('nan'), float('nan')

    return np.mean(lyapunovs), np.std(lyapunovs)


def lyapunov_sweep():
    """Sweep Lyapunov exponent across derivative values."""
    print("=" * 70)
    print("PART 1: LYAPUNOV EXPONENTS")
    print("=" * 70)
    print()
    print("λ < 0: attracting (convergent)")
    print("λ ≈ 0: edge of chaos (bifurcation boundary)")
    print("λ > 0: chaotic")
    print()

    derivs = [-0.5, -0.8, -0.9, -1.0, -1.1, -1.2, -1.4, -1.58, -1.8, -2.0, -2.5, -2.8, -3.0]

    print(f"{'deriv':>8} {'|p*(σ*)|':>10} {'a':>8} {'b':>9} {'c':>8} {'Lyapunov':>10} {'±std':>8} {'Regime':>15}")
    print("-" * 85)

    results = []
    for d in derivs:
        a, b, c = bifurcation_coeffs(d)
        lyap_mean, lyap_std = compute_lyapunov(d)

        if np.isnan(lyap_mean):
            regime = "DIVERGENT"
        elif lyap_mean < -0.1:
            regime = "attracting"
        elif lyap_mean < 0.05:
            regime = "edge of chaos"
        elif lyap_mean < 0.5:
            regime = "weakly chaotic"
        else:
            regime = "CHAOTIC"

        results.append((d, abs(d), a, b, c, lyap_mean, lyap_std, regime))
        print(f"{d:>8.2f} {abs(d):>10.2f} {a:>8.4f} {b:>9.4f} {c:>8.4f} {lyap_mean:>10.4f} {lyap_std:>8.4f} {regime:>15}")

    print()
    # Find the edge of chaos
    for i in range(len(results) - 1):
        if results[i][5] < 0 and results[i+1][5] > 0:
            print(f">> Edge of chaos between d={results[i][0]:.2f} (λ={results[i][5]:.4f}) "
                  f"and d={results[i+1][0]:.2f} (λ={results[i+1][5]:.4f})")

    return results


# =============================================================================
# 2. Period-2 Orbits
# =============================================================================

def find_period2_orbits(deriv, n_search=1000):
    """
    Find period-2 orbits: points where p(p(σ)) = σ but p(σ) ≠ σ.
    Also compute their stability: |p'(a) · p'(b)| for orbit {a, b}.
    """
    a, b, c = bifurcation_coeffs(deriv)

    # p(p(σ)) - σ = 0, but NOT p(σ) - σ = 0
    def pp_minus_sigma(s):
        return p(p(s, a, b, c), a, b, c) - s

    def p_minus_sigma(s):
        return p(s, a, b, c) - s

    # Search for roots of p(p(σ)) = σ
    sigmas = np.linspace(0.01, 1.5, n_search)
    vals = [pp_minus_sigma(s) for s in sigmas]

    roots = []
    for i in range(len(vals) - 1):
        if vals[i] * vals[i+1] < 0:
            try:
                root = brentq(pp_minus_sigma, sigmas[i], sigmas[i+1])
                # Check it's not a fixed point
                if abs(p_minus_sigma(root)) > 1e-6:
                    roots.append(root)
            except:
                pass

    # Deduplicate
    unique_roots = []
    for r in roots:
        if not any(abs(r - u) < 1e-6 for u in unique_roots):
            unique_roots.append(r)

    # Find orbit pairs
    orbits = []
    used = set()
    for r in unique_roots:
        if r in used:
            continue
        partner = p(r, a, b, c)
        # Verify
        back = p(partner, a, b, c)
        if abs(back - r) < 1e-6:
            stability = abs(p_prime(r, a, b, c) * p_prime(partner, a, b, c))
            orbits.append((min(r, partner), max(r, partner), stability))
            used.add(r)
            # Find the closest unique root to partner
            for u in unique_roots:
                if abs(u - partner) < 1e-4:
                    used.add(u)

    return orbits


def period2_analysis():
    """Analyze period-2 orbits across derivative values."""
    print()
    print("=" * 70)
    print("PART 2: PERIOD-2 ORBITS")
    print("=" * 70)
    print()
    print("Period-2: p(a)=b, p(b)=a, a≠b")
    print("Stable if |p'(a)·p'(b)| < 1")
    print()

    derivs = [-0.8, -0.9, -1.0, -1.1, -1.2, -1.4, -1.58, -1.8, -2.0, -2.5, -2.8]

    print(f"{'deriv':>8} {'σ_low':>8} {'σ_high':>8} {'|p*(a)·p*(b)|':>15} {'Stable?':>10}")
    print("-" * 55)

    for d in derivs:
        orbits = find_period2_orbits(d)
        if not orbits:
            print(f"{d:>8.2f}    (no period-2 orbit found — likely convergent)")
        else:
            for (lo, hi, stab) in orbits:
                stable = "YES" if stab < 1 else "NO"
                print(f"{d:>8.2f} {lo:>8.4f} {hi:>8.4f} {stab:>15.6f} {stable:>10}")

    # Also find fixed points
    print()
    print("Fixed points p(σ*) = σ* and their stability p'(σ*):")
    print(f"{'deriv':>8} {'σ*':>8} {'p*(σ*)':>10} {'|p*(σ*)|':>10} {'Stable?':>10}")
    print("-" * 50)

    for d in derivs:
        a, b, c = bifurcation_coeffs(d)
        # Fixed points: p(σ) = σ → (a-1)σ + bσ³ + cσ⁵ = 0 → σ[(a-1) + bσ² + cσ⁴] = 0
        # σ=0 is trivial. For positive fixed points, solve (a-1) + bσ² + cσ⁴ = 0
        # This is quadratic in σ²: cσ⁴ + bσ² + (a-1) = 0
        disc = b*b - 4*c*(a-1)
        if disc >= 0:
            s2_1 = (-b + np.sqrt(disc)) / (2*c)
            s2_2 = (-b - np.sqrt(disc)) / (2*c)
            for s2 in [s2_1, s2_2]:
                if s2 > 0:
                    sigma = np.sqrt(s2)
                    dp = p_prime(sigma, a, b, c)
                    stable = "YES" if abs(dp) < 1 else "NO"
                    print(f"{d:>8.2f} {sigma:>8.4f} {dp:>10.4f} {abs(dp):>10.4f} {stable:>10}")

    return


# =============================================================================
# 3. Invariant Measure (Ulam's Method)
# =============================================================================

def ulam_invariant_measure(deriv, K=500, interval=(0.01, 1.5)):
    """
    Compute invariant measure of x_{n+1} = p(x_n) via Ulam's method.

    Discretize interval into K bins, build transition matrix,
    find stationary distribution.
    """
    a, b, c = bifurcation_coeffs(deriv)
    lo, hi = interval
    edges = np.linspace(lo, hi, K + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = edges[1] - edges[0]

    # Build transition matrix T: T[i,j] = prob of going from bin j to bin i
    # Use fine sampling within each bin
    n_samples = 20  # samples per bin
    T = np.zeros((K, K))

    for j in range(K):
        samples = np.linspace(edges[j] + dx/n_samples/2, edges[j+1] - dx/n_samples/2, n_samples)
        for s in samples:
            image = p(s, a, b, c)
            # Which bin does it land in?
            if lo <= image <= hi:
                bin_idx = int((image - lo) / dx)
                bin_idx = min(bin_idx, K - 1)
                T[bin_idx, j] += 1.0 / n_samples

    # Normalize columns to get transition probabilities
    col_sums = T.sum(axis=0)
    col_sums[col_sums == 0] = 1
    T = T / col_sums

    # Find stationary distribution (left eigenvector with eigenvalue 1)
    eigenvalues, eigenvectors = eig(T, left=True, right=False)

    # Find eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, idx])

    # Normalize to probability
    stationary = np.abs(stationary)
    if stationary.sum() > 0:
        stationary = stationary / stationary.sum()

    return centers, stationary


def invariant_measure_analysis():
    """Compute and report invariant measures for key derivative values."""
    print()
    print("=" * 70)
    print("PART 3: INVARIANT MEASURES (ULAM'S METHOD)")
    print("=" * 70)
    print()

    derivs_to_analyze = [-1.0, -1.4, -1.58, -2.0, -2.8]

    for d in derivs_to_analyze:
        centers, measure = ulam_invariant_measure(d)

        # Find peaks (where measure is concentrated)
        threshold = measure.max() * 0.1
        peak_regions = centers[measure > threshold]

        # Compute statistics
        mean_sv = np.sum(centers * measure)
        var_sv = np.sum((centers - mean_sv)**2 * measure)
        std_sv = np.sqrt(var_sv)

        # Find mode(s)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(measure, height=measure.max() * 0.2, distance=20)
        peak_vals = centers[peaks] if len(peaks) > 0 else [centers[np.argmax(measure)]]

        a, b, c = bifurcation_coeffs(d)
        print(f"deriv = {d:.2f} (a={a:.4f}, b={b:.4f}, c={c:.4f}):")
        print(f"  Mean SV after convergence: {mean_sv:.4f}")
        print(f"  Std of SV distribution:    {std_sv:.4f}")
        print(f"  Mode(s):                   {', '.join(f'{v:.4f}' for v in peak_vals)}")
        print(f"  Support range:             [{peak_regions.min():.4f}, {peak_regions.max():.4f}]")

        # Report as histogram-style
        n_bins_report = 10
        bin_edges = np.linspace(peak_regions.min() - 0.05, peak_regions.max() + 0.05, n_bins_report + 1)
        print(f"  Distribution (coarse):")
        for i in range(n_bins_report):
            mask = (centers >= bin_edges[i]) & (centers < bin_edges[i+1])
            mass = measure[mask].sum()
            bar = '█' * int(mass * 200)
            if mass > 0.005:
                print(f"    [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}): {mass:.3f} {bar}")
        print()

    return


# =============================================================================
# 4. Minimax-Optimal Polynomial for Iterated Target 0.88
# =============================================================================

def iterated_max_error(params, target=0.88, n_iter=5, n_points=500):
    """Compute max |p^n(σ) - target| over σ ∈ [0.05, 1.0]."""
    a, b, c = params
    sigmas = np.linspace(0.05, 1.0, n_points)

    max_err = 0
    for s0 in sigmas:
        x = s0
        for _ in range(n_iter):
            x = p(x, a, b, c)
            if abs(x) > 1e6 or np.isnan(x):
                return 1e6
        err = abs(x - target)
        max_err = max(max_err, err)

    return max_err


def iterated_mean_error(params, target=0.88, n_iter=5, n_points=500):
    """Compute mean |p^n(σ) - target|² over σ ∈ [0.05, 1.0]."""
    a, b, c = params
    sigmas = np.linspace(0.05, 1.0, n_points)

    total = 0
    for s0 in sigmas:
        x = s0
        for _ in range(n_iter):
            x = p(x, a, b, c)
            if abs(x) > 1e6 or np.isnan(x):
                return 1e6
        total += (x - target) ** 2

    return total / n_points


def iterated_sv_variance(params, n_iter=5, n_points=500):
    """Compute variance of p^n(σ) over σ ∈ [0.05, 1.0] — measures equalization quality."""
    a, b, c = params
    sigmas = np.linspace(0.05, 1.0, n_points)

    outputs = []
    for s0 in sigmas:
        x = s0
        for _ in range(n_iter):
            x = p(x, a, b, c)
            if abs(x) > 1e6 or np.isnan(x):
                return 1e6
        outputs.append(x)

    outputs = np.array(outputs)
    return np.var(outputs)


def minimax_polynomial_search():
    """Find optimal polynomial coefficients for different objectives."""
    print()
    print("=" * 70)
    print("PART 4: MINIMAX-OPTIMAL POLYNOMIAL SEARCH")
    print("=" * 70)
    print()

    # Reference: standard Muon
    std_abc = (3.4445, -4.7750, 2.0315)
    print(f"Reference — Standard Muon (a={std_abc[0]}, b={std_abc[1]}, c={std_abc[2]}):")
    print(f"  Max |p⁵(σ) - 0.88|:  {iterated_max_error(std_abc):.6f}")
    print(f"  Mean |p⁵(σ) - 0.88|²: {iterated_mean_error(std_abc):.6f}")
    print(f"  Var[p⁵(σ)]:           {iterated_sv_variance(std_abc):.6f}")
    print()

    # Reference: our best d=-1.0
    d10_abc = bifurcation_coeffs(-1.0)
    print(f"Reference — d=-1.0 (a={d10_abc[0]:.4f}, b={d10_abc[1]:.4f}, c={d10_abc[2]:.4f}):")
    print(f"  Max |p⁵(σ) - 0.88|:  {iterated_max_error(d10_abc):.6f}")
    print(f"  Mean |p⁵(σ) - 0.88|²: {iterated_mean_error(d10_abc):.6f}")
    print(f"  Var[p⁵(σ)]:           {iterated_sv_variance(d10_abc):.6f}")
    print()

    # Objective 1: Minimize max error to 0.88
    print("Optimizing: min max |p⁵(σ) - 0.88| (Chebyshev/minimax)...")
    result_minimax = differential_evolution(
        iterated_max_error,
        bounds=[(0.5, 5.0), (-8.0, 0.0), (0.5, 4.0)],
        seed=42, maxiter=500, tol=1e-8,
        args=(0.88, 5, 300),
    )
    abc_minimax = tuple(result_minimax.x)
    print(f"  Optimal: a={abc_minimax[0]:.4f}, b={abc_minimax[1]:.4f}, c={abc_minimax[2]:.4f}")
    print(f"  Max |p⁵(σ) - 0.88|:  {iterated_max_error(abc_minimax):.6f}")
    print(f"  Var[p⁵(σ)]:           {iterated_sv_variance(abc_minimax):.6f}")

    # What's the fixed point and derivative?
    a, b, c = abc_minimax
    # Find fixed point numerically
    from scipy.optimize import brentq as _brentq
    try:
        fp = _brentq(lambda s: p(s, a, b, c) - s, 0.1, 1.5)
        dp = p_prime(fp, a, b, c)
        print(f"  Fixed point: σ*={fp:.4f}, p'(σ*)={dp:.4f}")
    except:
        print(f"  Could not find fixed point")
    print()

    # Objective 2: Minimize variance of p⁵ (best equalization)
    print("Optimizing: min Var[p⁵(σ)] (best equalization)...")
    result_var = differential_evolution(
        iterated_sv_variance,
        bounds=[(0.5, 5.0), (-8.0, 0.0), (0.5, 4.0)],
        seed=42, maxiter=500, tol=1e-10,
    )
    abc_var = tuple(result_var.x)
    print(f"  Optimal: a={abc_var[0]:.4f}, b={abc_var[1]:.4f}, c={abc_var[2]:.4f}")
    print(f"  Var[p⁵(σ)]:           {iterated_sv_variance(abc_var):.6f}")
    print(f"  Max |p⁵(σ) - target|: {iterated_max_error(abc_var):.6f}")

    # Check what the output mean is
    outputs = []
    for s0 in np.linspace(0.05, 1.0, 500):
        x = s0
        for _ in range(5):
            x = p(x, abc_var[0], abc_var[1], abc_var[2])
        outputs.append(x)
    mean_out = np.mean(outputs)
    print(f"  Mean p⁵(σ):            {mean_out:.4f}")

    try:
        fp = _brentq(lambda s: p(s, abc_var[0], abc_var[1], abc_var[2]) - s, 0.1, 1.5)
        dp = p_prime(fp, abc_var[0], abc_var[1], abc_var[2])
        print(f"  Fixed point: σ*={fp:.4f}, p'(σ*)={dp:.4f}")
    except:
        print(f"  Could not find fixed point")
    print()

    # Objective 3: Minimize mean squared error to 0.88 (L2)
    print("Optimizing: min E[|p⁵(σ) - 0.88|²] (L2/MSE)...")
    result_mse = differential_evolution(
        iterated_mean_error,
        bounds=[(0.5, 5.0), (-8.0, 0.0), (0.5, 4.0)],
        seed=42, maxiter=500, tol=1e-10,
        args=(0.88, 5, 300),
    )
    abc_mse = tuple(result_mse.x)
    print(f"  Optimal: a={abc_mse[0]:.4f}, b={abc_mse[1]:.4f}, c={abc_mse[2]:.4f}")
    print(f"  Mean |p⁵(σ) - 0.88|²: {iterated_mean_error(abc_mse):.6f}")
    print(f"  Var[p⁵(σ)]:           {iterated_sv_variance(abc_mse):.6f}")

    try:
        fp = _brentq(lambda s: p(s, abc_mse[0], abc_mse[1], abc_mse[2]) - s, 0.1, 1.5)
        dp = p_prime(fp, abc_mse[0], abc_mse[1], abc_mse[2])
        print(f"  Fixed point: σ*={fp:.4f}, p'(σ*)={dp:.4f}")
    except:
        print(f"  Could not find fixed point")

    return abc_minimax, abc_var, abc_mse


# =============================================================================
# 5. Chebyshev / QSVT Reference
# =============================================================================

def chebyshev_sign_coefficients():
    """
    Compute truncated Chebyshev series for sign(x) using odd terms only.
    sign(x) ≈ (4/π)[T₁(x) - T₃(x)/3 + T₅(x)/5 - ...]

    For degree-5 odd polynomial p(x) = αx + βx³ + γx⁵:
    Convert from Chebyshev to monomial basis.
    """
    print()
    print("=" * 70)
    print("PART 5: CHEBYSHEV / QSVT REFERENCE COEFFICIENTS")
    print("=" * 70)
    print()

    # Chebyshev polynomials (odd ones):
    # T₁(x) = x
    # T₃(x) = 4x³ - 3x
    # T₅(x) = 16x⁵ - 20x³ + 5x

    # sign(x) ≈ (4/π) Σ_{k=0}^{n} (-1)^k T_{2k+1}(x) / (2k+1)
    # For n=2 (terms T₁, T₃, T₅):

    # Coefficient of T₁: (4/π) * 1/1 = 4/π
    # Coefficient of T₃: (4/π) * (-1)/3 = -4/(3π)
    # Coefficient of T₅: (4/π) * 1/5 = 4/(5π)

    c1 = 4 / np.pi
    c3 = -4 / (3 * np.pi)
    c5 = 4 / (5 * np.pi)

    print(f"Chebyshev coefficients: c₁={c1:.6f}, c₃={c3:.6f}, c₅={c5:.6f}")

    # Convert to monomial basis p(x) = ax + bx³ + cx⁵
    # T₁(x) = x → contributes c1 to x coefficient
    # T₃(x) = 4x³ - 3x → contributes -3·c3 to x, 4·c3 to x³
    # T₅(x) = 16x⁵ - 20x³ + 5x → contributes 5·c5 to x, -20·c5 to x³, 16·c5 to x⁵

    a_cheb = c1 + (-3) * c3 + 5 * c5
    b_cheb = 4 * c3 + (-20) * c5
    c_cheb = 16 * c5

    print(f"Monomial basis: p(x) = {a_cheb:.6f}x + {b_cheb:.6f}x³ + {c_cheb:.6f}x⁵")
    print()

    # Compare to NS quintic
    print("Comparison:")
    print(f"  {'':>20} {'a':>10} {'b':>10} {'c':>10}")
    print(f"  {'Chebyshev sign':>20} {a_cheb:>10.4f} {b_cheb:>10.4f} {c_cheb:>10.4f}")
    print(f"  {'Standard NS':>20} {3.4445:>10.4f} {-4.7750:>10.4f} {2.0315:>10.4f}")

    d10_abc = bifurcation_coeffs(-1.0)
    print(f"  {'d=-1.0':>20} {d10_abc[0]:>10.4f} {d10_abc[1]:>10.4f} {d10_abc[2]:>10.4f}")

    d14_abc = bifurcation_coeffs(-1.4)
    print(f"  {'d=-1.4':>20} {d14_abc[0]:>10.4f} {d14_abc[1]:>10.4f} {d14_abc[2]:>10.4f}")
    print()

    # Evaluate Chebyshev polynomial performance
    print(f"Chebyshev polynomial iterated performance:")
    cheb_abc = (a_cheb, b_cheb, c_cheb)
    print(f"  Max |p⁵(σ) - 0.88|:  {iterated_max_error(cheb_abc):.6f}")
    print(f"  Var[p⁵(σ)]:           {iterated_sv_variance(cheb_abc):.6f}")

    # What does p⁵ actually converge to?
    outputs = []
    for s0 in np.linspace(0.05, 1.0, 500):
        x = s0
        for _ in range(5):
            x = p(x, a_cheb, b_cheb, c_cheb)
        outputs.append(x)
    outputs = np.array(outputs)
    print(f"  Mean p⁵(σ):            {np.mean(outputs):.4f}")
    print(f"  Std p⁵(σ):             {np.std(outputs):.4f}")
    print(f"  Range p⁵(σ):           [{np.min(outputs):.4f}, {np.max(outputs):.4f}]")

    # Fixed point?
    try:
        fp = brentq(lambda s: p(s, a_cheb, b_cheb, c_cheb) - s, 0.1, 1.2)
        dp = p_prime(fp, a_cheb, b_cheb, c_cheb)
        print(f"  Fixed point: σ*={fp:.4f}, p'(σ*)={dp:.4f}")
    except:
        print(f"  No fixed point in [0.1, 1.2]")

    return cheb_abc


# =============================================================================
# 6. SV Mapping Visualization (text-based)
# =============================================================================

def sv_mapping_comparison():
    """Compare how different polynomials map input SVs after 5 iterations."""
    print()
    print("=" * 70)
    print("PART 6: SV MAPPING COMPARISON (p⁵(σ) for σ ∈ [0.05, 1.0])")
    print("=" * 70)
    print()

    configs = {
        'Standard NS': (3.4445, -4.7750, 2.0315),
        'd=-1.0': bifurcation_coeffs(-1.0),
        'd=-1.4': bifurcation_coeffs(-1.4),
    }

    sigmas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.868, 0.9, 0.95, 1.0]

    header = f"{'σ_in':>6}"
    for name in configs:
        header += f"  {name:>12}"
    print(header)
    print("-" * (6 + 14 * len(configs)))

    for s0 in sigmas:
        row = f"{s0:>6.3f}"
        for name, (a, b, c) in configs.items():
            x = s0
            for _ in range(5):
                x = p(x, a, b, c)
            row += f"  {x:>12.4f}"
        print(row)

    print()
    # Also show the variance
    print("Equalization quality (Var[p⁵(σ)] over σ∈[0.05,1]):")
    for name, abc in configs.items():
        var = iterated_sv_variance(abc)
        print(f"  {name:>15}: {var:.6f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    lyap_results = lyapunov_sweep()
    period2_analysis()
    invariant_measure_analysis()
    abc_minimax, abc_var, abc_mse = minimax_polynomial_search()
    cheb_abc = chebyshev_sign_coefficients()
    sv_mapping_comparison()

    print()
    print("=" * 70)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 70)
    print()
    print("See analysis above for detailed results.")
    print("Key questions answered:")
    print("  1. Where exactly is the edge of chaos in derivative space?")
    print("  2. What are the period-2 orbits and are they stable?")
    print("  3. What does the invariant measure look like at different oscillation levels?")
    print("  4. Can we find a polynomial that equalizes better than standard NS?")
    print("  5. How does the Chebyshev-optimal sign polynomial compare?")
