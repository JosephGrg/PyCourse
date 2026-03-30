"""
Polynomial Solver — Progressive Implementation
Covers: quadratic solving → numpy → higher degrees → random coefficient generation
"""

import math
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ─────────────────────────────────────────────
# STEP 1 & 2: Solve quadratic, return 2 roots
# ─────────────────────────────────────────────

def solve_quadratic(a, b, c):
    """
    Solve ax² + bx + c = 0 using the quadratic formula.
    Returns (x1, x2) — real or complex roots.
    """
    discriminant = b**2 - 4 * a * c
    if discriminant >= 0:
        x1 = (-b + math.sqrt(discriminant)) / (2 * a)
        x2 = (-b - math.sqrt(discriminant)) / (2 * a)
    else:
        real = -b / (2 * a)
        imag = math.sqrt(-discriminant) / (2 * a)
        x1 = complex(real, imag)
        x2 = complex(real, -imag)
    return x1, x2


# ─────────────────────────────────────────────
# STEP 3: Write parameter file (JSON)
# ─────────────────────────────────────────────

PARAM_FILE = Path("quad_params.json")

def write_param_file(a, b, c, path=PARAM_FILE):
    data = {"a": a, "b": b, "c": c}
    path.write_text(json.dumps(data, indent=2))
    print(f"Parameters written to {path}")


# ─────────────────────────────────────────────
# STEP 4: Read parameter file and solve
# ─────────────────────────────────────────────

def solve_from_file(path=PARAM_FILE):
    data = json.loads(path.read_text())
    a, b, c = data["a"], data["b"], data["c"]
    print(f"\nRead from file: a={a}, b={b}, c={c}")
    x1, x2 = solve_quadratic(a, b, c)
    print(f"  x1 = {x1}")
    print(f"  x2 = {x2}")
    return x1, x2


# ─────────────────────────────────────────────
# STEP 5: Return x1, x2 AND the x-range array
# ─────────────────────────────────────────────

def solve_quadratic_with_range(a, b, c, n_points=400):
    """
    Returns (x1, x2, x_range) where x_range spans [x2, x1] (or symmetric range
    if roots are complex).
    """
    x1, x2 = solve_quadratic(a, b, c)

    if isinstance(x1, complex):
        center = -b / (2 * a)
        half_span = abs(x1.imag) * 3
        x_range = np.linspace(center - half_span, center + half_span, n_points)
    else:
        lo, hi = min(x1, x2), max(x1, x2)
        margin = (hi - lo) * 0.5 or 2.0
        x_range = np.linspace(lo - margin, hi + margin, n_points)

    return x1, x2, x_range


# ─────────────────────────────────────────────
# STEP 6: Also return y values
# ─────────────────────────────────────────────

def solve_quadratic_full(a, b, c, n_points=400):
    """
    Returns (x1, x2, x_values, y_values).
    """
    x1, x2, x_range = solve_quadratic_with_range(a, b, c, n_points)
    y_values = a * x_range**2 + b * x_range + c
    return x1, x2, x_range, y_values


# ─────────────────────────────────────────────
# STEP 7: Save x, y to a two-column file
# ─────────────────────────────────────────────

def save_xy(x_values, y_values, path="xy_data.txt"):
    data = np.column_stack([x_values, y_values])
    np.savetxt(path, data, header="x\ty", fmt="%.6f", delimiter="\t")
    print(f"\nX/Y data saved to {path}")


# ─────────────────────────────────────────────
# STEP 8 & 9: Plot with axes, legend, title, etc.
# ─────────────────────────────────────────────

def plot_quadratic(a, b, c, x_values, y_values, x1, x2, ax=None, label=None, color="royalblue"):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))

    lbl = label or f"{a:+g}x² {b:+g}x {c:+g}"
    ax.plot(x_values, y_values, color=color, linewidth=2, label=lbl)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Mark roots if real
    if not isinstance(x1, complex):
        ax.scatter([x1, x2], [0, 0], color="crimson", zorder=5,
                   label=f"Roots: x₁={x1:.3f}, x₂={x2:.3f}")

    if standalone:
        ax.set_title(f"Quadratic: {a:+g}x² {b:+g}x {c:+g}", fontsize=14, fontweight="bold")
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("f(x)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("quadratic_plot.png", dpi=150)
        print("Plot saved to quadratic_plot.png")
        plt.show()


# ─────────────────────────────────────────────
# STEP 10: Replace manual math with numpy
# ─────────────────────────────────────────────

def solve_polynomial_numpy(coeffs, x_lo=None, x_hi=None, n_points=500):
    """
    Solve a polynomial of ANY degree using numpy.
    coeffs = [a_n, a_{n-1}, ..., a_1, a_0]  (highest degree first, numpy convention)

    Returns (roots, x_values, y_values)
    """
    roots = np.roots(coeffs)          # complex array of all roots
    real_roots = roots[np.isreal(roots)].real

    poly = np.poly1d(coeffs)

    # Determine plot range
    if x_lo is None or x_hi is None:
        if len(real_roots) >= 2:
            lo, hi = real_roots.min(), real_roots.max()
            margin = max((hi - lo) * 0.4, 1.0)
        elif len(real_roots) == 1:
            lo = hi = real_roots[0]
            margin = 3.0
        else:
            center = -coeffs[-2] / (2 * coeffs[0]) if len(coeffs) >= 2 else 0
            lo = hi = center
            margin = 3.0
        x_lo = lo - margin
        x_hi = hi + margin

    x_values = np.linspace(x_lo, x_hi, n_points)
    y_values = poly(x_values)
    return roots, x_values, y_values


def degree_label(n):
    return {1: "Linear", 2: "Quadratic", 3: "Cubic", 4: "Quartic",
            5: "Quintic"}.get(n, f"Degree-{n}")


def plot_polynomial(coeffs, roots, x_values, y_values, ax=None, color="royalblue", label=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 6))

    degree = len(coeffs) - 1
    poly_str = np.poly1d(coeffs).__str__().replace("\n", "")
    lbl = label or f"p(x) = {poly_str}"

    ax.plot(x_values, y_values, color=color, linewidth=2.2, label=lbl)

    real_roots = roots[np.isreal(roots)].real
    if len(real_roots):
        ax.scatter(real_roots, np.zeros_like(real_roots), zorder=6,
                   color="crimson", s=70, label=f"Real roots ({len(real_roots)})")

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)

    if standalone:
        ax.set_title(f"{degree_label(degree)} Polynomial\np(x) = {poly_str}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("p(x)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("polynomial_plot.png", dpi=150)
        print("Plot saved to polynomial_plot.png")
        plt.show()


# ─────────────────────────────────────────────
# STEP 11: Random coefficient generator
# ─────────────────────────────────────────────

def random_polynomial_coeffs(degree=None, lo=-5, hi=5, seed=None):
    """
    Generate a random polynomial of random degree (2–5 if not specified).
    Returns list of coefficients [a_n, ..., a_0] (numpy convention).
    Leading coefficient is guaranteed nonzero.
    """
    rng = random.Random(seed)
    if degree is None:
        degree = rng.randint(2, 5)
    coeffs = [rng.randint(lo, hi) or 1 for _ in range(degree + 1)]
    coeffs[0] = coeffs[0] or 1   # ensure leading coeff != 0
    return coeffs


# ─────────────────────────────────────────────
# MAIN — run all steps sequentially
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  STEP 1–4: Manual quadratic solver + parameter file")
    print("=" * 60)

    a, b, c = 1, -5, 6          # roots: x=2 and x=3
    write_param_file(a, b, c)
    x1, x2 = solve_from_file()

    print("\n" + "=" * 60)
    print("  STEP 5–9: Full quadratic: range, y-values, save, plot")
    print("=" * 60)

    x1, x2, xs, ys = solve_quadratic_full(a, b, c)
    save_xy(xs, ys)
    plot_quadratic(a, b, c, xs, ys, x1, x2)

    print("\n" + "=" * 60)
    print("  STEP 10: numpy-based polynomial solver (degree 2)")
    print("=" * 60)

    coeffs_quad = [a, b, c]
    roots, xs, ys = solve_polynomial_numpy(coeffs_quad)
    print(f"  Roots: {roots}")
    plot_polynomial(coeffs_quad, roots, xs, ys)

    print("\n" + "=" * 60)
    print("  STEP 10b: Higher-degree example (cubic)")
    print("=" * 60)

    coeffs_cubic = [1, -6, 11, -6]    # (x-1)(x-2)(x-3)
    roots, xs, ys = solve_polynomial_numpy(coeffs_cubic)
    print(f"  Roots: {roots}")
    plot_polynomial(coeffs_cubic, roots, xs, ys)

    print("\n" + "=" * 60)
    print("  STEP 11: Random polynomial coefficient generator")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Two Random Polynomials", fontsize=16, fontweight="bold")

    colors = ["steelblue", "darkorange"]
    for i, ax in enumerate(axes):
        seed = 42 + i * 7
        coeffs = random_polynomial_coeffs(seed=seed)
        degree = len(coeffs) - 1
        print(f"\n  Polynomial {i+1} (seed={seed}): degree={degree}, coeffs={coeffs}")

        roots, xs, ys = solve_polynomial_numpy(coeffs)
        print(f"    Roots: {roots}")

        # Save per-polynomial data
        np.savetxt(f"random_poly_{i+1}_xy.txt",
                   np.column_stack([xs, ys]),
                   header="x\ty", fmt="%.6f", delimiter="\t")

        plot_polynomial(coeffs, roots, xs, ys, ax=ax, color=colors[i],
                        label=f"p{i+1}(x) — degree {degree}")
        ax.set_title(f"Random Polynomial {i+1}\ncoeffs = {coeffs}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("p(x)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("random_polynomials.png", dpi=150)
    print("\nCombined random-polynomial plot saved to random_polynomials.png")
    plt.show()

    print("\n✓ All steps complete.")
