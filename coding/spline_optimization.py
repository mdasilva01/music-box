"""
Spline smoothing optimizer.

Takes in existing cubic spline coefficients (local u in [0,1] per segment)
representing a closed loop, then produces a new set of splines that:
  - minimizes curvature / second derivative (L_smooth)
  - minimizes first derivative / squared speed, roughly encouraging shorter length (L_length)
  - stays faithful to the original curve (L_fit, sampled densely from input)
  - maintains C0+C1 continuity and closure

Each spline uses LOCAL u in [0,1]:
    xs(u) = As*u^3 + Bs*u^2 + Cs*u + Ds
    ys(u) = Es*u^3 + Fs*u^2 + Gs*u + Hs

Input format (list of dicts):
    spline = ({'A':..,'B':..,'C':..,'D':..}, {'E':..,'F':..,'G':..,'H':..})
    splines = [spline, ...]
"""

import numpy as np
import sys
from scipy.optimize import minimize


# ── Pack / unpack ─────────────────────────────────────────────────────────────

def pack(splines: list) -> np.ndarray:
    out = []
    for x_c, y_c in splines:
        out.extend([x_c['A'], x_c['B'], x_c['C'], x_c['D'],
                    y_c['E'], y_c['F'], y_c['G'], y_c['H']])
    return np.array(out, dtype=float)


def unpack(vec: np.ndarray, n: int) -> list:
    splines = []
    for s in range(n):
        A, B, C, D, E, F, G, H = vec[8*s: 8*s+8]
        splines.append(({'A': A, 'B': B, 'C': C, 'D': D},
                        {'E': E, 'F': F, 'G': G, 'H': H}))
    return splines


# ── Evaluation ────────────────────────────────────────────────────────────────

def eval_spline(splines: list, t: float) -> tuple[float, float]:
    """
    Evaluate position at global t in [0, 1).
    Maps to segment index s and local u in [0,1].
    """
    n = len(splines)
    s = min(int(np.floor(t * n)), n - 1)
    u = (t - s / n) * n
    x_c, y_c = splines[s]
    x = x_c['A']*u**3 + x_c['B']*u**2 + x_c['C']*u + x_c['D']
    y = y_c['E']*u**3 + y_c['F']*u**2 + y_c['G']*u + y_c['H']
    return x, y


def sample_reference(ref_splines: list, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Densely sample the reference spline curve at n_samples uniform t values.
    This is the ground truth the optimizer fits against.
    """
    ts = np.linspace(0, 1, n_samples, endpoint=False)
    pts = [eval_spline(ref_splines, t) for t in ts]
    px = np.array([p[0] for p in pts])
    py = np.array([p[1] for p in pts])
    return px, py


# ── Objective terms ───────────────────────────────────────────────────────────

def L_smooth(splines: list) -> float:
    """
    Integral of (x''(t))^2 + (y''(t))^2 over global t in [0,1].

    In local u:
        d^2x/dt^2 = n^2 * d^2x/du^2 = n^2 * (6Au + 2B)

    Because dt = du / n, the exact global-t integral would scale like n^3
    times the local-u second derivative integral.

    For optimization purposes, this implementation keeps the original n-scaling
    used in the supplied code. This still penalizes second derivative magnitude
    consistently across candidate curves with the same number of segments.
    """
    n = len(splines)
    total = 0.0
    for x_c, y_c in splines:
        A, B = x_c['A'], x_c['B']
        E, F = y_c['E'], y_c['F']
        total += n * (
            12*A**2 + 12*A*B + 4*B**2
            + 12*E**2 + 12*E*F + 4*F**2
        )
    return total


def L_length(splines: list) -> float:
    """
    Integral of x'(t)^2 + y'(t)^2 over global t in [0,1].

    This is not exact arc length, which would be:
        integral sqrt(x'(t)^2 + y'(t)^2) dt

    Instead, this minimizes squared speed. It is smoother and easier for SLSQP
    while still discouraging unnecessarily long/high-motion curves.

    Each segment uses local u in [0,1]:
        x(u) = Au^3 + Bu^2 + Cu + D
        y(u) = Eu^3 + Fu^2 + Gu + H

    Since u = nt - s:
        dx/dt = n * dx/du
        dy/dt = n * dy/du
        dt = du / n

    So per segment:
        integral [x'(t)^2 + y'(t)^2] dt
      = n * integral_0^1 [(dx/du)^2 + (dy/du)^2] du

    where:
        dx/du = 3Au^2 + 2Bu + C
        dy/du = 3Eu^2 + 2Fu + G

    Closed-form:
        integral_0^1 (3Au^2 + 2Bu + C)^2 du
      = 9A^2/5 + 3AB + 2AC + 4B^2/3 + 2BC + C^2
    """
    n = len(splines)
    total = 0.0

    for x_c, y_c in splines:
        A, B, C = x_c['A'], x_c['B'], x_c['C']
        E, F, G = y_c['E'], y_c['F'], y_c['G']

        x_term = (
            9*A**2/5
            + 3*A*B
            + 2*A*C
            + 4*B**2/3
            + 2*B*C
            + C**2
        )

        y_term = (
            9*E**2/5
            + 3*E*F
            + 2*E*G
            + 4*F**2/3
            + 2*F*G
            + G**2
        )

        total += n * (x_term + y_term)

    return total


def L_fit(splines: list, px_ref: np.ndarray, py_ref: np.ndarray) -> float:
    """
    Mean squared position error against the densely-sampled reference curve.
    Samples are at t_i = i/N, matching how px_ref/py_ref were generated.
    """
    N = len(px_ref)
    n = len(splines)
    ts = np.linspace(0, 1, N, endpoint=False)
    total = 0.0

    for i, t in enumerate(ts):
        s = min(int(np.floor(t * n)), n - 1)
        u = (t - s / n) * n
        x_c, y_c = splines[s]

        sx = x_c['A']*u**3 + x_c['B']*u**2 + x_c['C']*u + x_c['D']
        sy = y_c['E']*u**3 + y_c['F']*u**2 + y_c['G']*u + y_c['H']

        total += (sx - px_ref[i])**2 + (sy - py_ref[i])**2

    return total / N


# ── Continuity + closure constraints ─────────────────────────────────────────

def continuity_constraints(n: int) -> list:
    """
    C0 + C1 at each interior junction (u=1 of segment s == u=0 of s+1),
    plus C0 + C1 closure (end of last == start of first).
    All in local-u space so constraints are simple linear expressions.
    """
    constraints = []

    def make_c0_x(s):
        def fn(v):
            A1, B1, C1, D1 = v[8*s:8*s+4]
            D2 = v[8*(s+1)+3]
            return (A1 + B1 + C1 + D1) - D2
        return fn

    def make_c0_y(s):
        def fn(v):
            _, _, _, _, E1, F1, G1, H1 = v[8*s:8*(s+1)]
            H2 = v[8*(s+1)+7]
            return (E1 + F1 + G1 + H1) - H2
        return fn

    def make_c1_x(s):
        def fn(v):
            A1, B1, C1, _ = v[8*s:8*s+4]
            C2 = v[8*(s+1)+2]
            return (3*A1 + 2*B1 + C1) - C2
        return fn

    def make_c1_y(s):
        def fn(v):
            _, _, _, _, E1, F1, G1, _ = v[8*s:8*(s+1)]
            G2 = v[8*(s+1)+6]
            return (3*E1 + 2*F1 + G1) - G2
        return fn

    for s in range(n - 1):
        constraints += [
            {'type': 'eq', 'fun': make_c0_x(s)},
            {'type': 'eq', 'fun': make_c0_y(s)},
            {'type': 'eq', 'fun': make_c1_x(s)},
            {'type': 'eq', 'fun': make_c1_y(s)},
        ]

    # Closure: end of last segment == start of first
    def closure_c0_x(v):
        A, B, C, D = v[-8:-4]
        return (A + B + C + D) - v[3]

    def closure_c0_y(v):
        _, _, _, _, E, F, G, H = v[-8:]
        return (E + F + G + H) - v[7]

    def closure_c1_x(v):
        A, B, C, _ = v[-8:-4]
        return (3*A + 2*B + C) - v[2]

    def closure_c1_y(v):
        _, _, _, _, E, F, G, _ = v[-8:]
        return (3*E + 2*F + G) - v[6]

    constraints += [
        {'type': 'eq', 'fun': closure_c0_x},
        {'type': 'eq', 'fun': closure_c0_y},
        {'type': 'eq', 'fun': closure_c1_x},
        {'type': 'eq', 'fun': closure_c1_y},
    ]

    return constraints


# ── Main optimizer ────────────────────────────────────────────────────────────

def optimize_splines(
    input_splines: list[tuple[dict, dict]],
    alpha: float = 0.15,
    beta: float = 0.80,
    gamma: float = 0.05,
    n_samples: int = 1000,
    max_iter: int = 1000,
) -> list[tuple[dict, dict]]:
    """
    Smooth a closed loop of cubic splines while staying faithful to the
    original curve shape.

    Parameters
    ----------
    input_splines : list of (x_dict, y_dict) using local u in [0,1].
                    Must already form a closed loop.
    alpha         : weight on L_smooth, the second-derivative/curvature-like term
    beta          : weight on L_fit, the fidelity-to-original term
    gamma         : weight on L_length, the first-derivative/squared-speed term
    n_samples     : how many points to sample from the reference curve
                    for L_fit. More = more accurate but slower.
    max_iter      : SLSQP iteration limit

    Returns
    -------
    Optimized splines in the same format as input_splines.
    """
    assert abs(alpha + beta + gamma - 1.0) < 1e-9, (
        "alpha + beta + gamma must equal 1."
    )

    n = len(input_splines)
    assert n >= 3, "Need at least 3 splines."

    # Sample the reference curve densely.
    px_ref, py_ref = sample_reference(input_splines, n_samples)

    # Scale derivative losses to be roughly commensurate with L_fit.
    # This keeps the regularizers from dominating purely because of coordinate scale.
    x_range = px_ref.max() - px_ref.min()
    y_range = py_ref.max() - py_ref.min()
    coord_scale = (x_range**2 + y_range**2) / 2.0

    if coord_scale <= 1e-12:
        raise ValueError("Reference curve has near-zero coordinate scale.")

    x0 = pack(input_splines)
    cons = continuity_constraints(n)

    def objective(v):
        spl = unpack(v, n)

        smooth_term = L_smooth(spl) / coord_scale
        fit_term = L_fit(spl, px_ref, py_ref)
        length_term = L_length(spl) / coord_scale

        return (
            alpha * smooth_term
            + beta * fit_term
            + gamma * length_term
        )

    result = minimize(
        objective,
        x0,
        method='SLSQP',
        constraints=cons,
        options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': True},
    )

    if not result.success:
        print(f"[Warning] Optimizer: {result.message}", file=sys.stderr)

    return unpack(result.x, n)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_splines(
    original_splines: list[tuple[dict, dict]],
    optimized_splines: list[tuple[dict, dict]],
    save_path: str | None = None,
    samples_per_spline: int = 300,
) -> None:
    """
    Plot original reference curve and optimized splines on the same axes.
    Original shown as a single gray line; optimized segments in distinct colors.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = cm.tab10.colors
    n = len(optimized_splines)
    n_ref = len(original_splines)

    # Original curve — single dense gray line
    orig_x, orig_y = [], []
    for idx, (x_c, y_c) in enumerate(original_splines):
        us = np.linspace(0, 1, samples_per_spline, endpoint=(idx == n_ref - 1))
        orig_x.extend(x_c['A']*us**3 + x_c['B']*us**2 + x_c['C']*us + x_c['D'])
        orig_y.extend(y_c['E']*us**3 + y_c['F']*us**2 + y_c['G']*us + y_c['H'])

    ax.plot(
        orig_x,
        orig_y,
        color='#aaaaaa',
        linewidth=2,
        zorder=1,
        label='Original splines (reference)',
    )

    # Optimized splines — one color per segment
    knot_xs, knot_ys, knot_colors = [], [], []
    for idx, (x_c, y_c) in enumerate(optimized_splines):
        us = np.linspace(0, 1, samples_per_spline)
        seg_x = x_c['A']*us**3 + x_c['B']*us**2 + x_c['C']*us + x_c['D']
        seg_y = y_c['E']*us**3 + y_c['F']*us**2 + y_c['G']*us + y_c['H']

        color = colors[idx % len(colors)]
        lo, hi = idx / n, (idx + 1) / n

        ax.plot(
            seg_x,
            seg_y,
            color=color,
            linewidth=2.5,
            zorder=2,
            label=f'Opt. spline {idx+1}  t=[{lo:.2f},{hi:.2f})',
        )

        knot_xs.append(x_c['D'])   # u=0 => D
        knot_ys.append(y_c['H'])   # u=0 => H
        knot_colors.append(color)

    # Knot diamonds: start of each optimized segment
    for kx, ky, kc in zip(knot_xs, knot_ys, knot_colors):
        ax.scatter(
            kx,
            ky,
            marker='D',
            s=70,
            color=kc,
            edgecolors='black',
            linewidths=1.0,
            zorder=5,
        )

    ax.scatter(
        [],
        [],
        marker='D',
        s=70,
        color='grey',
        edgecolors='black',
        linewidths=1.0,
        label='Knots',
    )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Original vs optimized splines')
    ax.legend(loc='best', fontsize=7)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description=(
            "Optimize a closed loop of cubic splines for smoothness, length, "
            "and fidelity.\n"
            "Input: JSON file containing a list of spline coefficient dicts.\n"
            "Format: [[{A,B,C,D}, {E,F,G,H}], ...]  (local u in [0,1])"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        help="Path to JSON file with input spline coefficients",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Curvature / second-derivative weight (default 0.15)",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="Fidelity-to-original weight (default 0.80)",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Length / first-derivative weight (default 0.05)",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Reference sample count (default 1000)",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Optimizer iteration limit (default 1000)",
    )

    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Save optimized coefficients to JSON file",
    )

    parser.add_argument(
        "--plot",
        metavar="FILE",
        nargs="?",
        const="__show__",
        help="Show plot interactively or save to FILE",
    )

    args = parser.parse_args()

    # Load input splines from JSON.
    with open(args.input) as f:
        raw = json.load(f)

    input_splines = [
        (
            {
                'A': s[0]['A'],
                'B': s[0]['B'],
                'C': s[0]['C'],
                'D': s[0]['D'],
            },
            {
                'E': s[1]['E'],
                'F': s[1]['F'],
                'G': s[1]['G'],
                'H': s[1]['H'],
            },
        )
        for s in raw
    ]

    optimized = optimize_splines(
        input_splines,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        n_samples=args.samples,
        max_iter=args.max_iter,
    )

    print(f"\nOptimized {len(optimized)} splines (local u in [0,1] per segment):\n")

    for i, (x_c, y_c) in enumerate(optimized):
        s = i + 1
        print(f"Spline {s}  t in [{(s-1)/len(optimized):.4f}, {s/len(optimized):.4f})")
        print(
            f"  x: A={x_c['A']:.6f}  "
            f"B={x_c['B']:.6f}  "
            f"C={x_c['C']:.6f}  "
            f"D={x_c['D']:.6f}"
        )
        print(
            f"  y: E={y_c['E']:.6f}  "
            f"F={y_c['F']:.6f}  "
            f"G={y_c['G']:.6f}  "
            f"H={y_c['H']:.6f}"
        )
        print()

    if args.output:
        out_data = [
            [
                {
                    'A': x_c['A'],
                    'B': x_c['B'],
                    'C': x_c['C'],
                    'D': x_c['D'],
                },
                {
                    'E': y_c['E'],
                    'F': y_c['F'],
                    'G': y_c['G'],
                    'H': y_c['H'],
                },
            ]
            for x_c, y_c in optimized
        ]

        with open(args.output, 'w') as f:
            json.dump(out_data, f, indent=2)

        print(f"Saved to {args.output}")

    if args.plot:
        save = None if args.plot == "__show__" else args.plot
        plot_splines(input_splines, optimized, save_path=save)