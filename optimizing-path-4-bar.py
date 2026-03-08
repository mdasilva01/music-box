import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class Mechanism:
    """
    Parameter vector:
        x = [r1, r2, r3, r4, px, py, theta0]

    Geometry:
        A = (0, 0)
        D = (r1, 0)
        B = crank endpoint
        C = rocker-coupler joint

    Coupler point P is attached rigidly to the coupler BC using local coordinates
    (px, py), where the local x-axis points from B to C.
    """
    def __init__(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (7,):
            raise ValueError("x must have shape (7,)")

        self.x = x.copy()
        self.r1, self.r2, self.r3, self.r4, self.px, self.py, self.theta0 = x

        self.A = np.array([0.0, 0.0], dtype=float)
        self.D = np.array([self.r1, 0.0], dtype=float)

        self._validate()

    def _validate(self):
        lengths = np.array([self.r1, self.r2, self.r3, self.r4], dtype=float)
        if np.any(lengths <= 0):
            raise ValueError("All link lengths must be positive.")

        # Strict Grashof condition with a small margin
        links = np.sort(lengths)
        s, p, q, l = links
        if not (s + l < p + q - 1e-8):
            raise ValueError("Not strictly Grashof.")

    def _B_at(self, theta):
        ang = theta + self.theta0
        return np.array(
            [self.r2 * np.cos(ang), self.r2 * np.sin(ang)],
            dtype=float
        )

    def _C_candidates(self, theta):
        B = self._B_at(theta)
        D = self.D

        BD_vec = D - B
        d = float(np.linalg.norm(BD_vec))
        if d <= 1e-12:
            raise ValueError("Degenerate B-D distance.")

        # Circle intersection feasibility
        if not (abs(self.r3 - self.r4) <= d <= self.r3 + self.r4):
            raise ValueError("No real closure at this theta.")

        a = (self.r3**2 - self.r4**2 + d**2) / (2.0 * d)
        h_sq = self.r3**2 - a**2
        if h_sq < -1e-10:
            raise ValueError("Negative h^2.")
        h_sq = max(h_sq, 0.0)
        h = np.sqrt(h_sq)

        e = BD_vec / d
        M = B + a * e
        n = np.array([-e[1], e[0]], dtype=float)

        C1 = M + h * n
        C2 = M - h * n
        return B, C1, C2

    def point_at(self, theta, prev_C=None):
        B, C1, C2 = self._C_candidates(theta)

        if prev_C is None:
            C = C1 if C1[1] >= C2[1] else C2
        else:
            d1 = np.linalg.norm(prev_C - C1)
            d2 = np.linalg.norm(prev_C - C2)
            C = C1 if d1 <= d2 else C2

        # Local frame on coupler:
        # ex = unit vector from B to C
        # ey = perpendicular
        BC_vec = C - B
        BC_len = np.linalg.norm(BC_vec)
        if BC_len <= 1e-12:
            raise ValueError("Degenerate BC length during motion.")

        ex = BC_vec / BC_len
        ey = np.array([-ex[1], ex[0]], dtype=float)

        P = B + self.px * ex + self.py * ey
        return P, C

    def generate_loop(self, num_samples=300):
        thetas = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False)
        pts = []
        prev_C = None
        first_C = None

        for i, theta in enumerate(thetas):
            P, C = self.point_at(theta, prev_C=prev_C)
            if i == 0:
                first_C = C.copy()
            pts.append(P)
            prev_C = C

        pts = np.asarray(pts, dtype=float)

        # Seam continuity check
        _, end_C = self.point_at(2.0 * np.pi, prev_C=prev_C)
        if np.linalg.norm(end_C - first_C) > 1e-2 * max(self.r1, self.r2, self.r3, self.r4):
            raise ValueError("Assembly-mode continuity failed around full cycle.")

        return pts
    
    def configuration_at(self, theta: float, prev_C=None):
        """
        Return the full 4-bar configuration at input angle theta.

        Output dict contains:
            A, B, C, D, P
        plus the chosen C branch for continuity tracking.
        """
        B, C1, C2 = self._C_candidates(theta)

        if prev_C is None:
            C = C1 if C1[1] >= C2[1] else C2
        else:
            d1 = np.linalg.norm(prev_C - C1)
            d2 = np.linalg.norm(prev_C - C2)
            C = C1 if d1 <= d2 else C2

        BC_vec = C - B
        BC_len = np.linalg.norm(BC_vec)
        if BC_len <= 1e-12:
            raise ValueError("Degenerate BC length during motion.")

        ex = BC_vec / BC_len
        ey = np.array([-ex[1], ex[0]], dtype=float)

        P = B + self.px * ex + self.py * ey

        return {
            "A": self.A.copy(),
            "B": B.copy(),
            "C": C.copy(),
            "D": self.D.copy(),
            "P": P.copy(),
            "_C": C.copy(),
        }

    def plot_mechanism(self, theta: float, prev_C=None, ax=None, show_trace=True, trace_samples=300):
        """
        Plot one static configuration of the 4-bar at angle theta.
        """
        cfg = self.configuration_at(theta, prev_C=prev_C)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        if show_trace:
            try:
                loop = self.generate_loop(num_samples=trace_samples)
                ax.plot(loop[:, 0], loop[:, 1], color="purple", alpha=0.45, linewidth=1.5, label="P trace")
            except ValueError:
                pass

        A = cfg["A"]; B = cfg["B"]; C = cfg["C"]; D = cfg["D"]; P = cfg["P"]

        # Ground link
        ax.plot([A[0], D[0]], [A[1], D[1]], color="black", linewidth=2, label="AD (ground)")

        # Moving links
        ax.plot([A[0], B[0]], [A[1], B[1]], linewidth=2, label="AB (input crank)")
        ax.plot([B[0], C[0]], [B[1], C[1]], linewidth=2, label="BC (coupler)")
        ax.plot([C[0], D[0]], [C[1], D[1]], linewidth=2, label="CD (output link)")

        # Offset from B to traced point P
        ax.plot([B[0], P[0]], [B[1], P[1]], linestyle="--", linewidth=1.5, label="offset to P")

        pts = {"A": A, "B": B, "C": C, "D": D, "P": P}
        for name, pt in pts.items():
            ax.scatter(pt[0], pt[1], s=40)
            ax.text(pt[0], pt[1], f" {name}", fontsize=10)

        ax.set_aspect("equal")
        ax.legend(loc="best")
        ax.set_title(f"4-bar configuration at theta = {theta:.3f}")
        plt.show()

    def animate_mechanism(self, num_frames=120, trace_samples=300, interval=60):
        """
        Animate the 4-bar motion over one full input revolution.
        """
        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots(figsize=(7, 7))

        try:
            loop = self.generate_loop(num_samples=trace_samples)
            ax.plot(loop[:, 0], loop[:, 1], color="purple", alpha=0.35, linewidth=1.5, label="P trace")
        except ValueError:
            loop = None

        line_AB, = ax.plot([], [], linewidth=2)
        line_BC, = ax.plot([], [], linewidth=2)
        line_CD, = ax.plot([], [], linewidth=2)
        line_BP, = ax.plot([], [], linestyle="--", linewidth=1.5)

        scat = ax.scatter([], [], s=40)
        labels = {name: ax.text(0, 0, name, fontsize=10) for name in ["A", "B", "C", "D", "P"]}

        thetas = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
        prev_C = None
        cfgs = []
        all_pts = []

        if loop is not None:
            all_pts.append(loop)

        for theta in thetas:
            cfg = self.configuration_at(theta, prev_C=prev_C)
            prev_C = cfg["_C"]
            cfgs.append(cfg)
            arr = np.vstack([cfg[k] for k in ["A", "B", "C", "D", "P"]])
            all_pts.append(arr)

        all_pts = np.vstack(all_pts)
        xmin, ymin = np.min(all_pts, axis=0)
        xmax, ymax = np.max(all_pts, axis=0)
        pad = 0.1 * max(xmax - xmin, ymax - ymin, 1.0)

        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect("equal")
        ax.set_title("4-bar motion")

        def update(i):
            cfg = cfgs[i]
            A = cfg["A"]; B = cfg["B"]; C = cfg["C"]; D = cfg["D"]; P = cfg["P"]

            line_AB.set_data([A[0], B[0]], [A[1], B[1]])
            line_BC.set_data([B[0], C[0]], [B[1], C[1]])
            line_CD.set_data([C[0], D[0]], [C[1], D[1]])
            line_BP.set_data([B[0], P[0]], [B[1], P[1]])

            pts = np.vstack([A, B, C, D, P])
            scat.set_offsets(pts)

            for name, pt in zip(["A", "B", "C", "D", "P"], pts):
                labels[name].set_position((pt[0], pt[1]))

            return line_AB, line_BC, line_CD, line_BP, scat, *labels.values()

        ani = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False, repeat=True)
        plt.show()
        return ani


class PathEstimator:
    """
    Estimates:
        x = [r1, r2, r3, r4, px, py, theta0]

    Improvements:
    - feature-based anchor selection
    - weighted landmark fitting
    - coarse-to-fine optimization
    - scale-aware initialization
    - extra local refinements
    """
    def __init__(self, path_points, n_starts=80, num_samples=60, seed=None):
        self.points = np.asarray(path_points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("path_points must be an (N, 2) array.")

        self.n_starts = int(n_starts)
        self.num_samples = int(num_samples)
        self.rng = np.random.default_rng(seed)

        # Dense target used for evaluation and feature extraction
        self.target_dense = self._resample_loop(self.points, max(120, self.num_samples))
        self.target = self._resample_loop(self.points, self.num_samples)

        centered = self.target_dense - np.mean(self.target_dense, axis=0, keepdims=True)
        self.scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
        if self.scale <= 1e-12:
            raise ValueError("Degenerate target scale.")

        # Multi-stage target sets
        self.anchor_idx_10, self.anchor_w_10 = self._build_feature_anchors(self.target_dense, k=10)
        self.anchor_idx_20, self.anchor_w_20 = self._build_feature_anchors(self.target_dense, k=20)

        self.target_10 = self.target_dense[self.anchor_idx_10]
        self.target_20 = self.target_dense[self.anchor_idx_20]

        self.best_x = None
        self.best_loss = np.inf
        self.best_loop = None
        self.best_mech = None

        self.estimate = self.optimize_path()

    def _resample_loop(self, pts, num_samples):
        pts = np.asarray(pts, dtype=float)

        if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]

        segs = np.roll(pts, -1, axis=0) - pts
        seglens = np.linalg.norm(segs, axis=1)
        total = np.sum(seglens)
        if total <= 1e-12:
            raise ValueError("Degenerate loop.")

        cum = np.concatenate([[0.0], np.cumsum(seglens)])
        s_vals = np.linspace(0.0, total, num_samples, endpoint=False)

        out = []
        j = 0
        for s in s_vals:
            while j < len(seglens) - 1 and cum[j + 1] <= s:
                j += 1

            ds = seglens[j]
            if ds <= 1e-12:
                out.append(pts[j].copy())
            else:
                alpha = (s - cum[j]) / ds
                p = (1.0 - alpha) * pts[j] + alpha * pts[(j + 1) % len(pts)]
                out.append(p)

        return np.asarray(out, dtype=float)

    def _normalize(self, pts):
        pts = np.asarray(pts, dtype=float)
        centered = pts - np.mean(pts, axis=0, keepdims=True)
        scale = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
        if scale <= 1e-12:
            raise ValueError("Degenerate normalized loop.")
        return centered / scale

    def _curvature_scores(self, pts):
        """
        Simple discrete turning-angle score on a closed loop.
        """
        pts = np.asarray(pts, dtype=float)
        prev = np.roll(pts, 1, axis=0)
        nxt = np.roll(pts, -1, axis=0)

        v1 = pts - prev
        v2 = nxt - pts

        n1 = np.linalg.norm(v1, axis=1, keepdims=True)
        n2 = np.linalg.norm(v2, axis=1, keepdims=True)

        good = (n1[:, 0] > 1e-12) & (n2[:, 0] > 1e-12)
        scores = np.zeros(len(pts))

        u1 = np.zeros_like(v1)
        u2 = np.zeros_like(v2)
        u1[good] = v1[good] / n1[good]
        u2[good] = v2[good] / n2[good]

        dots = np.clip(np.sum(u1 * u2, axis=1), -1.0, 1.0)
        ang = np.arccos(dots)
        scores[good] = ang[good]
        return scores

    def _build_feature_anchors(self, pts, k=10):
        """
        Build anchor indices emphasizing:
        - extrema in x and y
        - high curvature points
        - evenly spaced coverage
        """
        pts = np.asarray(pts, dtype=float)
        n = len(pts)
        if k >= n:
            idx = np.arange(n)
            w = np.ones(n)
            return idx, w

        chosen = set()

        # Extrema
        chosen.add(int(np.argmin(pts[:, 0])))
        chosen.add(int(np.argmax(pts[:, 0])))
        chosen.add(int(np.argmin(pts[:, 1])))
        chosen.add(int(np.argmax(pts[:, 1])))

        # High curvature points
        curv = self._curvature_scores(pts)
        curv_order = np.argsort(curv)[::-1]
        for idx in curv_order:
            if len(chosen) >= min(k, 8):
                break
            chosen.add(int(idx))

        # Fill remainder evenly
        if len(chosen) < k:
            even = np.linspace(0, n - 1, k, dtype=int)
            for idx in even:
                if len(chosen) >= k:
                    break
                chosen.add(int(idx))

        # If still not enough, fill greedily
        if len(chosen) < k:
            for idx in range(n):
                if len(chosen) >= k:
                    break
                chosen.add(int(idx))

        idx = np.array(sorted(chosen), dtype=int)

        # If too many, keep a spread
        if len(idx) > k:
            keep = np.linspace(0, len(idx) - 1, k, dtype=int)
            idx = idx[keep]

        # Weights: extrema + curvature points matter more
        w = np.ones(len(idx))
        xmn = int(np.argmin(pts[:, 0])); xmx = int(np.argmax(pts[:, 0]))
        ymn = int(np.argmin(pts[:, 1])); ymx = int(np.argmax(pts[:, 1]))

        for i, j in enumerate(idx):
            if j in {xmn, xmx, ymn, ymx}:
                w[i] = 3.0
            elif curv[j] >= np.quantile(curv, 0.85):
                w[i] = 2.5
            else:
                w[i] = 1.0

        return idx, w

    def _weighted_best_rotation_mse(self, X, Y, w):
        """
        Weighted Procrustes-style rotation fit.
        X, Y: (n,2)
        w: (n,)
        """
        w = np.asarray(w, dtype=float)
        w = w / np.sum(w)

        Xc = X - np.sum(w[:, None] * X, axis=0, keepdims=True)
        Yc = Y - np.sum(w[:, None] * Y, axis=0, keepdims=True)

        H = Xc.T @ (w[:, None] * Yc)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        Xr = Xc @ R
        return np.sum(w * np.sum((Xr - Yc) ** 2, axis=1))

    def _weighted_cyclic_loss(self, target_pts, gen_pts, weights):
        """
        Weighted cyclic ordered loss with reversal check.
        target_pts and gen_pts should already have same length.
        """
        X = self._normalize(gen_pts)
        Y = self._normalize(target_pts)

        best = np.inf
        n = len(X)

        for Ycand in (Y, Y[::-1]):
            for k in range(n):
                Yshift = np.roll(Ycand, shift=k, axis=0)
                mse = self._weighted_best_rotation_mse(X, Yshift, weights)
                if mse < best:
                    best = mse

        return float(best)

    def _dense_cyclic_loss(self, target_pts, gen_pts):
        X = self._normalize(target_pts)
        Y = self._normalize(gen_pts)

        best = np.inf
        n = len(X)

        for Ycand in (Y, Y[::-1]):
            for k in range(n):
                Yshift = np.roll(Ycand, shift=k, axis=0)
                mse = self._best_rotation_mse(X, Yshift)
                if mse < best:
                    best = mse

        return float(best)

    def _best_rotation_mse(self, X, Y):
        H = X.T @ Y
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        Xr = X @ R
        return np.mean(np.sum((Xr - Y) ** 2, axis=1))

    def _regularization(self, x):
        """
        Mild shape-aware regularization:
        - discourage absurd coupler offsets
        - discourage crazy length ratios
        """
        r1, r2, r3, r4, px, py, theta0 = x
        lengths = np.array([r1, r2, r3, r4], dtype=float)
        ratio = np.max(lengths) / np.max([np.min(lengths), 1e-9])

        reg = 0.0
        reg += 5e-4 * (px**2 + py**2) / max(self.scale**2, 1e-9)
        reg += 2e-3 * max(0.0, ratio - 6.0)**2
        return float(reg)

    def loss_stage1(self, x):
        """
        Fit 10 weighted feature anchors.
        """
        try:
            mech = Mechanism(x)
            gen = mech.generate_loop(num_samples=max(80, self.num_samples))
            gen_dense = self._resample_loop(gen, len(self.target_dense))
            gen_10 = gen_dense[self.anchor_idx_10]
            return self._weighted_cyclic_loss(self.target_10, gen_10, self.anchor_w_10) + self._regularization(x)
        except ValueError:
            return 1e6

    def loss_stage2(self, x):
        """
        Fit 20 weighted feature anchors.
        """
        try:
            mech = Mechanism(x)
            gen = mech.generate_loop(num_samples=max(100, self.num_samples))
            gen_dense = self._resample_loop(gen, len(self.target_dense))
            gen_20 = gen_dense[self.anchor_idx_20]
            return self._weighted_cyclic_loss(self.target_20, gen_20, self.anchor_w_20) + self._regularization(x)
        except ValueError:
            return 1e6

    def loss_final(self, x):
        """
        Dense final loss.
        """
        try:
            mech = Mechanism(x)
            gen = mech.generate_loop(num_samples=self.num_samples)
            return self._dense_cyclic_loss(self.target, self._resample_loop(gen, self.num_samples)) + self._regularization(x)
        except ValueError:
            return 1e6

    def random_start(self):
        """
        Scale-aware initialization.
        """
        s = self.scale
        for _ in range(10000):
            x0 = np.array([
                self.rng.uniform(0.3 * s, 3.5 * s),   # r1
                self.rng.uniform(0.2 * s, 3.0 * s),   # r2
                self.rng.uniform(0.2 * s, 3.0 * s),   # r3
                self.rng.uniform(0.2 * s, 3.0 * s),   # r4
                self.rng.uniform(-1.8 * s, 1.8 * s),  # px
                self.rng.uniform(-1.8 * s, 1.8 * s),  # py
                self.rng.uniform(0.0, 2.0 * np.pi),   # theta0
            ], dtype=float)

            if self.loss_stage1(x0) < 1e5:
                return x0

        raise RuntimeError("Failed to find a valid starting point.")

    def _perturb(self, x, scale=0.08):
        y = np.asarray(x, dtype=float).copy()
        y[:4] *= (1.0 + self.rng.normal(0.0, scale, size=4))
        y[4:6] += self.rng.normal(0.0, scale * self.scale, size=2)
        y[6] = (y[6] + self.rng.normal(0.0, 0.2)) % (2.0 * np.pi)
        return y

    def optimize_path(self):
        if not SCIPY_AVAILABLE:
            # Fallback: random search using final loss
            for _ in range(self.n_starts * 40):
                x = self.random_start()
                f = self.loss_final(x)
                if f < self.best_loss:
                    self.best_loss = f
                    self.best_x = x.copy()

            self.best_mech = Mechanism(self.best_x)
            self.best_loop = self.best_mech.generate_loop(num_samples=self.num_samples)
            return {
                "x": self.best_x,
                "mechanism": self.best_mech,
                "loop": self.best_loop,
                "loss": self.best_loss,
            }

        s = self.scale
        bounds = [
            (0.05 * s, 6.0 * s),      # r1
            (0.05 * s, 6.0 * s),      # r2
            (0.05 * s, 6.0 * s),      # r3
            (0.05 * s, 6.0 * s),      # r4
            (-3.0 * s, 3.0 * s),      # px
            (-3.0 * s, 3.0 * s),      # py
            (0.0, 2.0 * np.pi),       # theta0
        ]

        for _ in range(self.n_starts):
            x0 = self.random_start()

            # Stage 1: 10 weighted anchors
            res1 = minimize(
                self.loss_stage1,
                x0,
                method="Powell",
                bounds=bounds,
                options={"maxiter": 180, "disp": False}
            )

            # Stage 2: 20 weighted anchors
            res2 = minimize(
                self.loss_stage2,
                res1.x,
                method="Powell",
                bounds=bounds,
                options={"maxiter": 220, "disp": False}
            )

            # Stage 3: dense refinement
            res3 = minimize(
                self.loss_final,
                res2.x,
                method="Powell",
                bounds=bounds,
                options={"maxiter": 260, "disp": False}
            )

            x_best = res3.x
            f_best = self.loss_final(x_best)

            if f_best < self.best_loss:
                self.best_loss = f_best
                self.best_x = x_best.copy()

        # Additional local refinements around current best
        if self.best_x is not None:
            for _ in range(12):
                x0 = self._perturb(self.best_x, scale=0.06)
                res = minimize(
                    self.loss_final,
                    x0,
                    method="Powell",
                    bounds=bounds,
                    options={"maxiter": 180, "disp": False}
                )
                f = self.loss_final(res.x)
                if f < self.best_loss:
                    self.best_loss = f
                    self.best_x = res.x.copy()

        self.best_mech = Mechanism(self.best_x)
        self.best_loop = self.best_mech.generate_loop(num_samples=self.num_samples)

        return {
            "x": self.best_x,
            "mechanism": self.best_mech,
            "loop": self.best_loop,
            "loss": self.best_loss,
        }

    def plot_best(self):
        if self.best_loop is None:
            raise RuntimeError("No valid estimate found.")

        target = self._normalize(self.target)
        gen = self._normalize(self.best_loop)

        plt.figure(figsize=(6, 6))
        plt.plot(target[:, 0], target[:, 1], label="target", linewidth=2)
        plt.plot(gen[:, 0], gen[:, 1], label="best generated", linewidth=2)
        plt.axis("equal")
        plt.legend()
        plt.show()


# ------------------------------------------------------------
# Example 1: sanity check with a target generated by the model
# ------------------------------------------------------------
# true_x = np.array([3.0, 1.2, 2.6, 2.1, 1.0, 0.7, 0.4])
# true_mech = Mechanism(true_x)
# path_points = true_mech.generate_loop(num_samples=200)

# est = PathEstimator(path_points, n_starts=10, num_samples=200, seed=0)

# print("best loss:", est.estimate["loss"])
# print("estimated x = [r1, r2, r3, r4, px, py, theta0]:")
# print(est.estimate["x"])

# est.plot_best()


# ------------------------------------------------------------
# Example 2: try an ellipse anyway
# ------------------------------------------------------------
t = np.linspace(0, 2*np.pi, 400, endpoint=False)
r = 1.4 + 0.55*np.sin(3*t) + 0.35*np.cos(5*t)
path_points = np.column_stack((r*np.cos(t), r*np.sin(t)))
est = PathEstimator(path_points, n_starts=120, num_samples=36, seed=0)
print("best loss:", est.estimate["loss"])
print(est.estimate["x"])
est.plot_best()
mech = est.best_mech
mech.plot_mechanism(0.8)
ani = mech.animate_mechanism()