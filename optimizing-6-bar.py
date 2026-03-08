import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class Mechanism:
    """
    Stephenson-type planar 6-bar mechanism.

    Parameter vector:
        x = [
            r1, r2, r3, r4,      # base 4-bar lengths
            ex, ey,              # point E on coupler BC, in BC-local frame
            gx, gy,              # fixed ground pivot G for the added dyad
            r5, r6,              # lengths EF and FG
            px, py,              # traced point P on link EF, in EF-local frame
            theta0               # phase offset
        ]

    Geometry:
        A = (0, 0)
        D = (r1, 0)

        Base 4-bar:
            A --(r2)--> B --(r3)--> C --(r4)--> D

        Added Stephenson dyad:
            E is a rigid point on coupler BC
            F is the moving joint from circles centered at E and G
            P is a rigid point on link EF

    So the total mechanism is:
        ground + AB + BC(ternary with E) + CD + EF + FG
    """

    def __init__(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        if x.shape != (13,):
            raise ValueError("x must have shape (13,)")

        self.x = x.copy()
        (
            self.r1, self.r2, self.r3, self.r4,
            self.ex, self.ey,
            self.gx, self.gy,
            self.r5, self.r6,
            self.px, self.py,
            self.theta0
        ) = x

        self.A = np.array([0.0, 0.0], dtype=float)
        self.D = np.array([self.r1, 0.0], dtype=float)
        self.G = np.array([self.gx, self.gy], dtype=float)

        self._validate()

    def _validate(self) -> None:
        lengths = np.array([self.r1, self.r2, self.r3, self.r4, self.r5, self.r6], dtype=float)
        if np.any(lengths <= 0):
            raise ValueError("All link lengths must be positive.")

        # Use strict Grashof on the primary 4-bar to encourage full crank rotation.
        links4 = np.sort(np.array([self.r1, self.r2, self.r3, self.r4], dtype=float))
        s, p, q, l = links4
        if not (s + l < p + q - 1e-8):
            raise ValueError("Primary 4-bar is not strictly Grashof.")

    @staticmethod
    def _circle_intersections(center1, radius1, center2, radius2):
        """
        Return the two circle intersections, or raise ValueError if none exist.
        """
        center1 = np.asarray(center1, dtype=float)
        center2 = np.asarray(center2, dtype=float)

        vec = center2 - center1
        d = float(np.linalg.norm(vec))
        if d <= 1e-12:
            raise ValueError("Degenerate circle centers.")

        if not (abs(radius1 - radius2) <= d <= radius1 + radius2):
            raise ValueError("No real circle intersection.")

        a = (radius1**2 - radius2**2 + d**2) / (2.0 * d)
        h_sq = radius1**2 - a**2
        if h_sq < -1e-10:
            raise ValueError("Negative h^2 in circle intersection.")
        h_sq = max(h_sq, 0.0)
        h = np.sqrt(h_sq)

        e = vec / d
        m = center1 + a * e
        n = np.array([-e[1], e[0]], dtype=float)

        p1 = m + h * n
        p2 = m - h * n
        return p1, p2

    def _B_at(self, theta: float) -> np.ndarray:
        ang = theta + self.theta0
        return np.array(
            [self.r2 * np.cos(ang), self.r2 * np.sin(ang)],
            dtype=float
        )

    def _C_candidates(self, theta: float):
        B = self._B_at(theta)
        C1, C2 = self._circle_intersections(B, self.r3, self.D, self.r4)
        return B, C1, C2

    def _choose_branch(self, cand1, cand2, prev):
        if prev is None:
            return cand1 if cand1[1] >= cand2[1] else cand2
        d1 = np.linalg.norm(prev - cand1)
        d2 = np.linalg.norm(prev - cand2)
        return cand1 if d1 <= d2 else cand2

    def point_at(self, theta: float, prev_C=None, prev_F=None):
        # Solve primary 4-bar.
        B, C1, C2 = self._C_candidates(theta)
        C = self._choose_branch(C1, C2, prev_C)

        BC_vec = C - B
        BC_len = np.linalg.norm(BC_vec)
        if BC_len <= 1e-12:
            raise ValueError("Degenerate BC during motion.")

        ex_bc = BC_vec / BC_len
        ey_bc = np.array([-ex_bc[1], ex_bc[0]], dtype=float)

        # Point E is rigidly attached to coupler BC.
        E = B + self.ex * ex_bc + self.ey * ey_bc

        # Solve second dyad: F from circles centered at E and G.
        F1, F2 = self._circle_intersections(E, self.r5, self.G, self.r6)
        F = self._choose_branch(F1, F2, prev_F)

        EF_vec = F - E
        EF_len = np.linalg.norm(EF_vec)
        if EF_len <= 1e-12:
            raise ValueError("Degenerate EF during motion.")

        ex_ef = EF_vec / EF_len
        ey_ef = np.array([-ex_ef[1], ex_ef[0]], dtype=float)

        # Traced point P is rigidly attached to EF.
        P = E + self.px * ex_ef + self.py * ey_ef
        return P, C, F

    def generate_loop(self, num_samples: int = 300) -> np.ndarray:
        thetas = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False)
        pts = []
        prev_C = None
        prev_F = None
        first_C = None
        first_F = None

        for i, theta in enumerate(thetas):
            P, C, F = self.point_at(theta, prev_C=prev_C, prev_F=prev_F)
            if i == 0:
                first_C = C.copy()
                first_F = F.copy()
            pts.append(P)
            prev_C = C
            prev_F = F

        pts = np.asarray(pts, dtype=float)

        # Seam continuity check for both loops.
        _, end_C, end_F = self.point_at(2.0 * np.pi, prev_C=prev_C, prev_F=prev_F)
        scale = max(self.r1, self.r2, self.r3, self.r4, self.r5, self.r6)
        if np.linalg.norm(end_C - first_C) > 1e-2 * scale:
            raise ValueError("C-branch continuity failed around full cycle.")
        if np.linalg.norm(end_F - first_F) > 1e-2 * scale:
            raise ValueError("F-branch continuity failed around full cycle.")

        return pts
    
    def configuration_at(self, theta: float, prev_C=None, prev_F=None):
        """
        Return the full configuration of the 6-bar at angle theta.

        Outputs a dict containing:
            A, B, C, D, E, F, G, P
        plus the chosen branch points C and F for continuity tracking.
        """
        B, C1, C2 = self._C_candidates(theta)
        C = self._choose_branch(C1, C2, prev_C)

        BC_vec = C - B
        BC_len = np.linalg.norm(BC_vec)
        if BC_len <= 1e-12:
            raise ValueError("Degenerate BC during motion.")

        ex_bc = BC_vec / BC_len
        ey_bc = np.array([-ex_bc[1], ex_bc[0]], dtype=float)

        E = B + self.ex * ex_bc + self.ey * ey_bc

        F1, F2 = self._circle_intersections(E, self.r5, self.G, self.r6)
        F = self._choose_branch(F1, F2, prev_F)

        EF_vec = F - E
        EF_len = np.linalg.norm(EF_vec)
        if EF_len <= 1e-12:
            raise ValueError("Degenerate EF during motion.")

        ex_ef = EF_vec / EF_len
        ey_ef = np.array([-ex_ef[1], ex_ef[0]], dtype=float)

        P = E + self.px * ex_ef + self.py * ey_ef

        return {
            "A": self.A.copy(),
            "B": B.copy(),
            "C": C.copy(),
            "D": self.D.copy(),
            "E": E.copy(),
            "F": F.copy(),
            "G": self.G.copy(),
            "P": P.copy(),
            "_C": C.copy(),
            "_F": F.copy(),
        }

    def plot_mechanism(self, theta: float, prev_C=None, prev_F=None, ax=None, show_trace=True, trace_samples=300):
        """
        Plot one static configuration of the 6-bar at a given theta.
        """
        cfg = self.configuration_at(theta, prev_C=prev_C, prev_F=prev_F)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        if show_trace:
            try:
                loop = self.generate_loop(num_samples=trace_samples)
                ax.plot(loop[:, 0], loop[:, 1], color="purple", alpha=0.5, linewidth=1.5, label="P trace")
            except ValueError:
                pass

        A = cfg["A"]; B = cfg["B"]; C = cfg["C"]; D = cfg["D"]
        E = cfg["E"]; F = cfg["F"]; G = cfg["G"]; P = cfg["P"]

        # Ground / fixed structure
        ax.plot([A[0], D[0]], [A[1], D[1]], color="black", linewidth=2, label="ground AD")
        ax.plot([G[0]], [G[1]], marker="s", color="black")

        # Primary 4-bar
        ax.plot([A[0], B[0]], [A[1], B[1]], linewidth=2, label="AB")
        ax.plot([B[0], C[0]], [B[1], C[1]], linewidth=2, label="BC")
        ax.plot([C[0], D[0]], [C[1], D[1]], linewidth=2, label="CD")

        # Added Stephenson dyad
        ax.plot([E[0], F[0]], [E[1], F[1]], linewidth=2, label="EF")
        ax.plot([G[0], F[0]], [G[1], F[1]], linewidth=2, label="GF")

        # Rigid offsets to traced point
        ax.plot([E[0], P[0]], [E[1], P[1]], linestyle="--", linewidth=1.5, label="offset to P")

        # Draw points
        pts = {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F, "G": G, "P": P}
        for name, pt in pts.items():
            ax.scatter(pt[0], pt[1], s=40)
            ax.text(pt[0], pt[1], f" {name}", fontsize=10)

        ax.set_aspect("equal")
        ax.legend(loc="best")
        ax.set_title(f"Stephenson-type 6-bar at theta = {theta:.3f}")
        plt.show()
    
    def animate_mechanism(self, num_frames=120, trace_samples=300, interval=60):
        """
        Animate the 6-bar motion over one full cycle.
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
        line_EF, = ax.plot([], [], linewidth=2)
        line_GF, = ax.plot([], [], linewidth=2)
        line_EP, = ax.plot([], [], linestyle="--", linewidth=1.5)

        scat = ax.scatter([], [], s=40)

        labels = {name: ax.text(0, 0, name, fontsize=10) for name in ["A", "B", "C", "D", "E", "F", "G", "P"]}

        # Set axis limits
        all_pts = []
        if loop is not None:
            all_pts.append(loop)
        thetas = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
        prev_C = None
        prev_F = None
        cfgs = []
        for theta in thetas:
            cfg = self.configuration_at(theta, prev_C=prev_C, prev_F=prev_F)
            prev_C = cfg["_C"]
            prev_F = cfg["_F"]
            cfgs.append(cfg)
            arr = np.vstack([cfg[k] for k in ["A", "B", "C", "D", "E", "F", "G", "P"]])
            all_pts.append(arr)

        all_pts = np.vstack(all_pts)
        xmin, ymin = np.min(all_pts, axis=0)
        xmax, ymax = np.max(all_pts, axis=0)
        pad = 0.1 * max(xmax - xmin, ymax - ymin, 1.0)

        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect("equal")
        ax.set_title("Stephenson-type 6-bar motion")

        def update(i):
            cfg = cfgs[i]
            A = cfg["A"]; B = cfg["B"]; C = cfg["C"]; D = cfg["D"]
            E = cfg["E"]; F = cfg["F"]; G = cfg["G"]; P = cfg["P"]

            line_AB.set_data([A[0], B[0]], [A[1], B[1]])
            line_BC.set_data([B[0], C[0]], [B[1], C[1]])
            line_CD.set_data([C[0], D[0]], [C[1], D[1]])
            line_EF.set_data([E[0], F[0]], [E[1], F[1]])
            line_GF.set_data([G[0], F[0]], [G[1], F[1]])
            line_EP.set_data([E[0], P[0]], [E[1], P[1]])

            pts = np.vstack([A, B, C, D, E, F, G, P])
            scat.set_offsets(pts)

            for name, pt in zip(["A", "B", "C", "D", "E", "F", "G", "P"], pts):
                labels[name].set_position((pt[0], pt[1]))

            return line_AB, line_BC, line_CD, line_EF, line_GF, line_EP, scat, *labels.values()

        ani = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False, repeat=True)
        plt.show()
        return ani


class PathEstimator:
    """
    Estimates the 13-parameter Stephenson-type 6-bar:
        x = [r1, r2, r3, r4, ex, ey, gx, gy, r5, r6, px, py, theta0]
    """

    def __init__(self, path_points, n_starts=20, num_samples=120, seed=None):
        self.points = np.asarray(path_points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("path_points must be an (N, 2) array.")

        self.n_starts = int(n_starts)
        self.num_samples = int(num_samples)
        self.rng = np.random.default_rng(seed)

        self.target = self._resample_loop(self.points, self.num_samples)

        # Scale heuristic from the target.
        centered = self.target - np.mean(self.target, axis=0, keepdims=True)
        self.scale = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
        if self.scale <= 1e-12:
            raise ValueError("Degenerate target scale.")

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

    def _best_rotation_mse(self, X, Y):
        H = X.T @ Y
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        Xr = X @ R
        return np.mean(np.sum((Xr - Y) ** 2, axis=1))

    def loss_from_x(self, x):
        try:
            mech = Mechanism(x)
            gen = mech.generate_loop(num_samples=self.num_samples)

            X = self._normalize(self._resample_loop(gen, self.num_samples))
            Y = self._normalize(self.target)

            best = np.inf
            for Ycand in (Y, Y[::-1]):
                for k in range(self.num_samples):
                    Yshift = np.roll(Ycand, shift=k, axis=0)
                    mse = self._best_rotation_mse(X, Yshift)
                    if mse < best:
                        best = mse

            # Mild regularization:
            # keep coupler offsets and second-loop geometry from blowing up.
            reg = 1e-3 * (
                x[4]**2 + x[5]**2 +     # ex, ey
                x[10]**2 + x[11]**2     # px, py
            )
            return float(best + reg)

        except ValueError:
            return 1e6

    def random_start(self):
        s = self.scale

        for _ in range(10000):
            r1 = self.rng.uniform(0.5 * s, 3.0 * s)
            r2 = self.rng.uniform(0.3 * s, 2.5 * s)
            r3 = self.rng.uniform(0.3 * s, 2.5 * s)
            r4 = self.rng.uniform(0.3 * s, 2.5 * s)

            ex = self.rng.uniform(-1.5 * s, 1.5 * s)
            ey = self.rng.uniform(-1.5 * s, 1.5 * s)

            gx = self.rng.uniform(-2.5 * s, 2.5 * s)
            gy = self.rng.uniform(-2.5 * s, 2.5 * s)

            r5 = self.rng.uniform(0.3 * s, 2.5 * s)
            r6 = self.rng.uniform(0.3 * s, 2.5 * s)

            px = self.rng.uniform(-1.5 * s, 1.5 * s)
            py = self.rng.uniform(-1.5 * s, 1.5 * s)

            theta0 = self.rng.uniform(0.0, 2.0 * np.pi)

            x0 = np.array([
                r1, r2, r3, r4,
                ex, ey,
                gx, gy,
                r5, r6,
                px, py,
                theta0
            ], dtype=float)

            if self.loss_from_x(x0) < 1e5:
                return x0

        raise RuntimeError("Failed to find a valid starting point.")

    def optimize_path(self):
        if not SCIPY_AVAILABLE:
            for _ in range(self.n_starts * 50):
                x = self.random_start()
                f = self.loss_from_x(x)
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
            (0.1 * s, 5.0 * s),          # r1
            (0.1 * s, 5.0 * s),          # r2
            (0.1 * s, 5.0 * s),          # r3
            (0.1 * s, 5.0 * s),          # r4
            (-3.0 * s, 3.0 * s),         # ex
            (-3.0 * s, 3.0 * s),         # ey
            (-4.0 * s, 4.0 * s),         # gx
            (-4.0 * s, 4.0 * s),         # gy
            (0.1 * s, 5.0 * s),          # r5
            (0.1 * s, 5.0 * s),          # r6
            (-3.0 * s, 3.0 * s),         # px
            (-3.0 * s, 3.0 * s),         # py
            (0.0, 2.0 * np.pi),          # theta0
        ]

        for _ in range(self.n_starts):
            x0 = self.random_start()

            res = minimize(
                self.loss_from_x,
                x0,
                method="Powell",
                bounds=bounds,
                options={"maxiter": 400, "disp": False}
            )

            x_best = res.x
            f_best = self.loss_from_x(x_best)

            if f_best < self.best_loss:
                self.best_loss = f_best
                self.best_x = x_best.copy()

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
# true_x = np.array([
#     3.0, 1.0, 2.2, 1.8,    # r1..r4
#     0.9, 0.4,              # ex, ey
#     1.5, 1.8,              # gx, gy
#     1.4, 1.6,              # r5, r6
#     0.6, 0.3,              # px, py
#     0.2                    # theta0
# ])
# true_mech = Mechanism(true_x)
# path_points = true_mech.generate_loop(num_samples=200)
# est = PathEstimator(path_points, n_starts=10, num_samples=100, seed=0)
# print("best loss:", est.estimate["loss"])
# print(est.estimate["x"])
# est.plot_best()


# ------------------------------------------------------------
# Example 2: more complicated target
# ------------------------------------------------------------
t = np.linspace(0, 2*np.pi, 400, endpoint=False)
r = 1.4 + 0.55*np.sin(3*t) + 0.35*np.cos(5*t)
path_points = np.column_stack((r*np.cos(t), r*np.sin(t)))

est = PathEstimator(path_points, n_starts=20, num_samples=50, seed=0)
print("best loss:", est.estimate["loss"])
print(est.estimate["x"])
est.plot_best()

est.best_mech.plot_mechanism(theta=0.5)
est.best_mech.animate_mechanism()