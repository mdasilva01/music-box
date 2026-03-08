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


class PathEstimator:
    """
    Estimates:
        x = [r1, r2, r3, r4, px, py, theta0]
    """
    def __init__(self, path_points, n_starts=12, num_samples=250, seed=None):
        self.points = np.asarray(path_points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("path_points must be an (N, 2) array.")

        self.n_starts = int(n_starts)
        self.num_samples = int(num_samples)
        self.rng = np.random.default_rng(seed)

        self.target = self._resample_loop(self.points, self.num_samples)

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

            # Mild regularization so optimizer doesn't wander into absurdly huge offsets
            reg = 1e-3 * (x[4]**2 + x[5]**2)
            return float(best + reg)

        except ValueError:
            return 1e6

    def random_start(self):
        for _ in range(5000):
            x0 = np.array([
                self.rng.uniform(0.3, 6.0),          # r1
                self.rng.uniform(0.3, 6.0),          # r2
                self.rng.uniform(0.3, 6.0),          # r3
                self.rng.uniform(0.3, 6.0),          # r4
                self.rng.uniform(-4.0, 4.0),         # px
                self.rng.uniform(-4.0, 4.0),         # py
                self.rng.uniform(0.0, 2.0*np.pi),    # theta0
            ], dtype=float)

            if self.loss_from_x(x0) < 1e5:
                return x0

        raise RuntimeError("Failed to find a valid starting point.")

    def optimize_path(self):
        if not SCIPY_AVAILABLE:
            # Fallback: random search only
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

        bounds = [
            (0.05, 10.0),             # r1
            (0.05, 10.0),             # r2
            (0.05, 10.0),             # r3
            (0.05, 10.0),             # r4
            (-8.0, 8.0),              # px
            (-8.0, 8.0),              # py
            (0.0, 2.0*np.pi),         # theta0
        ]

        for _ in range(self.n_starts):
            x0 = self.random_start()

            res = minimize(
                self.loss_from_x,
                x0,
                method="Powell",
                bounds=bounds,
                options={"maxiter": 300, "disp": False}
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
t = np.linspace(0, 2*np.pi, 200, endpoint=False)
path_points = np.column_stack((2*np.cos(t), 1*np.sin(t)))
est = PathEstimator(path_points, n_starts=40, num_samples=200, seed=0)
print("best loss:", est.estimate["loss"])
print(est.estimate["x"])
est.plot_best()