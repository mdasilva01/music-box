import numpy as np
import matplotlib.pyplot as plt
import time as t

def perform_algo_1d(t: np.ndarray, f: np.ndarray):
    
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    n = len(f)

    dt = np.empty(n, float)
    dt[:-1] = t[1:] - t[:-1]
    period = (t[-1] - t[0]) + dt[0] 

    wrap_dt = dt[:-1].mean() if n > 2 else dt[0]
    dt[-1] = wrap_dt

    A = np.zeros((n, n), float)
    b = np.zeros(n, float)

    for i in range(n):
        im1 = (i - 1) % n
        ip1 = (i + 1) % n

        dt1 = dt[im1]            
        dt2 = dt[i]              
        df1 = f[i] - f[im1]
        df2 = f[ip1] - f[i]

        c1 = 1.0 / dt1
        c2 = 2.0 * (1.0 / dt1 + 1.0 / dt2)
        c3 = 1.0 / dt2
        const = 3.0 * (df1 / (dt1**2) + df2 / (dt2**2))

        A[i, im1] = c1
        A[i, i]   = c2
        A[i, ip1] = c3
        b[i]      = const

    ks = np.linalg.solve(A, b)
    return ks, dt

def ab_from_ks(ks, dt, f):
    """
    Wikipedia a,b for each segment [i -> i+1]:
      a = k_i * dt_i - (f_{i+1}-f_i)
      b = -k_{i+1} * dt_i + (f_{i+1}-f_i)
    """
    n = len(f)
    a_b = []
    for i in range(n):
        ip1 = (i + 1) % n
        dfi = f[ip1] - f[i]
        a = ks[i] * dt[i] - dfi
        b = -ks[ip1] * dt[i] + dfi
        a_b.append((a, b))
    return a_b

def q(u, u0, du, f0, f1, a, b):
    tau = (u - u0) / du
    return (1 - tau) * f0 + tau * f1 + tau * (1 - tau) * ((1 - tau) * a + tau * b)


def build_parametric_spline(points, samples_per_seg=80):
    P = np.asarray(points, float)
    n = len(P)

    t = np.arange(n, dtype=float)

    kx, dt = perform_algo_1d(t, P[:, 0])
    ky, _  = perform_algo_1d(t, P[:, 1])

    abx = ab_from_ks(kx, dt, P[:, 0])
    aby = ab_from_ks(ky, dt, P[:, 1])

    xs, ys = [], []
    for i in range(n):
        ip1 = (i + 1) % n
        t0 = t[i]
        du = dt[i]

        for s in range(samples_per_seg):
            u = t0 + (s / samples_per_seg) * du
            x = q(u, t0, du, P[i, 0], P[ip1, 0], abx[i][0], abx[i][1])
            y = q(u, t0, du, P[i, 1], P[ip1, 1], aby[i][0], aby[i][1])
            xs.append(x); ys.append(y)

    return np.array(xs), np.array(ys)

def plot_points(points, xs, ys):
    px, py = zip(*points)

    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys)
    plt.scatter(px, py)
    for i, (x, y) in enumerate(points):
        plt.text(x, y, f"{i+1}", fontsize=16, fontweight="bold")
    plt.axis("equal")

from matplotlib.animation import FuncAnimation

class CamPairDesigner:
    """
    Two-cam design for a 2D path:
      cam_x controls x(theta)
      cam_y controls y(theta)

    We model each cam as a radial profile:
      r_x(theta) = base_radius + gain * x_centered(theta)
      r_y(theta) = base_radius + gain * y_centered(theta)

    Then the follower outputs are recovered by:
      x(theta) = (r_x(theta) - base_radius) / gain
      y(theta) = (r_y(theta) - base_radius) / gain
    """

    def __init__(self, xs, ys, base_radius=2.0, gain=1.0):
        self.xs = np.asarray(xs, dtype=float)
        self.ys = np.asarray(ys, dtype=float)

        if self.xs.shape != self.ys.shape:
            raise ValueError("xs and ys must have the same shape.")
        if len(self.xs) < 10:
            raise ValueError("Need at least 10 sampled points.")

        self.n = len(self.xs)
        self.theta = np.linspace(0.0, 2.0*np.pi, self.n, endpoint=False)

        self.base_radius = float(base_radius)
        self.gain = float(gain)

        # Center the motion so cam radii stay well-behaved
        self.x_center = np.mean(self.xs)
        self.y_center = np.mean(self.ys)

        self.x_disp = self.xs - self.x_center
        self.y_disp = self.ys - self.y_center

        self.cam_x = self.base_radius + self.gain * self.x_disp
        self.cam_y = self.base_radius + self.gain * self.y_disp

        if np.min(self.cam_x) <= 0 or np.min(self.cam_y) <= 0:
            raise ValueError(
                "Cam radius became nonpositive. Increase base_radius or reduce gain."
            )

    @classmethod
    def from_points(cls, points, samples_per_seg=120, base_radius=2.0, gain=1.0):
        xs, ys = build_parametric_spline(points, samples_per_seg=samples_per_seg)
        return cls(xs, ys, base_radius=base_radius, gain=gain)

    def reconstruct_path(self):
        """
        Reconstruct the path from the cam radii.
        """
        xr = (self.cam_x - self.base_radius) / self.gain + self.x_center
        yr = (self.cam_y - self.base_radius) / self.gain + self.y_center
        return xr, yr

    def plot_cams(self):
        """
        Plot the two cam profiles in polar form.
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": "polar"})

        axs[0].plot(self.theta, self.cam_x, linewidth=2)
        axs[0].set_title("Cam X profile")

        axs[1].plot(self.theta, self.cam_y, linewidth=2)
        axs[1].set_title("Cam Y profile")

        plt.tight_layout()
        plt.show()

    def plot_path_comparison(self):
        """
        Plot original spline path and reconstructed path.
        """
        xr, yr = self.reconstruct_path()

        plt.figure(figsize=(6, 6))
        plt.plot(self.xs, self.ys, label="original spline path", linewidth=2)
        plt.plot(xr, yr, "--", label="reconstructed from cams", linewidth=2)
        plt.axis("equal")
        plt.legend()
        plt.title("Target path vs cam-generated path")
        plt.show()

    def plot_displacement_functions(self):
        """
        Plot x(theta), y(theta) and corresponding cam radii.
        """
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        axs[0].plot(self.theta, self.x_disp, label="x displacement")
        axs[0].plot(self.theta, self.y_disp, label="y displacement")
        axs[0].legend()
        axs[0].set_ylabel("displacement")

        axs[1].plot(self.theta, self.cam_x, label="cam_x radius")
        axs[1].plot(self.theta, self.cam_y, label="cam_y radius")
        axs[1].legend()
        axs[1].set_xlabel("theta")
        axs[1].set_ylabel("radius")

        plt.tight_layout()
        plt.show()

    def animate(self, interval=40):
        """
        Animate:
        - the two cams in polar-ish Cartesian view
        - the generated 2D follower path
        """
        xr, yr = self.reconstruct_path()

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        # Precompute Cartesian cam shapes
        camx_X = self.cam_x * np.cos(self.theta)
        camx_Y = self.cam_x * np.sin(self.theta)
        camy_X = self.cam_y * np.cos(self.theta)
        camy_Y = self.cam_y * np.sin(self.theta)

        # Static cam outlines
        ax1.plot(camx_X, camx_Y, alpha=0.4)
        ax2.plot(camy_X, camy_Y, alpha=0.4)

        ax1.set_aspect("equal")
        ax2.set_aspect("equal")
        ax1.set_title("Cam X")
        ax2.set_title("Cam Y")

        padx = 0.2 * np.max(np.abs(camx_X))
        pady = 0.2 * np.max(np.abs(camx_Y))
        ax1.set_xlim(np.min(camx_X)-padx, np.max(camx_X)+padx)
        ax1.set_ylim(np.min(camx_Y)-pady, np.max(camx_Y)+pady)

        padx = 0.2 * np.max(np.abs(camy_X))
        pady = 0.2 * np.max(np.abs(camy_Y))
        ax2.set_xlim(np.min(camy_X)-padx, np.max(camy_X)+padx)
        ax2.set_ylim(np.min(camy_Y)-pady, np.max(camy_Y)+pady)

        # Path plot
        ax3.plot(self.xs, self.ys, alpha=0.25, label="target path")
        ax3.set_aspect("equal")
        ax3.set_title("Follower path")
        ax3.legend()

        path_line, = ax3.plot([], [], linewidth=2, color="tab:orange")
        path_dot, = ax3.plot([], [], "o", color="tab:red")

        # Rotating radius indicators on cams
        camx_radius, = ax1.plot([], [], color="tab:red", linewidth=2)
        camx_dot, = ax1.plot([], [], "o", color="tab:red")

        camy_radius, = ax2.plot([], [], color="tab:blue", linewidth=2)
        camy_dot, = ax2.plot([], [], "o", color="tab:blue")

        drawn_x = []
        drawn_y = []

        def update(i):
            th = self.theta[i]

            # Cam X current point
            cx = self.cam_x[i] * np.cos(th)
            cy = self.cam_x[i] * np.sin(th)
            camx_radius.set_data([0, cx], [0, cy])
            camx_dot.set_data([cx], [cy])

            # Cam Y current point
            dx = self.cam_y[i] * np.cos(th)
            dy = self.cam_y[i] * np.sin(th)
            camy_radius.set_data([0, dx], [0, dy])
            camy_dot.set_data([dx], [dy])

            drawn_x.append(xr[i])
            drawn_y.append(yr[i])
            path_line.set_data(drawn_x, drawn_y)
            path_dot.set_data([xr[i]], [yr[i]])

            return camx_radius, camx_dot, camy_radius, camy_dot, path_line, path_dot

        ani = FuncAnimation(fig, update, frames=self.n, interval=interval, blit=False, repeat=True)
        plt.tight_layout()
        plt.show()
        return ani

def random_closed_curve(
    num_points=20,
    radius=1.0,
    harmonics=20,
    noise_scale=0.35,
    seed=None
):
    """
    Generates a random smooth closed loop using a random Fourier radius.

    Parameters
    ----------
    num_points : number of control points returned
    radius : base radius of the shape
    harmonics : number of sinusoidal modes
    noise_scale : amplitude of randomness
    seed : random seed for reproducibility
    """

    rng = np.random.default_rng(seed)

    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)

    r = np.full(num_points, radius)

    for k in range(1, harmonics + 1):
        amp = noise_scale * rng.uniform(0.2, 1.0)
        phase = rng.uniform(0, 2*np.pi)
        r += amp * np.sin(k * angles + phase)

    points = [(r[i]*np.cos(angles[i]), r[i]*np.sin(angles[i])) for i in range(num_points)]

    return points

def main(num_points):
    # points = [(np.cos((2*np.pi*i) / num_points), np.sin((2*np.pi * i) / num_points)) for i in range(num_points)]
    # rng = np.random.default_rng(0)
    # angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    # r = 1 + 0.4 * rng.normal(size=len(angles))
    # points = [(r[i]*np.cos(angles[i]), r[i]*np.sin(angles[i])) for i in range(len(angles))] 
    points = random_closed_curve()
    active_points = points
    counter = 0

    while True:
        counter += 1
        s = t.time()
        xs, ys = build_parametric_spline(active_points, samples_per_seg=120)
        e = t.time()
        print(f"That took {e-s} seconds.")

        plot_points(active_points, xs, ys)
        plt.savefig(f"figs/plot_{counter}.png")
        plt.show()

        # NEW: build cams from the spline
        camsys = CamPairDesigner(xs, ys, base_radius=3.0, gain=1.0)
        camsys.plot_cams()
        camsys.plot_displacement_functions()
        camsys.plot_path_comparison()
        ani = camsys.animate()

        print("Current points:")
        print(f"{active_points}\n")
        print("What's your desired point index?\n")
        i = int(input())
        if i < 1:
            break
        print("What's your desired x-coordinate\n")
        x = float(input())
        print("What's your desired y-coordinate?\n")
        y = float(input())
        active_points[i-1] = (x, y)


# points = [(1,0), (0.5,np.sqrt(3)/2), (-0.5,np.sqrt(3)/2), (-1,0), (-0.5,-np.sqrt(3)/2), (0.5,-np.sqrt(3)/2)]
# plot_points(10)
main(20)