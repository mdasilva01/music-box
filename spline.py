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

def main(num_points):
    points = [(np.cos((2*np.pi*i) / num_points), np.sin((2*np.pi * i) / num_points)) for i in range(num_points)]
    og_points = points
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