import numpy as np
import matplotlib.pyplot as plt
from reservoirpy.datasets import lorenz
from reservoirpy.activationsfunc import sigmoid
from scipy.interpolate import make_interp_spline

if __name__ == "__main__":
    dt = 0.025
    tot_time = 135

    n_timesteps = N = round(tot_time / dt)

    x0 = [17.67715816276679, 12.931379185960404, 43.91404334248268]
    X = lorenz(n_timesteps, x0=x0, h=dt, method="RK23")

    t = np.linspace(0, tot_time, n_timesteps)
    tt = np.linspace(0, tot_time, n_timesteps * 4)

    bspl = make_interp_spline(t, X, k=5, axis=0)

    XX = np.array([*bspl(tt)])
    XX = ((XX - XX.min()) / XX.ptp())*2 - 1.0
    XX[:,0] = XX[:, 0]
    XX[:,1] = XX[:, 1]
    XX[:,2] = XX[:, 2]
    NN = N * 4

    fig = plt.figure(figsize=(15, 15))
    ax  = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.axis(False)

    w = 1.0
    b = 3.0

    D = sigmoid(w*np.sqrt(XX[:,0]**2) + b)
    dds = []
    for i in range(NN-1):
        x, y, z = XX[i:i+2, 0], XX[i:i+2, 1], XX[i:i+2, 2]
        d = (sigmoid(w*np.sqrt(x[0]**2) + b) - D.min()) / D.ptp()
        dds.append(d)
        ax.plot(x, y, z, color=plt.cm.YlOrRd(int(255*d)), lw=1)

    plt.tight_layout()
    plt.show()

    D = sigmoid(1.0*np.sqrt(XX[:,0]**2) + 3.0)
    dds = []
    for i in range(NN-1):
        x, y, z = XX[i:i+2, 0], XX[i:i+2, 1], XX[i:i+2, 2]
        d = (sigmoid(1.0*np.sqrt(x[0]**2) + 3.0) - D.min()) / D.ptp()
        dds.append(d)
    plt.plot(np.array(dds)[:10000])

    fig.savefig("../lorenz63.png", dpi=300, transparent=True)
    fig.savefig("../lorenz63_w.png", dpi=300, transparent=True,
                facecolor="white")
    fig.savefig("../lorenz63_b.png", dpi=300, transparent=True,
                facecolor="black")
