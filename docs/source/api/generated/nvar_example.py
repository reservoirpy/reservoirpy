# Author: Nathan Trouvain at 11/03/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import matplotlib.pyplot as plt
import numpy as np

from reservoirpy.nodes import NVAR, Ridge

nvar = NVAR(delay=2, order=2, strides=1)
readout = Ridge(3, ridge=2.5e-6)
model = nvar >> readout
from reservoirpy.datasets import lorenz

X = lorenz(5400, x0=[17.677, 12.931, 43.914], h=0.025, method="RK23")
Xi = X[:600]
dXi = X[1:601] - X[:600]
model = model.fit(Xi, dXi)
model.fit(Xi, dXi, warmup=200)
u = X[600]
res = np.zeros((5400 - 600, readout.output_dim))
for i in range(5400 - 600):
    u = u + model(u)
    res[i, :] = u

N = 5400 - 600
Y = X[600:]

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(121, projection="3d")
ax.set_title("Generated attractor")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")
ax.grid(False)

for i in range(N - 1):
    ax.plot(
        res[i : i + 2, 0],
        res[i : i + 2, 1],
        res[i : i + 2, 2],
        color=plt.cm.magma(255 * i // N),
        lw=1.0,
    )

ax2 = fig.add_subplot(122, projection="3d")
ax2.set_title("Real attractor")
ax2.grid(False)

for i in range(N - 1):
    ax2.plot(
        Y[i : i + 2, 0],
        Y[i : i + 2, 1],
        Y[i : i + 2, 2],
        color=plt.cm.magma(255 * i // N),
        lw=1.0,
    )

plt.show()
