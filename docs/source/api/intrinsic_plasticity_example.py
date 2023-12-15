# Author: Nathan Trouvain at 11/03/2022 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import matplotlib.pyplot as plt
import numpy as np

from reservoirpy.datasets import narma

steps = 1000
X = narma(steps)
X = (X - X.min()) / (X.ptp())
sigma = 0.1
from reservoirpy.nodes import IPReservoir

reservoir = IPReservoir(100, mu=0.0, sigma=sigma, sr=0.95, activation="tanh", epochs=10)
reservoir.fit(X, warmup=100).run(X[:100])
states = reservoir.run(X[100:])
from scipy.stats import norm


def heavyside(x):
    return 1.0 if x >= 0 else 0.0


def bounded(dist, x, mu, sigma, a, b):
    num = dist.pdf(x, loc=mu, scale=sigma) * heavyside(x - a) * heavyside(b - x)
    den = dist.cdf(b, loc=mu, scale=sigma) - dist.cdf(a, loc=mu, scale=sigma)
    return num / den


fig, (ax1) = plt.subplots(1, 1, figsize=(10, 7))
ax1.set_xlim(-1.0, 1.0)
ax1.set_ylim(0, 16)
for s in range(states.shape[1]):
    hist, edges = np.histogram(states[:, s], density=True, bins=200)
    points = [np.mean([edges[i], edges[i + 1]]) for i in range(len(edges) - 1)]
    ax1.scatter(points, hist, s=0.2, color="gray", alpha=0.25)
ax1.hist(
    states.flatten(),
    density=True,
    bins=200,
    histtype="step",
    label="Global activation",
    lw=3.0,
)
x = np.linspace(-1.0, 1.0, 200)
pdf = [bounded(norm, xi, 0.0, sigma, -1.0, 1.0) for xi in x]
ax1.plot(x, pdf, label="Target distribution", linestyle="--", lw=3.0)
ax1.set_xlabel("Reservoir activations")
ax1.set_ylabel("Probability density")
plt.legend()
plt.show()
