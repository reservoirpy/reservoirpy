import numpy as np
from matplotlib import pyplot as plt
import pdb
import sys
sys.path.insert(0, '../')
from reservoirpy.nodes import Reservoir, Ridge, Input, ScikitNode


X = np.sin(np.linspace(0, 6*np.pi, 100)).reshape(-1, 1)
X = np.array([X, X])
X_train = X[:, :50]
Y_train = X[:, 1:51]
reservoir = Reservoir(100, lr=0.5, sr=0.9)
# ridge = Ridge(ridge=1e-7)
ridge = ScikitNode(name="Ridge", alpha=1.0, tol=1e-3)
esn_model = reservoir >> ridge

esn_model = esn_model.fit(X_train, Y_train, warmup=10)
Y_pred = esn_model.run(X[:, 50:])
Y_pred = np.array(Y_pred)

plt.figure(figsize=(10, 3))
plt.title("A sine wave and its future.")
plt.xlabel("$t$")
plt.plot(Y_pred[0], label="Predicted sin(t+1)", color="blue")
plt.plot(X[0, 51:], label="Real sin(t+1)", color="red")
plt.legend()
plt.show()

