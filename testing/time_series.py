import numpy as np
from matplotlib import pyplot as plt
import pdb
import sys
sys.path.insert(0, '../')
from reservoirpy.nodes import Reservoir, Ridge, Input, SklearnNode
from sklearn.linear_model import SGDRegressor
from pympler.asizeof import asizeof

X = np.sin(np.linspace(0, 6*np.pi, 10000)).reshape(-1, 1)

train_size = int(0.8 * len(X))
X_train = X[:train_size-1]
Y_train = X[1:train_size]
X_test = X[train_size:train_size+50]
Y_test = X[train_size+1:train_size+51]
pdb.set_trace()
reservoir = Reservoir(500, lr=0.5, sr=0.9)
readout = SGDRegressor()

def time_series_generator(data, chunk_size=100):
    num_chunks = (len(data) + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(data))
        yield data[start_index:end_index]


# esn_model = reservoir >> ridge
# esn_model = esn_model.fit(X_train, Y_train, warmup=10)
# Y_pred = esn_model.run(X[50:])
# Y_pred = np.array(Y_pred)

for X_batch, y_batch in zip(time_series_generator(X_train, 100), time_series_generator(Y_train, 100)):
    # Process the chunk of 100 time steps
    batch_states = reservoir.run(X_batch)
    # pdb.set_trace()
    readout.partial_fit(batch_states, y_batch)


test_states = reservoir.run(X_test)
Y_pred = readout.predict(test_states)

plt.figure(figsize=(10, 3))
plt.title("A sine wave and its future.")
plt.xlabel("$t$")
# plt.plot(Y_pred, label="Predicted sin(t+1)", color="blue")
plt.plot(Y_test, label="Real sin(t+1)", color="red")
plt.legend()
plt.show()

