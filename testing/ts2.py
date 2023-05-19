import numpy as np
from reservoirpy.nodes import SklearnNode, Ridge, Input, Reservoir
from reservoirpy.utils.sklearn_helper import TransformInputSklearn, TransformOutputSklearn
# Function to generate synthetic time series data for forecasting
from matplotlib import pyplot as plt
import pdb
np.random.seed(0)
# Function to generate synthetic time series data for forecasting
def generate_time_series_forecasting_data(num_samples, input_time_steps, output_time_steps, noise_factor=0.1):
    total_time_steps = input_time_steps + output_time_steps
    full_data = np.sin(np.linspace(0, (num_samples + total_time_steps) * 2 * np.pi / num_samples, num_samples + total_time_steps)) + noise_factor * np.random.randn(num_samples + total_time_steps)

    X = np.zeros((num_samples, input_time_steps, 1))
    y = np.zeros((num_samples, output_time_steps, 1))

    for i in range(num_samples):
        X[i, :, 0] = full_data[i:i + input_time_steps]
        y[i, :, 0] = full_data[i + input_time_steps:i + total_time_steps]

    return X, y

# Generate the dataset
num_samples = 100
input_time_steps = 50
output_time_steps = 50
X, y = generate_time_series_forecasting_data(num_samples, input_time_steps, output_time_steps, noise_factor=0.1)
encoder = TransformInputSklearn()
X, y = encoder(X, y, task="regression")
X_train, y_train = X[:99], y[:99]
X_test, y_test = X[99:], y[99:]
source = Input()
reservoir = Reservoir(100, lr=0.5, sr=0.9)
readout = SklearnNode(method="Ridge", alpha=1e-4)
# readout = Ridge(ridge=1e-3)

esn = source >> reservoir >> readout
res = esn.fit(X_train, y_train)
y_pred = esn.run(X_test)

plt.figure(figsize=(10, 3))
plt.title("A sine wave and its future.")
plt.xlabel("$t$")
plt.plot(y_pred.squeeze(), label="Predicted sin(t)", color="blue")
plt.plot(y_test.squeeze(), label="Real sin(t+1)", color="red")
plt.legend()
plt.show()
###################################################
#Function to generate synthetic time series data for classification
import numpy as np

# Function to generate synthetic time series data for classification
def generate_time_series_classification_data(num_samples, num_time_steps, noise_factor=0.1):
    X = np.zeros((num_samples, num_time_steps, 1))
    y = np.zeros(num_samples)

    for i in range(num_samples):
        if i % 2 == 0:
            X[i, :, 0] = np.sin(np.linspace(0, 6 * np.pi, num_time_steps)) + noise_factor * np.random.randn(num_time_steps)
            y[i] = 0
        else:
            X[i, :, 0] = np.cos(np.linspace(0, 6 * np.pi, num_time_steps)) + noise_factor * np.random.randn(num_time_steps)
            y[i] = 1

    return X, y

# Generate the dataset
num_samples = 1000
num_time_steps = 100
X, y = generate_time_series_classification_data(num_samples, num_time_steps)
from reservoirpy.utils.sklearn_helper import TransformInputSklearn, TransformOutputSklearn
enocoder = TransformInputSklearn()
X, y = enocoder(X, y, task="classification")
from sklearn.metrics import accuracy_score
indices = np.arange(num_samples)
np.random.shuffle(indices)

X, y = X[indices], y[indices]
train_split = int(0.8*num_samples)
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

units = 100
leak_rate = 0.3
spectral_radius = 1.25
input_scaling = 1.0
connectivity = 0.1
input_connectivity = 0.2
seed = 1234


reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                      lr=leak_rate, rc_connectivity=connectivity,
                      input_connectivity=input_connectivity, seed=seed)

readout   = SklearnNode(method="RidgeClassifier")

esn = reservoir >> readout
res = esn.fit(X_train, y_train)
y_pred = esn.run(X_test)
pdb.set_trace()
y_pred, y_test = TransformOutputSklearn()(y_pred, y_test)
print(accuracy_score(y_test, y_pred)*100)
