import numpy as np
from reservoirpy.nodes import SklearnNode, Ridge, Input, Reservoir
# Function to generate synthetic time series data for forecasting
def generate_time_series_forecasting_data(num_samples, input_time_steps, output_time_steps, noise_factor=0.1):
   import numpy as np

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
num_samples = 1000
input_time_steps = 50
output_time_steps = 50
X, y = generate_time_series_forecasting_data(num_samples, input_time_steps, output_time_steps)

from reservoirpy.utils.sklearn_helper import check_sklearn_dim
source = Input()
reservoir = Reservoir(100)
readout = SklearnNode(method="Ridge", alpha=1e-3)
esn = source >> reservoir >> readout
print(X.shape)
print(y.shape)
X, y = check_sklearn_dim(X, y, readout)
print(X.shape)
print(y.shape)
X_train, y_train = X[:900], y[:900]
X_test, y_test = X[900:], y[900:]

source = Input()
reservoir = Reservoir(100)
readout = SklearnNode(method="Ridge", alpha=1e-3)
esn = source >> reservoir >> readout
res = esn.fit(X_train, y_train)