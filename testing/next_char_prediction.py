import numpy as np
from reservoirpy.nodes import Reservoir, Input, SklearnNode
from reservoirpy.utils.sklearn_helper import check_sklearn_dim
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from tqdm import *
import pickle
import os
# from sklearn.linear_model import Ridge

data_path = "testing/data/shakespear.txt"

text = open(data_path, 'r').read()

alphabet = sorted(list(set(text)))
# Create mappings between characters and integers
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(alphabet)

# One-hot encode the integer data
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


# Prepare the data
if os.path.exists('testing/data/Xy.pickle'):
    with open('testing/data/Xy.pickle', 'rb') as handle:
        save = pickle.load(handle)
    X, y = save
else:
    # Create input-output pairs from the text data
    X = []
    y = []
    for i in trange(len(text) - 1):
        X.append(onehot_encoded[label_encoder.transform([text[i]])[0]])
        y.append(onehot_encoded[label_encoder.transform([text[i + 1]])[0]])

    X = np.array(X)
    y = np.array(y)

    save = [X, y]
    with open('testing/data/Xy.pickle', 'wb') as handle:
        pickle.dump(save, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Split data into training and testing
train_size = int(0.8 * len(X))
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:(train_size+1000)]
y_test = y[train_size:(train_size+1000)]
# Create the reservoir and readout nodes
source = Input()
reservoir = Reservoir(500)
readout = SklearnNode(name="SGDRegressor", max_iter=1000, tol=1e-3)


X_train, y_train = check_sklearn_dim(X_train, y_train, readout)

# Train the ESN model
# import pdb;pdb.set_trace()
states = reservoir.run(X_train[:1000000])

res = reservoir.fit(X_train[:100000], y_train[:100000])

# Make predictions

y_pred = esn.run(X_test)

# Decode predictions
y_pred_decoded = label_encoder.inverse_transform([np.argmax(pred) for pred in y_pred])
y_test_decoded = label_encoder.inverse_transform([np.argmax(true) for true in y_test])
print("".join(y_pred_decoded[:100]))
# Evaluate the model
accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
print(f"Accuracy: {accuracy:.2f}")
