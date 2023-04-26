import numpy as np
from reservoirpy.nodes import Reservoir, Input, ScikitNode
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
# from sklearn.linear_model import Ridge

# Prepare the data
text = "hello world this is a simple text data for \
testing the scikit node with esn for next character prediction"

alphabet = sorted(list(set(text)))

# Create mappings between characters and integers
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(alphabet)

# One-hot encode the integer data
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# Create input-output pairs from the text data
X = []
y = []
for i in range(len(text) - 1):
    X.append(onehot_encoded[label_encoder.transform([text[i]])[0]])
    y.append(onehot_encoded[label_encoder.transform([text[i + 1]])[0]])

X = np.array(X)
y = np.array(y)

# Split data into training and testing
train_size = int(0.8 * len(X))
X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

# Create the reservoir and readout nodes
source = Input()
reservoir = Reservoir(100)
readout = ScikitNode(name="Ridge", alpha=1.0, tol=1e-3)
esn = reservoir >> readout

# Train the ESN model
import pdb;pdb.set_trace()
esn.fit(X_train, y_train)

# Make predictions
y_pred = esn.run(X_test)

# Decode predictions
y_pred_decoded = label_encoder.inverse_transform([np.argmax(pred) for pred in y_pred])
y_test_decoded = label_encoder.inverse_transform([np.argmax(true) for true in y_test])

# Evaluate the model
accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
print(f"Accuracy: {accuracy:.2f}")
