import numpy as np
import pandas as pd 
import sys
sys.path.insert(0, '../')
from reservoirpy.nodes import Reservoir, Input, LinearRegression, RidgeRegression, \
ElasticNet, Ridge, Lasso
# from reservoirpy.datasets import japanese_vowels
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pdb
# data = load_iris()
from reservoirpy.datasets import mackey_glass
import matplotlib.pyplot as plt
# from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
# def plot_pred(y_pred, y_test):
#     plt.figure(figsize=(10, 3))
#     plt.title("Visualizing prediction")
#     plt.xlabel("$t$")
#     plt.plot(y_pred, label="Predicted sin(t+1)", color="blue")
#     plt.plot(y_test, label="Real sin(t+1)", color="red")
#     plt.legend()
#     plt.show()

# X = mackey_glass(2000)
# X = 2 * (X - X.min()) / (X.max() - X.min()) - 1
# train_len = 1000

# X_train = X[:train_len]
# y_train = X[1 : train_len + 1]

# X_test = X[train_len : -1]
# y_test = X[train_len + 1:]

# reservoir = Reservoir(100, lr=0.5, sr=0.9)
# readout = RidgeRegression()
# # readout = Ridge()
# model = reservoir >> readout
# pdb.set_trace()
# model.fit(X_train, y_train)
# pred = model.run(X_test)
# plot_pred(pred, y_test)


label2id = {"SITTING":0, "WALKING_DOWNSTAIRS":1,
"WALKING_UPSTAIRS":2, "STANDING":3, "WALKING":4,
"LAYING":5
}
train = pd.read_csv("data/train.csv")
X_train= train.iloc[:,:-2].to_numpy()
Y_train= train.iloc[:,-1].to_numpy()
Y_train = np.array([label2id[Y_train[i]] for i in range(len(Y_train))])
test= pd.read_csv("data/test.csv")
X_test=test.iloc[:,:-2].to_numpy()
Y_test= test.iloc[:,-1].to_numpy()
Y_test = np.array([label2id[Y_test[i]] for i in range(len(Y_test))])

print(f" Classes: {set(Y_train)} ")
Y_train_onehot = np.zeros((Y_train.shape[0], len(set(Y_train))))
for i in range(len(Y_train)):
	Y_train_onehot[i, Y_train[i]] = 1
Y_train = Y_train_onehot

#################################################
# source = Input()
# reservoir = Reservoir(500, sr=0.9, lr=0.1)
# readout = ElasticNet(alpha=1e-4)

# # readout = Ridge(ridge=1e-6)
# model = source >> reservoir >> readout

# states_train = []
# for x in X_train:
#     states = reservoir.run(x, reset=True)
#     states_train.append(states[-1, np.newaxis])

# readout.fit(np.array(states_train).squeeze(), np.array(Y_train).squeeze())
# # readout.fit(X_train.squeeze(), np.array(Y_train).squeeze())
# states_test = []
# for x in X_test:
#     states = reservoir.run(x, reset=True)
#     states_test.append(states[-1, np.newaxis])

# y_pred = readout.run(np.array(states_test).squeeze())


# # y_pred = readout.run(X_test.squeeze())

# Y_pred_class = np.argmax(y_pred, axis=1)

# # Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]
# # Y_test_class = [np.argmax(y_t) for y_t in Y_test]
# score = accuracy_score(Y_test, Y_pred_class)

# print("Accuracy: ", f"{score * 100:.3f} %")
##############################################


source = Input()
readout = Lasso(alpha=1, tol=1e-4)
model = source >> readout
readout.fit(X_train, Y_train)
y_pred = readout.run(X_test)
Y_pred_class = np.argmax(y_pred, axis=1)
score = accuracy_score(Y_test, Y_pred_class)
print("Accuracy: ", f"{score * 100:.3f} %")