import numpy as np
import pandas as pd 
import sys
sys.path.insert(0, '../')
from reservoirpy.nodes import Reservoir, Input, ScikitNodes
# from reservoirpy.datasets import japanese_vowels
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pdb
# data = load_iris()
from reservoirpy.datasets import mackey_glass
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron, LogisticRegression
from reservoirpy.nodes import Reservoir, Input, ScikitNodes, Input, Ridge
from sklearn.metrics import accuracy_score

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

source = Input()
reservoir = Reservoir(500, sr=0.9, lr=0.1)

readout = ScikitNodes(name="RidgeClassifier", alpha=1.0, tol=1e-3)
# readout = Ridge()
model = readout

# pdb.set_trace()
model = model.fit(X_train[:, :, None], Y_train[:, :, None])
Y_pred = model.run(X_test[:, :, None])
score = accuracy_score(Y_test, Y_pred)
print("Accuracy: ", f"{score * 100:.3f} %")


# readout = ScikitNodes(name="Perceptron")
# # model = source >> readout
# readout.fit(X_train, Y_train)
# y_pred = readout.run(X_test)
# Y_pred_class = np.argmax(y_pred, axis=1)
# score = accuracy_score(Y_test, Y_pred_class)
# print("Accuracy: ", f"{score * 100:.3f} %")