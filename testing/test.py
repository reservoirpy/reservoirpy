import numpy as np
import pandas as pd 
import sys
# sys.path.insert(0, '../')
# from reservoirpy.nodes import Reservoir, Input, ScikitNode
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pdb
from reservoirpy.datasets import mackey_glass
import matplotlib.pyplot as plt
# from sklearn.linear_model import Perceptron, LogisticRegrepytssion
from reservoirpy.nodes import SklearnNode, Input, Ridge, Reservoir
from reservoirpy.utils.sklearn_helper import check_sklearn_dim
from sklearn.metrics import accuracy_score

label2id = {"SITTING":0, "WALKING_DOWNSTAIRS":1,
"WALKING_UPSTAIRS":2, "STANDING":3, "WALKING":4,
"LAYING":5
}

source = Input()
reservoir = Reservoir(500, sr=0.9, lr=0.1)
method = "RidgeClassifier"
readout = SklearnNode(method=method, alpha=1e-3)
model = source >> reservoir >> readout

train = pd.read_csv("testing/data/train.csv")
X_train= train.iloc[:,:-2].to_numpy()
Y_train= train.iloc[:,-1].to_numpy()
Y_train = np.array([label2id[Y_train[i]] for i in range(len(Y_train))])
X_train, Y_train = check_sklearn_dim(X_train, Y_train, readout)
X_test, Y_test = check_sklearn_dim(X_train, Y_train, readout)
test= pd.read_csv("testing/data/test.csv")
X_test=test.iloc[:,:-2].to_numpy()
Y_test= test.iloc[:,-1].to_numpy()
Y_test = np.array([label2id[Y_test[i]] for i in range(len(Y_test))])
# import pdb;pdb.set_trace()
model = model.fit(X_train, Y_train)
Y_pred = model.run(X_test)

# Y_pred = np.array(Y_pred).squeeze()
# if method in ["LogisticRegression", "Perceptron", "SGDClassifier"]:
# 	score = accuracy_score(Y_test, Y_pred)
# else:
# 	score = accuracy_score(Y_test, np.argmax(Y_pred, axis=1))
# print("Accuracy: ", f"{score * 100:.3f} %")

