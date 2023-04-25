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
from reservoirpy.nodes import ScikitNode, Input, Ridge, Reservoir
from reservoirpy.scikit_helper import check_scikit_dim
from sklearn.metrics import accuracy_score

label2id = {"SITTING":0, "WALKING_DOWNSTAIRS":1,
"WALKING_UPSTAIRS":2, "STANDING":3, "WALKING":4,
"LAYING":5
}

source = Input()
reservoir = Reservoir(500, sr=0.9, lr=0.1)
readout = ScikitNode(name="Ridge", alpha=1.0, tol=1e-3)
model = source >> reservoir >> readout

train = pd.read_csv("testing/data/train.csv")
X_train= train.iloc[:,:-2].to_numpy()
Y_train= train.iloc[:,-1].to_numpy()
Y_train = np.array([label2id[Y_train[i]] for i in range(len(Y_train))])
X_train, Y_train = check_scikit_dim(X_train, Y_train, readout)

test= pd.read_csv("testing/data/test.csv")
X_test=test.iloc[:,:-2].to_numpy()
Y_test= test.iloc[:,-1].to_numpy()
Y_test = np.array([label2id[Y_test[i]] for i in range(len(Y_test))])

model = model.fit(X_train, Y_train)
Y_pred = model.run(X_test)
Y_pred = np.array(Y_pred).squeeze()
score = accuracy_score(Y_test, np.argmax(Y_pred, axis=1))
print("Accuracy: ", f"{score * 100:.3f} %")

