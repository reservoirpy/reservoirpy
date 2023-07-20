from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../')
import pdb
from reservoirpy.datasets import japanese_vowels
from reservoirpy import set_seed, verbosity
from reservoirpy.observables import nrmse, rsquare
from reservoirpy.nodes import Reservoir, Input, ScikitNode, Input, Ridge
from reservoirpy.scikit_helper import check_scikit_dim
from sklearn.metrics import accuracy_score

set_seed(42)
verbosity(0)

# repeat_target ensure that we obtain one label per timestep, and not one label per utterance.
X_train, Y_train, X_test, Y_test = japanese_vowels()

source = Input()
reservoir = Reservoir(500, sr=0.9, lr=0.1)
readout = ScikitNode(name="Ridge")

states_train = []
for x in X_train:
    states = reservoir.run(x, reset=True)
    states_train.append(states[-1, np.newaxis])
states_train, Y_train = np.array(states_train).squeeze(), np.array(Y_train).squeeze()
states_train, Y_train = check_scikit_dim(states_train, Y_train, readout)
res = readout.fit(states_train, Y_train)
states_test = []
for x in X_test:
    states = reservoir.run(x, reset=True)
    states_test.append(states[-1, np.newaxis])
y_pred = readout.run(np.array(states_test).squeeze())

Y_pred_class = np.argmax(y_pred, axis=1)
Y_test_class = [np.argmax(y_t) for y_t in Y_test]
score = accuracy_score(Y_test_class, Y_pred_class)
print("Accuracy: ", f"{score * 100:.3f} %")