from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../')
import pdb
from reservoirpy.datasets import japanese_vowels
from reservoirpy import set_seed, verbosity
from reservoirpy.observables import nrmse, rsquare
from reservoirpy.nodes import Reservoir, Input, ScikitNodes, Input, Ridge
from sklearn.metrics import accuracy_score

set_seed(42)
verbosity(0)

# repeat_target ensure that we obtain one label per timestep, and not one label per utterance.
X_train, Y_train, X_test, Y_test = japanese_vowels()

source = Input()
reservoir = Reservoir(500, sr=0.9, lr=0.1)
readout = ScikitNodes(name="RidgeClassifier")
model = readout


# states_train = []
# for x in X_train:
#     states = reservoir.run(x, reset=True)
#     states_train.append(states[-1, np.newaxis])
# readout.fit(np.array(states_train).squeeze(), np.array(Y_train).squeeze())
# import pdb;pdb.set_trace()
# readout.fit(np.array(states_train).squeeze(), np.argmax(np.array(Y_train).squeeze(), axis=1))

states_test = []
for x in X_test:
    states = reservoir.run(x, reset=True)
    states_test.append(states[-1, np.newaxis])
y_pred = readout.run(np.array(states_test).squeeze())
Y_pred_class = np.argmax(y_pred, axis=1)

# Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]
Y_test_class = [np.argmax(y_t) for y_t in Y_test]

score = accuracy_score(Y_test_class, Y_pred_class)

print("Accuracy: ", f"{score * 100:.3f} %")