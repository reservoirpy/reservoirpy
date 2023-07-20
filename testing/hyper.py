import pdb
import numpy as np
import pandas as pd 
import sys
sys.path.insert(0, '../')
from reservoirpy.nodes import Reservoir, Input, LinearRegression, RidgeRegression
from reservoirpy.observables import nrmse, rsquare
from reservoirpy.hyper import research
from reservoirpy.hyper import plot_hyperopt_report

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# First read the data using pandas and convert into numpy format
train = pd.read_csv("data/train.csv")
X_train= train.iloc[:,:-2].to_numpy()
Y_train= train.iloc[:,-1]
test= pd.read_csv("data/test.csv")
X_test=test.iloc[:,:-2].to_numpy()
Y_test= test.iloc[:,-1]

X_train = 2 * (X_train - X_train.min()) / (X_train.max() - X_train.min()) - 1
X_test = 2 * (X_test - X_test.min()) / (X_test.max() - X_test.min()) - 1
label2id = {"SITTING":0, "WALKING_DOWNSTAIRS":1,
"WALKING_UPSTAIRS":2, "STANDING":3, "WALKING":4,
"LAYING":5
}
Y_train = np.array([label2id[Y_train[i]] for i in range(len(Y_train))])
Y_test = np.array([label2id[Y_test[i]] for i in range(len(Y_test))])

print(f" Classes: {set(Y_train)} ")
Y_train_onehot = np.zeros((Y_train.shape[0], len(set(Y_train))))
for i in range(len(Y_train)):
	Y_train_onehot[i, Y_train[i]] = 1
Y_train = Y_train_onehot

dataset = ((X_train, Y_train), (X_test, Y_test))

def objective(dataset, config, *, tol, seed):

	# This step may vary depending on what you put inside 'dataset'
	train_data, validation_data = dataset
	X_train, y_train = train_data
	X_val, y_val = validation_data
	# You can access anything you put in the config
	# file from the 'config' parameter.
	instances = config["instances_per_trial"]
	# The seed should be changed across the instances,
	# to be sure there is no bias in the results
	# due to initialization.
	variable_seed = seed

	losses = []; r2s = [];
	scores = []
	# for n in range(instances):
		# Build your model given the input parameters
	source = Input()
	reservoir = Reservoir(500,
						  sr=0.9,
						  lr=0.1,
						  seed=variable_seed)

	readout = RidgeRegression()

	model = source >> reservoir >> readout


	# Train your model and test your model.
	states_train = []
	for x in X_train:
		states = reservoir.run(x, reset=True)
		states_train.append(states[-1, np.newaxis])
	readout.fit(np.array(states_train).squeeze(), np.array(Y_train).squeeze())
	states_test = []
	for x in X_test:
		states = reservoir.run(x, reset=True)
		states_test.append(states[-1, np.newaxis])

	y_pred = readout.run(np.array(states_test).squeeze())
	predictions = np.argmax(y_pred, axis=1)
	loss = nrmse(Y_test, predictions, norm_value=np.ptp(X_train))
	r2 = rsquare(Y_test, predictions)
	
	score = accuracy_score(Y_test, predictions)
	# Change the seed between instances
	variable_seed += 1

	losses.append(loss)
	r2s.append(r2)
	scores.append(score)
	# Return a dictionnary of metrics. The 'loss' key is mandatory when
	# using hyperopt.
	return {'loss': np.mean(losses),
			'r2': np.mean(r2s),
			'score':np.mean(scores)}


if __name__ == "__main__":
	hyperopt_config = {
		"exp": f"hyperopt-scikit", # the experimentation name
		"hp_max_evals": 5,             # the number of differents sets of parameters hyperopt has to try
		"hp_method": "random",           # the method used by hyperopt to chose those sets (see below)
		"seed": 42,                      # the random state seed, to ensure reproducibility
		"instances_per_trial": 3,        # how many random ESN will be tried with each sets of parameters
		"hp_space": {                    # what are the ranges of parameters explored
	        "tol": ["loguniform", 1e-2, 10],        # and so is the regularization parameter.
	        "seed": ["choice", 1234]          # an other random seed for the ESN initialization
		}
	}


	import json

	# we precautionously save the configuration in a JSON file
	# each file will begin with a number corresponding to the current experimentation run number.
	with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
		json.dump(hyperopt_config, f)
	best = research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")
	print(best)