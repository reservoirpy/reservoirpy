import os
import json
import numpy as np
import pandas as pd
import pdb
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import reservoirpy as rpy
from reservoirpy.datasets import to_forecasting
from reservoirpy.nodes import Reservoir, SklearnNode, Ridge, Input
from reservoirpy.observables import nrmse, rsquare, mse
from reservoirpy.hyper import research
from reservoirpy.hyper import plot_hyperopt_report

from testing.phds import *
from testing.utils import *
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
rpy.verbosity(0)
def objective(dataset, config, *, iss, N, sr, lr, alpha, seed):

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
	for n in range(instances):
		# Build your model given the input parameters
		reservoir = Reservoir(N,
							  sr=sr,
							  lr=lr,
							  inut_scaling=iss,
							  seed=variable_seed)

		readout = SklearnNode(method="Ridge", alpha=alpha)

		model = reservoir >> readout


		# Train your model and test your model.
		predictions = model.fit(X_train, y_train) \
						   .run(X_test)
		predictions = np.array(predictions)
		loss = mse(y_test[:, :, -1], predictions[:, :, -1])
		r2 = rsquare(y_test[:, :, -1], predictions[:, :, -1])

		# Change the seed between instances
		variable_seed += 1

		losses.append(loss)
		r2s.append(r2)

	# Return a dictionnary of metrics. The 'loss' key is mandatory when
	# using hyperopt.
	return {'loss': np.mean(losses),
			'r2': np.mean(r2s)}

if __name__ == "__main__":
	# get the unique ids
	opt = parse_option()
	seed_timesteps = 10
	forecast = 1
	df = pd.read_csv('testing/data/simulated_data_4reservoirpy.csv', delimiter=';')
	unique_ids = df['Id'].unique()
	unique_groups = df["Group"].unique()
	# loop over the unique ids and create a csv file for each
	X = []
	for i, uid in enumerate(unique_ids):
		sub_df = df[df['Id'] == uid]
		x = create_input_sequences(sub_df, opt)
		X.append(x)
	X = np.array(X)
	X_train, X_test = split_train_test_data(X)
	X_train, _ =  rescale(X_train)
	X_test, _ =  rescale(X_test)


	hyperopt_config = {
	"exp": f"hyperopt-hiv-sr-lr-iss",
	"hp_max_evals": 50,
	"hp_method": "random",
	"seed": 42,
	"instances_per_trial": 3,
	"hp_space": {
		"N": ["choice", 100],
		"sr": ["choice", 0.5],
		"lr": ["choice", 1], 
		"iss": ["loguniform", 0.2, 0.7],
		"alpha": ["choice", 1e-6],
		"seed": ["choice", 1234]}
		}
	root = "testing/runs"
	save_path = os.path.join(root, f"{hyperopt_config['exp']}")
	os.makedirs(save_path, exist_ok=True)
	config_dir = os.path.join(root, "configs")
	os.makedirs(config_dir, exist_ok=True)
	

	X_train, y_train = to_forecasting(X_train, forecast=forecast)
	train_subset = np.random.choice(X_train.shape[0], size=200, replace=False)
	X_train, y_train = X_train[train_subset], y_train[train_subset]
	X_train, y_train = create_sub_sequences(X_train, window_size=10, reservoir=True)
	X_train, y_train = X_train.reshape(-1, 10, 5), y_train.reshape(-1, 10, 5)
	X_test, y_test = to_forecasting(X_test, forecast=forecast)
	X_test, y_test = X_test[:, :seed_timesteps, :], y_test[:, :seed_timesteps, :]
	dataset = ((X_train, y_train), (X_test, y_test))
	
	config_path = os.path.join(config_dir, f"{hyperopt_config['exp']}.config.json")
	with open(config_path, "w+") as f:
		json.dump(hyperopt_config, f, indent=4)
	
	best = research(
			objective, 
			dataset, 
			config_path=config_path, 
			report_path=f"{root}")
	print(best)
	fig = plot_hyperopt_report(
		f"{save_path}", 
		("sr", "lr", "iss"), 
		metric="r2")
	fig.savefig(f"{save_path}/plot.png")