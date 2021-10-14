import numpy as np

from sklearn import metrics

from reservoirpy import ESN, mat_gen
from reservoirpy.hyper import research, plot_hyperopt_report


if __name__ == "__main__":

    def objective(dataset, config, *, iss, N, sr, leak, ridge):

        # unpack train and test data, with target values.
        train_data, test_data = dataset
        x_train, y_train = train_data
        x_test, y_test = test_data

        x_train, y_train = x_train.reshape(1, -1), y_train.reshape(1, -1)
        x_test, y_test = x_test.reshape(1, -1), y_test.reshape(1, -1)

        nb_features = x_train.shape[1]

        instances = config["instances_per_trial"]

        losses = []; rmse = [];
        for n in range(instances):
            # builds an ESN given the input parameters
            W = mat_gen.fast_spectral_initialization(N=N, sr=sr)

            Win = mat_gen.generate_input_weights(nbr_neuron=N, dim_input=nb_features,
                                                input_bias=True, input_scaling=iss)


            reservoir = ESN(lr=leak, W=W, Win=Win, input_bias=True, ridge=ridge)


            # train and test the model
            reservoir.train(inputs=[x_train], teachers=[y_train],
                            wash_nr_time_step=20, verbose=False, workers=1)

            outputs, _ = reservoir.run(inputs=[x_test], verbose=False, workers=1)

            losses.append(metrics.mean_squared_error(outputs[0], y_test))
            rmse.append(metrics.mean_squared_error(outputs[0], y_test, squared=False))

        # returns a dictionnary of metrics. The 'loss' key is mandatory when
        # using Hyperopt.
        return {'loss': np.mean(losses),
                'rmse': np.mean(rmse)}

    # 10,000 timesteps of Mackey-Glass timeseries
    mackey_glass = np.loadtxt('MackeyGlass_t17.txt').reshape(-1, 1)

    # split data
    train_frac = 0.6
    train_start, train_end = 0, int(train_frac * len(mackey_glass))
    test_start, test_end = train_end, len(mackey_glass) - 2

    # pack it
    train_data = (mackey_glass[train_start:train_end], mackey_glass[train_start+1:train_end+1])
    test_data = (mackey_glass[test_start:test_end], mackey_glass[test_start+1:test_end+1])

    dataset = (train_data, test_data)

    x_train, y_train = train_data

    # run the random search
    best = research(objective, dataset, "mackeyglass-config.json", "report")

    # plot the results (fetch results from the report directory)
    fig = plot_hyperopt_report("report/hyperopt-mackeyglass", params=["sr", "leak", "ridge"], metric='loss')

    fig.savefig("test.png")
