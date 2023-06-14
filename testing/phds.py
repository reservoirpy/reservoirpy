from reservoirpy.datasets import to_forecasting
from reservoirpy.nodes import Reservoir, SklearnNode, Ridge, Input
from reservoirpy.observables import nrmse, rsquare, mse
import reservoirpy as rpy
from testing.utils import *
rpy.verbosity(0)

def get_esn():
    units = 100
    leak_rate = 0.99
    spectral_radius = 0.50
    input_scaling = 0.38
    connectivity = 0.1
    input_connectivity = 0.2
    regularization = 1e-8
    seed = 1234

    reservoir = Reservoir(units, input_scaling=input_scaling, sr=spectral_radius,
                          lr=leak_rate, rc_connectivity=connectivity,
                          input_connectivity=input_connectivity, seed=seed)
    readout = SklearnNode(method="Ridge", alpha=1e-6)
    source = Input()
    esn = source >> reservoir >> readout
    return esn

def create_minibatches(arr1, arr2, batch_size):
    """
    Generator function that creates mini-batches from two input arrays.

    Parameters:
    arr1, arr2: Input numpy arrays of shape (N, T, D), where N is number of data points, 
                T is the number of time steps, and D is the feature dimension.
    batch_size: The size of the mini-batches.

    Yields:
    batches of data from arr1 and arr2
    """

    assert arr1.shape == arr2.shape, "Input arrays must have the same shape."
    
    indices = np.random.permutation(arr1.shape[0])
    arr1 = arr1[indices]
    arr2 = arr2[indices]
    arr1, arr2 = arr1.reshape(-1, 10, 5), arr2.reshape(-1, 10, 5)
    num_batches = arr1.shape[0] // batch_size
    for i in range(num_batches):
        yield (arr1[i * batch_size:(i + 1) * batch_size], 
               arr2[i * batch_size:(i + 1) * batch_size])
        
    if arr1.shape[0] % batch_size != 0:
        yield (arr1[num_batches * batch_size:], 
               arr2[num_batches * batch_size:])

if __name__ == "__main__":
    opt = parse_option()
    # read the csv file
    df = pd.read_csv('testing/data/simulated_data_4reservoirpy.csv', delimiter=';')
    # get the unique ids
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
    X_train, _ = rescale(X_train)
    X_test, test_scaler = rescale(X_test)
    esn = get_esn()
    if opt.mode == "simple":
        x, y = to_forecasting(X_train, forecast=opt.forecast)
        num_timesteps = x.shape[1]
        train_ts = opt.tts
        test_ts = 56
        if opt.only_v == 1:
            X_train, y_train = x[:, :train_ts, :-1], y[:, :train_ts, -1]
            X_test, y_test = x[:, -test_ts:, :-1], y[:, -test_ts:, -1]
            y_train, y_test = y_train[:, :, None], y_test[:, :, None]
            esn = esn.fit(X_train, y_train)
            y_pred = esn.run(X_test)
            r2 = rsquare(y_test, y_pred), nrmse(y_test, y_pred)
        else:
            X_train, y_train = x[:, :train_ts, :], y[:, :train_ts, :]
            X_test, y_test = x[:, -test_ts:, :], y[:, -test_ts:, :]
            esn = esn.fit(X_train, y_train)
            y_pred = esn.run(X_test)
            y_pred = np.array(y_pred)
            r2 = rsquare(y_test[:, :, -1], y_pred[:, :, -1]), nrmse(y_test[:, :, -1], y_pred[:, :, -1])
        print(r2[1])
    elif opt.mode == "gen":
        forecast = 1
        seed_timesteps = 10
        window_size = 10
        X_train, y_train = create_sub_sequences(X_train, window_size=window_size, reservoir=True)
        X_test, y_test = to_forecasting(X_test, forecast=forecast)
        resume = False
        if resume:
            esn = np.load("esn.npy")
        else:
            # for x_mini, y_mini in create_minibatches(X_train, y_train, batch_size=1024):
            X_train, y_train = X_train.reshape(-1, 10, 5), y_train.reshape(-1, 10, 5)
            esn = esn.fit(X_train, y_train)
            np.save("esn.npy", esn)
        warming_inputs = X_test[:, :seed_timesteps, :]
        warming_out = esn.run(warming_inputs)
        plot_test_pred_data(y_test[:, :seed_timesteps], np.array(warming_out))
        
        nb_generations = (X_test.shape[1] - seed_timesteps)
        X_gen = np.zeros((X_test.shape[0], nb_generations, 5))
        y = np.array(warming_out)
        for t in range(nb_generations):  # generation
            if forecast == 1:
                y = np.array(esn.run(y))
                y_last = y[:, -1, :]
                X_gen[:, t, :] = y_last
                y = np.concatenate((y[:, 1:, :], y_last[:, None, :]), axis=1)
            else:
                start = t*forecast
                end = (t*forecast)+forecast
                y = esn.run(y)
                X_gen[:, start:end, :] = y
        r2 = rsquare(y_test[:, seed_timesteps:, -1], X_gen[:, :, -1]), mse(y_test[:, seed_timesteps:, -1], X_gen[:, :, -1])
        plot_test_pred_data(inverse_transform_data(y_test[:, seed_timesteps:, :], test_scaler), 
            inverse_transform_data(X_gen, test_scaler))
        print(r2)
        