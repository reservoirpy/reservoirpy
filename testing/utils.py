import os
import argparse
import numpy as np
import pandas as pd

import pdb
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

def parse_option():
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--mode", type=str)
    parser.add_argument("--only_v", type=int)
    parser.add_argument("--forecast", type=int)
    parser.add_argument("--tts", type=int)
    parser.add_argument("--data_type", type=str)
    parser.add_argument("--err", type=int)
    opt = parser.parse_args()
    return opt

def rescale(matrix):
    # Reshape the matrix to 2D (N*T, D) for normalization
    N, T, D = matrix.shape
    reshaped_matrix = np.reshape(matrix, (N * T, D))

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler to the reshaped matrix and transform the data
    normalized_matrix = scaler.fit_transform(reshaped_matrix)

    # Reshape the normalized matrix back to the original shape
    normalized_matrix = np.reshape(normalized_matrix, (N, T, D))

    return normalized_matrix, scaler

def inverse_transform_data(normalized_matrix, scaler):
    # Reshape the matrix to 2D (N*T, D) for inverse transformation
    N, T, D = normalized_matrix.shape
    reshaped_matrix = np.reshape(normalized_matrix, (N * T, D))

    # Perform the inverse transformation using the provided scaler
    inverse_transformed_matrix = scaler.inverse_transform(reshaped_matrix)

    # Reshape the inverse transformed matrix back to the original shape
    inverse_transformed_matrix = np.reshape(inverse_transformed_matrix, (N, T, D))

    return inverse_transformed_matrix

def to_forecasting(X, forecast):
    """
    Generate input (x) and target (y) time series for forecasting.

    Args:
        X (ndarray): The input data of shape (num_samples, seq_length, num_features).
        forecast (int): The number of time steps to forecast ahead.

    Returns:
        x (ndarray): The input time series of shape (num_samples, seq_length - forecast, num_features).
        y (ndarray): The target forecasting time series of shape (num_samples, forecast, num_features).
    """
    num_samples, seq_length, num_features = X.shape

    # Determine the dimensions of the input (x) and target (y) time series
    input_length = seq_length - forecast
    # Generate input (x) and target (y) time series
    x = X[:, :input_length, :]
    y = X[:, forecast:seq_length, :]

    return x, y

def create_input_sequences(df, opt):
    # Extract the T, I, P, E, and V columns from the DataFrame
    if opt.err == 0:
        T = df['T'].values
        I = df['I'].values
        P = df['P'].values
        E = df['E'].values
        V = df['V'].values
    else:
        T = df['Terr'].values
        I = df['Ierr'].values
        P = df['Perr'].values
        E = df['Eerr'].values
        V = df['Verr'].values
    input_sequence = np.vstack((T, I, P, E, V))
    return input_sequence.transpose()

def create_sub_sequences(X, window_size=10, reservoir=False):
    """
    Create input sequences and corresponding targets for a given dataset and window size.

    Parameters:
    X: Input dataset of shape (N, T, D), where N is number of data points, 
       T is the number of time steps, and D is the feature dimension.
    window_size: Size of the window to use when creating the sequence.

    Returns:
    X_seq: Input sequences for the model, of shape (N, T - window_size, window_size, D).
    y_seq: Corresponding targets for each input sequence, of shape (N, T - window_size, D).
    """
    N, T, D = X.shape
    X_seq = np.empty((N, T - window_size, window_size, D))
    if reservoir:
        y_seq = np.empty((N, T - window_size, window_size, D))
    else:
        y_seq = np.empty((N, T - window_size, D))

    for i in range(T - window_size):
        X_seq[:, i, :, :] = X[:, i:i+window_size, :]
        if reservoir:
            y_seq[:, i, :, :] = X[:, i+1:i+window_size+1, :]
        else:
            y_seq[:, i, :] = X[:, i+window_size, :]
    return X_seq, y_seq



def split_train_test_data(X):
    # Get the number of patients and time instances
    np.random.seed(42)
    num_patients, num_time_instances, num_variables = X.shape
    group1_indices = np.arange(0, 500)
    group2_indices = np.arange(500, 1000)
    test_group1_indices = np.random.choice(group1_indices, size=100, replace=False)
    test_group2_indices = np.random.choice(group2_indices, size=100, replace=False)
    train_group1_indices = np.setdiff1d(group1_indices, test_group1_indices)
    train_group2_indices = np.setdiff1d(group2_indices, test_group2_indices)
    train_indices = np.concatenate([train_group1_indices, train_group2_indices])
    test_indices = np.concatenate([test_group1_indices, test_group2_indices])
    X_train, X_test = X[train_indices], X[test_indices]
    return X_train, X_test



def add_missing_data(array, p=0.1):
    # Convert the array to a pandas DataFrame
    df = pd.Panel(array.swapaxes(1, 2)).to_frame()
    
    # Randomly replace some data points with 'NA'
    for col in df.columns:
        df.loc[random.sample(df.index.tolist(), int(p * len(df))), col] = np.nan

    # Convert the DataFrame back to a numpy array
    return df.to_xarray().values.swapaxes(1, 2)


def plot_test_pred_data(X_test, X_pred):
    # Get the number of individuals and time instances
    num_individuals, num_time_instances, num_variables = X_test.shape
    
    # Create a color palette for test and pred plots
    test_color_palette = plt.cm.get_cmap('Blues', 10+1)
    pred_color_palette = plt.cm.get_cmap('Reds', 10+1)
    
    # Plot individual test curves
    for i in range(5):
        test_individual_data = X_test[i, :, -1]  # Select the last dimension for plotting
        plt.plot(range(num_time_instances), test_individual_data, color=test_color_palette(i), linewidth=0.5)
    
    # Plot the mean test curve in bold blue
    mean_test_data = np.mean(X_test[:, :, -1], axis=0)
    plt.plot(range(num_time_instances), mean_test_data, color='blue', linewidth=2, label='Test Mean')
    
    # Plot individual pred curves
    for i in range(10):
        pred_individual_data = X_pred[i, :, -1]  # Select the last dimension for plotting
        plt.plot(range(num_time_instances), pred_individual_data, color=pred_color_palette(i), linewidth=0.5)
    
    # Plot the mean pred curve in bold red
    mean_pred_data = np.mean(X_pred[:, :, -1], axis=0)
    plt.plot(range(num_time_instances), mean_pred_data, color='red', linewidth=2, label='Pred Mean')
    
    # Set plot labels and title
    plt.xlabel('Time')
    plt.ylabel('Variable')
    plt.title('Test and Pred Data')
    
    # Add legend
    plt.legend()
    
    # Show the plot
    plt.show()