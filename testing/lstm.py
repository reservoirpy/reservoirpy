import numpy as np
from tqdm import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from testing.phds import *
from testing.utils import *

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, hidden = self.lstm(x)
        output = self.linear(lstm_out[:, -1, :])
        return output

def train_model(model, train_loader, num_epochs=100):
    criterion = nn.MSELoss()  # or nn.L1Loss() for MAE
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in trange(num_epochs):
        for batch in train_loader:
            X_train_batch, y_train_batch = batch
            model.train()
            optimizer.zero_grad()
            output = model(X_train_batch)
            loss = criterion(output, y_train_batch[:, -1, :])
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

    return model


def generate_predictions(model, seed_input, num_generations):
    model.eval()
    current_input = seed_input
    predictions = []

    with torch.no_grad():
        for _ in range(num_generations):
            output = model(current_input)
            predictions.append(output)
            current_input = torch.cat((current_input[:, 1:, :], output.unsqueeze(1)), dim=1)

    return torch.stack(predictions, dim=1)


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
    forecast = 1
    seed_timesteps = 10
    window_size = 10
    X_train, y_train = create_sub_sequences(X_train, window_size=10)
    X_train, y_train = X_train.reshape(-1, window_size, 5), y_train.reshape(-1, 1, 5)
    X_test, y_test = to_forecasting(X_test, forecast=forecast)
    # Assuming X_train and y_train are your training data and labels
    # Cast your data to PyTorch tensors and add an extra dimension for the features
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    # Define dimensions
    input_dim = X_train.shape[-1]
    hidden_dim = 100  # for example
    output_dim = 5  # number of unique classes in y_train

    # Create an instance of the model
    model = LSTMModel(input_dim, hidden_dim, output_dim)

    # Train the model
    model = train_model(model, train_loader, num_epochs=20)

    # Generate predictions
    num_generations = y_test.shape[1]-seed_timesteps  # for example
    warming_inputs = X_test[:, :seed_timesteps, :]
    X_gen = generate_predictions(model, warming_inputs, num_generations)
    r2 = rsquare(y_test[:, (seed_timesteps):, -1], X_gen[:, :, -1]), nrmse(y_test[:, (seed_timesteps):, -1], X_gen[:, :, -1])
    plot_test_pred_data(inverse_transform_data(y_test[:, (seed_timesteps):, :], test_scaler), 
        inverse_transform_data(X_gen, test_scaler))
    print(r2)