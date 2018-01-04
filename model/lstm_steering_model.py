from torch import nn

from ..model.take_first_module import TakeFirst


# A PyTorch LSTM model designed for computing a steering angle based on
# a constantly changing line of best fit of the center of the road
# Created by brendon-ai, January 2018


# Main function to create model using the number of training timesteps
def lstm_steering_model():
    # Create the neural network model using an LSTM followed by fully connected layers
    model = nn.Sequential(
        nn.LSTM(input_size=2, hidden_size=10, num_layers=1),
        TakeFirst(),
        nn.ReLU(),
        nn.Linear(10, 4),
        nn.ReLU(),
        nn.Linear(4, 1)
    )

    return model
