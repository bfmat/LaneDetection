from keras.layers import Dense, LSTM
from keras.models import Sequential


# An LSTM model designed for computing a steering angle based on
# a constantly changing line of best fit of the center of the road
# Created by brendon-ai, November 2017


# Main function to create model using the number of training timesteps
def lstm_steering_model(training_timesteps):
    # Hyperbolic tangent activation function
    activation = 'tanh'

    # Initialize the Sequential model
    model = Sequential()

    # Add a single LSTM layer with 10 output neurons
    model.add(LSTM(10, input_shape=(None, 2)))

    # Add two fully connected layers, bringing the output space down to a single neuron representing steering angle
    model.add(Dense(4, activation=activation))
    model.add(Dense(1, activation=activation))

    # Compile the model with Adadelta optimizer
    model.compile(
        loss='mean_squared_error',
        optimizer='adadelta'
    )

    return model
