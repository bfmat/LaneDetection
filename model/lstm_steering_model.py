from keras.models import Sequential


# An LSTM model designed for computing a steering angle based on
# a constantly changing line of best fit of the center of the road
# Created by brendon-ai, November 2017


# Main function to create model
def lstm_steering_model():
    # Hyperbolic tangent activation function
    activation = 'tanh'

    # Initialize the Sequential model
    model = Sequential()

    # Compile the model with Adadelta optimizer
    model.compile(
        loss='mean_squared_error',
        optimizer='adadelta'
    )

    return model
