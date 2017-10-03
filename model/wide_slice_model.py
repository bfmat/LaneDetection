from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D


# A convolutional neural network model intended to predict the lateral position of an object in a wide, short image
# Created by brendon-ai, September 2017


def wide_slice_model(slice_dimensions):

    # Hyperbolic tangent activation function
    activation = 'tanh'

    # Initialize the sequential model
    model = Sequential()

    # Input shape is provided, but add the channel axis to it
    input_shape = slice_dimensions + (3,)

    # Four convolutional layers
    model.add(Conv2D(
        input_shape=input_shape,
        kernel_size=4,
        filters=10,
        activation=activation
    ))

    model.add(Conv2D(
        kernel_size=3,
        filters=32,
        activation=activation
    ))

    model.add(Conv2D(
        kernel_size=3,
        filters=108,
        activation=activation
    ))

    model.add(MaxPooling2D(pool_size=3))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation=activation))
    model.add(Dense(1))

    # Compile model with Adadelta optimizer
    model.compile(
        loss='mean_squared_error',
        optimizer='adadelta'
    )

    return model
