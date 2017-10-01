from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D


# A convolutional neural network model intended to predict the lateral position of an object in a wide, short image
# Created by brendon-ai, September 2017


def wide_slice_model(slice_dimensions):

    # Hyperbolic tangent activation function
    activation = 'tanh'

    # Initialize the sequential format
    x = Sequential()

    # Input shape is provided, but add the channel axis to it
    input_shape = slice_dimensions + (3,)

    # Define the input
    input_layer = Input(shape=input_shape)

    # Four convolutional layers
    x = Conv2D(
        input_shape=input_shape,
        kernel_size=4,
        filters=10,
        activation=activation
    )(input_layer)

    x = Conv2D(
        kernel_size=3,
        filters=32,
        activation=activation
    )(x)

    x = Conv2D(
        kernel_size=3,
        filters=108,
        activation=activation
    )(x)

    x = MaxPooling2D(pool_size=3)(x)

    # Fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation=activation)(x)
    output_layer = Dense(1)(x)

    # Define the model
    model = Model(
        inputs=input_layer,
        outputs=output_layer
    )

    # Compile model with Adadelta optimizer
    model.compile(
        loss='mean_squared_error',
        optimizer='adadelta'
    )

    return model
