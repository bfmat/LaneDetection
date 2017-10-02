from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D


# A convolutional neural network model intended to predict the lateral position of an object in a wide, short image
# Created by brendon-ai, September 2017


def wide_slice_model(window_size, num_windows):

    # Hyperbolic tangent activation function
    activation = 'relu'

    # Input size is provided, but add the channel axis to it
    input_shape = (window_size, window_size, 3)

    # List for outputs from each of the window predictions
    window_predictions = []

    # List for corresponding input windows
    input_windows = []

    # Define the layers to be applied to each of the windows
    window_layers = [

        # Two convolutional layers
        Conv2D(
            kernel_size=2,
            filters=16,
            strides=2,
            activation=activation
        ),

        Conv2D(
            kernel_size=2,
            filters=16,
            activation=activation
        ),

        # Fully connected layers
        Flatten(),
        Dense(64, activation=activation),

        # Sigmoid activation is used for the last layer because its outputs are in the range of 0 to 1
        Dense(1, activation='sigmoid')

    ]

    # Step through the width of the image using the height as a step, dividing it into windows
    for i in range(num_windows):

        # Create an input layer for this window and add it to the list
        input_layer = Input(shape=input_shape)
        input_windows.append(input_layer)

        # Apply each of the predefined layers in series to the input
        window_x = input_layer
        for layer in window_layers:
            window_x = layer(window_x)

        # Append the output to the list of window predictions
        window_predictions.append(window_x)

    # Merge the window predictions and send that as an output
    output_layer = Concatenate()(window_predictions)

    # Define the model
    model = Model(
        inputs=input_windows,
        outputs=output_layer
    )

    # Compile model with Adadelta optimizer
    model.compile(
       loss='mean_squared_error',
       optimizer='adadelta'
    )

    return model
