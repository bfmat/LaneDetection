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

<<<<<<< HEAD
    # Define the input
    input_layer = Input(shape=input_shape)

    # List for outputs from each of the window predictions
    window_predictions = []

    # Step through the width of the image using the height as a step, dividing it into windows
    window_size = slice_dimensions[0]
    slice_width = slice_dimensions[1]
    for i in range(0, slice_width, window_size):

        # Create a lambda layer slicing the relevant window out of the image
        window = Lambda(
            function=lambda full_slice: full_slice[:, :, i:i + window_size, :],
            output_shape=(window_size, window_size, 3)
        )(input_layer)

        # Convolutional layer
        window_x = Conv2D(
            kernel_size=2,
            filters=16,
            activation=activation
        )(window)

        # Fully connected layers
        window_x = Flatten()(window_x)
        window_x = Dense(128, activation=activation)(window_x)
        window_output_layer = Dense(1)(window_x)

        # Append the output to the list of window predictions
        window_predictions.append(window_output_layer)

    # Merge the window predictions
    x = Concatenate()(window_predictions)

    # Fully connected layers
    x = Dense(128, activation=activation)(x)
    output_layer = Dense(1)(x)

    # Define the model
    model = Model(
        inputs=input_layer,
        outputs=output_layer
    )
=======
    # Four convolutional layers
    model.add(Conv2D(
        input_shape=input_shape,
        kernel_size=4,
        filters=8,
        activation=activation
    ))

    model.add(Conv2D(
        kernel_size=3,
        filters=8,
        activation=activation
    ))

    model.add(Conv2D(
        kernel_size=3,
        filters=8,
        activation=activation
    ))

    model.add(MaxPooling2D(pool_size=3))

    # Flatten and pass straight to the output layer
    model.add(Flatten())
    model.add(Dense(16, activation=activation))
    model.add(Dense(1))
>>>>>>> 6111d3260f798d54a3c4653be8ad151d54834b1e

    # Compile model with Adadelta optimizer
    model.compile(
        loss='mean_squared_error',
        optimizer='adadelta'
    )

    return model
