from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D


def sliding_window_model(window_size):

    # Rectified linear activation function
    activation = 'tanh'

    # Initialize the sequential model
    model = Sequential()

    # One convolutional layer
    model.add(Conv2D(
        input_shape=(window_size, window_size, 3),
        kernel_size=2,
        filters=16,
        strides=2,
        activation=activation
    ))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(64, activation=activation))

    # Sigmoid activation is used because its outputs are in the range of 0 to 1
    model.add(Dense(1, activation='sigmoid'))

    # Compile model with Adadelta optimizer
    model.compile(
        loss='binary_crossentropy',
        optimizer='adadelta'
    )

    return model
