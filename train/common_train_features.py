from __future__ import print_function

import time

from keras.callbacks import ModelCheckpoint


# A script containing features that are common to all training implementations
# Created by brendon-ai, September 2017


# Train a model given all of the necessary parameters and train it with snapshots
def train_and_save(model, trained_model_folder, images, labels, epochs, batch_size, validation_split):
    # Print a summary of the model architecture
    print('\nSummary of model:')
    print(model.summary())

    # Name the model with the current Unix time
    unix_time = int(time.time())

    # Train the model and save snapshots
    model.fit(
        images,
        labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[ModelCheckpoint(
            '{}/batch={}-epoch={{epoch:d}}-val_loss={{val_loss:f}}.h5'
                .format(trained_model_folder, unix_time)
        )]
    )
