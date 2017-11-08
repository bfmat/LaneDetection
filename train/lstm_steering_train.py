import os
import sys

from scipy.misc import imread

from ..infer import InferenceAndSteeringWrapper
from ..model import lstm_steering_model
from ..train.common_train_features import train_and_save

# A script for training an LSTM network used for steering a car based on the constantly updating line of best fit
# Created by brendon-ai, November 2017

# Training hyperparameters
EPOCHS = 100

# Check that the number of command line arguments is correct
if len(sys.argv) != 5:
    print(
        'Usage:', sys.argv[0],
        '<image folder> <left line trained model> <right line trained model> <trained LSTM folder>')
    sys.exit()

# Load the paths to the image folder, sliding window models, and trained model folder provided as command line arguments
image_folder = os.path.expanduser(sys.argv[1])
model_paths = sys.argv[2:]
trained_model_folder = os.path.expanduser(sys.argv[2])

# Create an inference and steering wrapper using the supplied model paths
inference_and_steering_wrapper = InferenceAndSteeringWrapper(model_paths)

# Create a list of steering angles and a list of lines of best fit
steering_angle_list = []
line_of_best_fit_list = []

# Load all of the images from the provided folder
for image_name in os.listdir(image_folder):
    # Load the image from disk, using its fully qualified path
    image_path = image_folder + '/' + image_name
    image = imread(image_path)

    # Run inference on the image and collect the line of best fit and steering angle
    output_values, _, _, line_of_best_fit = inference_and_steering_wrapper.infer(image)
    steering_angle = output_values[0]

    # Add the steering angle to the list
    steering_angle_list.append(steering_angle)

# Create a model and train it
model = lstm_steering_model()
train_and_save(
    model=model,
    trained_model_folder=trained_model_folder,
    x=line_of_best_fit_list,
    y=steering_angle_list,
    epochs=EPOCHS,
    batch_size=1,
    validation_split=0
)
