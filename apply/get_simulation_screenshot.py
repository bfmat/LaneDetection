import os

from scipy.misc import imread

# A function for reading the screenshots saved by the simulation into the temp folder, finding the one with the greatest
# number in its name, loading the image, cropping it, and returning it
# Created by brendon-ai, January 2018


# Path to look for images in and record classifications in, ending with a slash
TEMP_PATH = '/tmp/'


# The main function that will return an image as a NumPy array
def get_simulation_screenshot(remove_images):
    # Parse the names of each of the images in the temp folder and convert them to numbers
    image_names = os.listdir(TEMP_PATH)
    image_numbers = [int(name[3:-4]) for name in image_names if 'sim' in name and '.png' in name]
    # If there are no numbered images, return None
    if not image_numbers:
        return None

    # Get the maximum number and format it into a file name
    max_number = max(image_numbers)
    max_numbered_path = get_image_with_number(max_number)

    # Create a placeholder for the image
    image = None

    # If the file exists
    if os.path.isfile(max_numbered_path):
        # Read the file and crop it into a format that the neural network should accept
        image = imread(max_numbered_path)[90:]

    # Delete all of the images found if the argument is true
    if remove_images:
        for image_number in image_numbers:
            os.remove(get_image_with_number(image_number))

    # Return the placeholder, which may contain an image or None
    return image


# Get the image in the temp folder corresponding to the provided number
def get_image_with_number(number):
    # Simply format the string with the provided number
    return '{}sim{}.png'.format(TEMP_PATH, number)
