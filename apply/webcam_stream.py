#!/usr/bin/env python2

from os import system, listdir, path
from socket import socket, AF_INET, SOCK_STREAM
from sys import argv
from threading import Thread
from time import sleep

from evdev import InputDevice, categorize, ecodes
from keras.models import load_model
from libssh2 import Session
from scipy.misc import imread

from ..infer import SteeringEngine, SlidingWindowInferenceEngine
from ..infer.lane_center_calculation import calculate_lane_center_positions

recording_encoder = False
auto_drive = False
last_max_file = -1
last_steering_angle = 0.0


# Parallel thread that accepts joystick input and sets a global flag
def handle_gamepad_input():
    global recording_encoder
    global auto_drive

    # Get the input from the device file (specific to the joystick I am using)
    joystick = InputDevice('/dev/input/by-id/usb-Logitech_Logitech_Dual_Action_E89BB55E-event-joystick')

    # A provided loop that will run forever, iterating on inputs as they come
    for event in joystick.read_loop():

        # Is the input a button or key press?
        if event.type == ecodes.EV_KEY:
            # Get the identifier of the button that was pressed
            key_event = categorize(event)
            keycode = key_event.keycode

            # Set the global flag to true if the A button was pressed, false if B was pressed
            if keycode == "BTN_THUMB":
                recording_encoder = True
            elif keycode == "BTN_THUMB2":
                recording_encoder = False
            elif keycode == "BTN_PINKIE":
                auto_drive = True
            else:
                auto_drive = False


# Take the latest image and run a regression neural network on it
def compute_steering_angle():
    global last_steering_angle

    # A list that contains all correctly formatted image files in the temp folder
    file_list = [f for f in listdir(image_folder) if "sim" in f and ".jpg" in f]
    for file in file_list:
        # Loop over a copy of the list
        for old_file in file_list[:]:
            # Compare the index of the current image and an image from the rest of the list
            current_index = int(file[3:-4])
            old_index = int(old_file[3:-4])
            # If there is an image that has a lower index than the current one, delete it
            if old_index < current_index:
                file_list.remove(old_file)
                system('rm -f %s/%s' % (image_folder, old_file))

    try:
        # We want to use the latest file
        newest_file = "%s/%s" % (image_folder, file_list[0])
    except:
        print('that error')
        return last_steering_angle

    # Load the image
    image_raw = imread(newest_file)

    # Resize the image, rearrange the dimensions and add an extra one for batch stacking
    image_cropped = image_raw[60:-60]

    # List containing yellow line and white line
    prediction_tensors = [inference_engine.infer(image_cropped) for inference_engine in inference_engines]

    # Calculate the center line positions and add them to the list
    center_line_positions, outer_road_lines = calculate_lane_center_positions(
        left_line_prediction_tensor=prediction_tensors[0],
        right_line_prediction_tensor=prediction_tensors[1],
        minimum_prediction_confidence=0.9,
        original_image_shape=image_cropped.shape,
        window_size=inference_engines[0].window_size
    )

    # Calculate a steering angle from the lines with the steering engine
    try:
        print center_line_positions
        steering_angle, error, slope = steering_engine.compute_steering_angle(center_line_positions)
    except:
        print 'fail'
        return last_steering_angle

    # Move the newest file to the archive directory
    system("mv %s %s/%s.error%s.slope%s.angle%s.jpg" % (
        newest_file, archive_folder, file_list[0], error, slope, steering_angle))

    # Set the last steering angle
    last_steering_angle = steering_angle

    return steering_angle


# Save images in folder provided as a command line argument
image_folder = "/tmp/"
archive_folder = argv[1]

# List of two inference engines, one for each line
inference_engines = []

# For each of the two models passed as command line arguments
for arg in argv[2:]:
    # Format the fully qualified path of the trained model
    model_path = path.expanduser(arg)

    # Load the model from disk
    model = load_model(model_path)

    # Create an inference engine
    inference_engine = SlidingWindowInferenceEngine(
        model=model,
        slice_size=16,
        stride=(4, 4)
    )

    # Add the inference engine to the list
    inference_engines.append(inference_engine)

# Create a steering engine
steering_engine = SteeringEngine(
    proportional_multiplier=0.0025,
    derivative_multiplier=0,
    max_distance_from_line=10,
    ideal_center_x=190,
    center_y=0,
    steering_limit=0.2
)

# Clear the image folder
system('rm -f %s/sim*.jpg' % image_folder)

# Configure the webcam
system('v4l2-ctl -d /dev/video1 --set-ctrl=exposure_auto=3')
system('v4l2-ctl -d /dev/video1 --set-ctrl=exposure_auto_priority=1')
system('v4l2-ctl -d /dev/video1 --set-ctrl=exposure_absolute=250')

# Start the camera capture daemon process from the command line
system(
    'gst-launch-1.0 -v v4l2src device=/dev/video1 ! image/jpeg, width=320, height=180, framerate=30/1 ! jpegparse ! multifilesink location="%s/sim%%d.jpg" &' % image_folder)

# Create the data transfer temp file
system('touch /tmp/drive.path')

# Open an SSH session to the robot controller
sock = socket(AF_INET, SOCK_STREAM)
sock.connect(('192.168.0.230', 22))

session = Session()
session.startup(sock)
session.userauth_password('admin', '')

channel = session.open_session()
channel.shell()

# Start the joystick input thread
thread = Thread(target=handle_gamepad_input)
thread.daemon = True
thread.start()

# Wait for gstreamer to start saving images to disk
sleep(1)

# Loop forever, recording steering angle data along with images
while True:
    # If auto drive is enabled, call the function to compute the steering angle
    steering_angle = 0.0
    if auto_drive:
        # Encoder cannot be recorded when auto drive is enabled
        recording_encoder = False
        steering_angle = -compute_steering_angle()

        # Print out the steering angle
        print(steering_angle)

    # Compose values for transfer to robot controller into a single string
    values_to_jetson = (int(recording_encoder), int(auto_drive), steering_angle)
    values_str = ''
    for value in values_to_jetson:
        values_str += (str(value) + '\n')
    values_str = values_str[:-1]

    # Send values over SSH to the robot controller by writing them to a temp file and then renaming it
    channel.write('. /home/lvuser/run_and_return_newline.sh "printf \'%s\' > /home/lvuser/values.txt"\n' % values_str)
    channel.read(1024)

    # To be executed if we are supposed to be recording steering angle data currently
    if recording_encoder:
        # Prompt robot controller to send us current encoder position
        channel.write('cat /home/lvuser/latest.encval\n')
        stdout = channel.read(1024)

        # Loop over SSH output
        for line in stdout.split("\n"):
            # Only one iteration should occur each time because the input should have just one line containing 'out'
            if 'out' in line:
                # Extract the encoder position from the line
                encoder_value = float(line[3:])

                # Loop over all camera capture images currently in the temp directory
                # Tracking of the last file with the highest number ensures no duplicate data is recorded
                max_file = last_max_file
                for file_name in listdir(image_folder):
                    if "sim" in file_name and ".jpg" in file_name:
                        # Extract the Unix timestamp from the file name
                        file_number = int(file_name[3:-4])
                        # Find the newest image file
                        if file_number > max_file:
                            max_file = file_number
                        # Delete everything but the newest image
                        elif file_number < max_file:
                            system('rm -f %s/%s' % (image_folder, file_name))

                # If a new value has been obtained, record it in the file name of the latest image
                if max_file > last_max_file:
                    system('mv %s/sim%d.jpg %s/%f_sim%d.jpg' % (
                        image_folder, max_file, archive_folder, encoder_value, max_file))

                # Set the previous image counter to the current image's timestamp
                last_max_file = max_file
    elif not auto_drive:
        # If we are not recording, constantly clear the image folder
        system('rm -f %s/sim*.jpg' % image_folder)
