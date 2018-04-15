from __future__ import print_function

import glob
import json
import os
import sys
import time

import numpy as np

from ..apply.get_simulation_screenshot import get_simulation_screenshot, TEMP_PATH
from ..infer.inference_wrapper_single_line import InferenceWrapperSingleLine
from ..model.deep_q_network_agent import DeepQNetworkAgent

# A client for the deep Q-network agent that communicates with the simulation, controlling it using the agent
# which trains and runs inference simultaneously
# Created by brendon-ai, January 2018

# The number of games the agent should play to train
EPISODES = 10000
# The number of past time steps that should be trained on every episode
BATCH_SIZE = 200

# The number of values in the state array passed to the neural network
# These are the two values that compose the road center line, and one containing the current steering angle
STATE_SIZE = 3
# The number of values in the action array passed from the neural network to the simulation
# The actions are: remaining still, followed by steering in the negative direction (left),
# followed by steering in the positive direction (right)
ACTION_SIZE = 3

# Seconds in the past from which samples will be collected that are used to compute the rolling squared error
SQUARED_ERROR_TIME = 300

# The path to the file that will contain the state of the car and other data
INFORMATION_PATH = TEMP_PATH + 'information.json'
# The path to the file that will contain the action chosen by the agent
ACTION_PATH = TEMP_PATH + 'action.txt'

# Only run if this script is being executed directly and not imported
if __name__ == "__main__":

    # Verify that the number of command line arguments is correct
    if len(sys.argv) != 2:
        print('Usage:', sys.argv[0], '<right line trained model>')
        sys.exit()

    # Delete all old images and data files from the temp folder
    for file_path in glob.iglob(TEMP_PATH + '*sim*'):
        os.remove(file_path)

    # Delete the information and action files if they currently exist
    for file_path in [INFORMATION_PATH, ACTION_PATH]:
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Get the simulation started with an arbitrary action
    with open(ACTION_PATH, 'w') as action_file:
        action_file.write('0')

    # Load the sliding window model using the path provided as a command line argument
    sliding_window_model_path = os.path.expanduser(sys.argv[1])
    # Create an inference wrapper using the model path
    inference_wrapper = InferenceWrapperSingleLine(sliding_window_model_path)

    # Create the deep Q-network agent
    agent = DeepQNetworkAgent(STATE_SIZE, ACTION_SIZE)
    # Create a list to add past squared errors from the center of the road and loss function values to,
    # alongside the Unix times at which they were recorded
    diagnostic_data = []
    # For each of the training episodes
    for episode in range(EPISODES):
        # Initialize the state variable using zeroes
        state = np.array([[0] * STATE_SIZE])
        # The number of time iterations that pass during each episode must be tracked
        time_passed = 0

        # Iterate over the training loop for loss values; it should never exit
        for loss in agent.train():

            # Initialize the values that will be calculated in the following loop
            reward = None
            done = None
            # Try to open the information file
            try:
                with open(INFORMATION_PATH) as information_file:
                    # Try to load the file as JSON
                    try:
                        steering_angle, reward, done = json.load(information_file)
                    # If the file is not valid JSON (it has been incompletely or improperly written)
                    except ValueError:
                        # Continue with the next iteration of the waiting loop
                        continue
            # If an error occurs because the file does not exist
            except IOError:
                # Continue with the next iteration of the training loop
                continue

            # Increment the time variable
            time_passed += 1

            # Reduce the epsilon before choosing an action
            agent.decay()
            # Calculate an action based on the previous state
            action = agent.act(state)
            # Serialize the action value to the action file
            with open(ACTION_PATH, 'w') as action_file:
                action_file.write(str(action))

            # If the episode has ended
            if done:
                # Set the reward to a negative value
                reward = -10

            # Calculate the squared error from the center of the road based on the reward, which is one more than the
            # negative of the squared error; add it to the list alongside the loss and current Unix time
            squared_error = -(reward - 1)
            current_time = time.time()
            diagnostic_data.append((squared_error, loss, current_time))

            # Output the error and loss once every thousand iterations (or when the car crashes)
            if time_passed % 1000 == 0 or done:
                # Create lists of the squared errors and losses within a specified span of time in the past
                squared_errors, losses = \
                    zip(*[[squared_error, data_loss]
                          for (squared_error, data_loss, time_of_data_point) in diagnostic_data
                          if time_of_data_point - current_time < SQUARED_ERROR_TIME])
                # Ignore the loss values that are None
                losses = filter(lambda l: l is not None, losses)

                # Calculate the average of the recent squared errors and loss values, and output them to the console
                average_recent_squared_error = sum(squared_errors) / len(squared_errors)
                average_recent_loss = sum(losses) / len(losses)
                print('Over last {} seconds, average squared error is {} and average loss is {}'
                      .format(SQUARED_ERROR_TIME, average_recent_squared_error, average_recent_loss))

            # Loop until a valid image is found
            image = None
            while image is None:
                # Get the greatest-numbered image in the temp folder
                image = get_simulation_screenshot(False)

            # Run a prediction on this image using the inference wrapper and get the center line of best fit as a list
            center_line_of_best_fit = inference_wrapper.infer(image)[3]
            # Get the next state by appending the present steering angle to the line of best fit array
            next_state = np.append(center_line_of_best_fit, steering_angle)
            # Add a batch dimension to the beginning of the state
            next_state = np.expand_dims(next_state, 0)
            # Now that the next state has been calculated, the agent should remember the current experience
            agent.remember(state, action, reward, next_state, done)
            # Shift the next state to the current state
            state = next_state

            # Delete the information file
            os.remove(INFORMATION_PATH)

            # If the current training episode has ended
            if done:
                # Print an message describing the results of the episode
                print("episode: {}/{}, score: {}, epsilon: {}".format(episode, EPISODES, time_passed, agent.epsilon))
                # Move on to the next episode
                break
