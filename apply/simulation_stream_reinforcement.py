from __future__ import print_function

import glob
import json
import os
import sys

import numpy as np

from ..apply.get_simulation_screenshot import get_simulation_screenshot, TEMP_PATH
from ..infer.inference_wrapper_single_line import InferenceWrapperSingleLine
from ..model.reinforcement_learning_agent import ReinforcementSteeringAgent

# A client for the reinforcement learning agent that communicates with the simulation, controlling it using the agent
# which trains and runs inference simultaneously
# Created by brendon-ai, January 2018

# The number of games the agent should play to train
EPISODES = 1000
# The number of past time steps that should be trained on every episode
BATCH_SIZE = 200

# The number of values in the state array passed to the neural network
# These are the two values that compose the road center line
STATE_SIZE = 2
# The number of values in the action array passed from the neural network to the simulation
# The actions are: remaining still, followed by steering in the negative direction (left),
# followed by steering in the positive direction (right)
ACTION_SIZE = 3

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

    # Load the sliding window model using the path provided as a command line argument
    sliding_window_model_path = os.path.expanduser(sys.argv[1])
    # Create an inference wrapper using the model path
    inference_wrapper = InferenceWrapperSingleLine(sliding_window_model_path)

    # Create the reinforcement learning agent
    agent = ReinforcementSteeringAgent(STATE_SIZE, ACTION_SIZE)

    # For each of the training episodes
    for episode in range(EPISODES):
        # Initialize the state variable using zeroes
        state = np.array([[0] * STATE_SIZE])
        # The number of time iterations that pass during each episode must be tracked
        time_passed = 0

        # Iterate until the game ends and the loop is broken out of
        while True:
            # Increment the time variable
            time_passed += 1

            # Calculate an action based on the previous state
            action = agent.act(state)
            # Serialize the action value to the action file
            with open(ACTION_PATH, 'w') as action_file:
                action_file.write(str(action))

            # Initialize the values that will be calculated in the following loop
            reward = None
            done = None

            # Do nothing until the information file can be read from
            information_loaded = False
            while not information_loaded:
                # Try to open the information file
                try:
                    with open(INFORMATION_PATH) as information_file:
                        # Try to load the file as JSON
                        try:
                            _, reward, done = json.load(information_file)
                        # If the file is not valid JSON (it has been incompletely or improperly written)
                        except ValueError:
                            # Continue with the next iteration of the waiting loop
                            continue
                # If an error occurs because the file does not exist
                except IOError:
                    # Continue with the next iteration of the waiting loop
                    continue
                # If we get down to this point, the data has been successfully read
                information_loaded = True

            # If the episode has ended
            if done:
                # Set the reward to a negative value
                reward = -10

            # Loop until a valid image is found
            image = None
            while image is None:
                # Get the greatest-numbered image in the temp folder
                image = get_simulation_screenshot()

            # Run a prediction on this image using the inference wrapper and get the center line of best fit as a list
            # This will serve as the next state
            next_state = inference_wrapper.infer(image)[3]
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

        # If there is sufficient data in the memory to extract a full batch for training
        if len(agent.memory) > BATCH_SIZE:
            # Run a training iteration
            agent.replay(BATCH_SIZE)
