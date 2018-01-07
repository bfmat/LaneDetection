from __future__ import print_function

import json
import os
import sys

from keras.models import load_model

from ..apply.get_simulation_screenshot import get_simulation_screenshot
from ..infer.inference_wrapper_single_line import InferenceWrapperSingleLine
from ..model.reinforcement_learning_agent import ReinforcementSteeringAgent

# A client for the reinforcement learning agent that communicates with the simulation, controlling it using the agent
# which trains and runs inference simultaneously
# Created by brendon-ai, January 2018

# The number of games the agent should play to train
EPISODES = 1000
# The number of past time steps that should be trained on every episode
BATCH_SIZE = 32

# The number of values in the state array passed to the neural network
# The two values that compose the road center line, followed by the current steering angle
STATE_SIZE = 3
# The number of values in the action array passed from the neural network to the simulation
# Steering in the negative direction (left), followed by steering in the positive direction (right)
ACTION_SIZE = 2

# The path to the file that will contain the state of the car and other data
INFORMATION_PATH = '/tmp/information.json'
# The path to the file that will contain the action chosen by the agent
ACTION_PATH = '/tmp/action.json'

# Only run if this script is being executed directly and not imported
if __name__ == "__main__":

    # Verify that the number of command line arguments is correct
    if len(sys.argv) != 2:
        print('Usage:', sys.argv[0], '<right line trained model>')

    # Load the sliding window model using the path provided as a command line argument
    sliding_window_model_path = os.path.expanduser(sys.argv[1])
    sliding_window_model = load_model(sliding_window_model_path)
    # Create an inference wrapper using the model
    inference_wrapper = InferenceWrapperSingleLine(sliding_window_model)

    # Create the reinforcement learning agent
    agent = ReinforcementSteeringAgent(STATE_SIZE, ACTION_SIZE)

    # For each of the training episodes
    for episode in range(EPISODES):
        # Initialize the state variable using zeroes
        state = [0] * STATE_SIZE
        # The number of time iterations that pass during each episode must be tracked
        time_passed = 0

        # Iterate until the game ends and the loop is broken out of
        while True:
            # Increment the time variable
            time_passed += 1

            # Delete the information file if this is not the first iteration
            if time_passed > 1:
                os.remove(INFORMATION_PATH)
            # Calculate an action based on the previous state
            action = agent.act(state)
            # Serialize the action value to the action file
            with open(ACTION_PATH, 'w') as action_file:
                json.dump(action, action_file)

            # Do nothing until the information file is written to again
            while not os.path.isfile(INFORMATION_PATH):
                pass
            # Load information about the car from the information path
            with open(INFORMATION_PATH) as information_file:
                steering_angle, reward, done = json.load(information_file)
            # If the episode has ended
            if done:
                # Set the reward to a negative value
                reward = 10

            # Get the greatest-numbered image in the temp folder
            image = get_simulation_screenshot()
            # Run a prediction on this image using the inference wrapper and get the center line of best fit
            center_line_of_best_fit = inference_wrapper.infer(image)
            # Get the next state by appending the present steering angle to the line of best fit array
            next_state = center_line_of_best_fit.append(steering_angle)

            # Now that the next state has been calculated, the agent should remember the current experience
            agent.remember(state, action, reward, next_state, done)

            # If the current training episode has ended
            if done:
                # Print an message describing the results of the episode
                print("episode: {}/{}, score: {}, epsilon: {}".format(episode, EPISODES, time, agent.epsilon))
                # Move on to the next episode
                break

        # If there is sufficient data in the memory to extract a full batch for training
        if len(agent.memory) > BATCH_SIZE:
            # Run a training iteration
            agent.replay(BATCH_SIZE)
