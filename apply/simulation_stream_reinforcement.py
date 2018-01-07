import gym
import numpy as np

from ..model.reinforcement_learning_agent import ReinforcementSteeringAgent

# A client for the reinforcement learning agent that communicates with the simulation, controlling it using the agent
# which trains and runs inference simultaneously
# Created by brendon-ai, January 2018


# The number of games the agent should play to train
EPISODES = 1000
# The number of past time steps that should be trained on every episode
BATCH_SIZE = 32

# Only run if this script is being executed directly and not imported
if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create the reinforcement learning agent
    agent = ReinforcementSteeringAgent(state_size, action_size)
    # For each of the training episodes
    for episode in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, epsilon: {}".format(episode, EPISODES, time, agent.epsilon))
                break
        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)
