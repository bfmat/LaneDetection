from ..model.evolutionary_model import EvolutionaryModel

# A script for driving in the simulation using a neural network trained with evolutionary algorithms for steering
# Created by brendon-ai, January 2018

# Create an initial evolutionary model with the default weights
evolutionary_model = EvolutionaryModel()
evolutionary_model = evolutionary_model.with_noise()
print(evolutionary_model([1, 2]))
