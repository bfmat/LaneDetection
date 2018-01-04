from torch import nn


# A module that takes a tuple input from a previous layer in a PyTorch model and returns the first element of the tuple
# Created by brendon-ai, January 2018

# The main class, implemented as a subclass of Module
class TakeFirst(nn.Module):
    # The main function, provided input from the previous layer
    def forward(self, x):
        # Return the first element of the input
        return x[0]
