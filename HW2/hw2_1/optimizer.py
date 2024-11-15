# You are not allowed to import any other libraries or modules.

import torch

class SGD:
    def __init__(self, params, learning_rate):
        """
        Initialize the Stochastic Gradient Descent (SGD) optimizer.
        
        Args:
            params: List of model parameters to be updated.
            learning_rate: The learning rate for updating the parameters.
        """
        self.params = list(params)  # Model parameters to be updated
        self.learning_rate = learning_rate

    def step(self):
        """
        Perform a parameter update using Stochastic Gradient Descent (SGD).
        """
        for param in self.params:
            if param.grad is not None:
                # Update the parameter using the gradient and learning rate
                param.data -= self.learning_rate * param.grad.data
                # Detach the gradient to prevent it from being tracked in the computation graph
                param.grad.data.zero_()

    def zero_grad(self):
        """
        Zero the gradients for all parameters.
        """
        for param in self.params:
            if param.grad is not None:
                # Zero out the gradient for each parameter
                param.grad.data.zero_()
