import stat
from matplotlib.pyplot import axes, axis
import torch
import numpy as np

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.ones(linear_1_out_features, linear_1_in_features),
            b1 = torch.ones(linear_1_out_features),
            W2 = torch.ones(linear_2_out_features, linear_2_in_features),
            b2 = torch.ones(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        self.activation = {
            'relu': self._relu,
            'sigmoid': self._sigmoid,
            'identity': self._identity
        }
        
        self.grad_activation = {
            'relu': self._grad_relu,
            'sigmoid': self._grad_sigmoid,
            'identity': self._grad_identity
        }

        # put all the cache value you need in self.cache
        self.cache = dict()
    
    @staticmethod
    def _relu(x):
        return torch.maximum(torch.zeros(x.shape), x)
    
    @staticmethod
    def _grad_relu(x):
        return (x > 0.) * 1.0

    @staticmethod
    def _sigmoid(x):
        return 1. / (1. + torch.exp(-x))

    @staticmethod
    def _grad_sigmoid(x):
        return MLP._sigmoid(x) * (1 - MLP._sigmoid(x))

    @staticmethod   
    def _identity(x):
        return x
    
    @staticmethod
    def _grad_identity(x):
        return torch.ones(x.shape)

    def linear1(self, x):
        return (x @ self.parameters['W1'].T) + self.parameters['b1']

    def linear2(self, x):
        return (x @ self.parameters['W2'].T) + self.parameters['b2']

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        z0 = self.linear1(x)
        z1 = self.activation[self.f_function](z0)
        z2 = self.linear2(z1)
        y_hat = self.activation[self.g_function](z2)
        self.cache['z2'] = z2
        self.cache['z1'] = z1
        self.cache['z0'] = z0
        self.cache['x'] = x
        return y_hat
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        dy_hatdz2 = self.grad_activation[self.g_function](self.cache['z2'])
        dz2dW2 = self.cache['z1']
        dJdz2 = dJdy_hat * dy_hatdz2
        dJdW2 = dJdz2.T @ dz2dW2
        dJdb2 = torch.sum(dJdz2, keepdim=True, axis=0)
        dz2dz1 = self.parameters['W2']
        dz1dz0 = self.grad_activation[self.f_function](self.cache['z0'])
        dz0dW1 = self.cache['x']
        dJdW1 = ((dJdz2 @ dz2dz1) * dz1dz0).T @ dz0dW1
        dJdb1 = torch.sum(((dJdz2 @ dz2dz1) * dz1dz0), axis=0, keepdim=True)
        self.grads['dJdW2'] = dJdW2
        self.grads['dJdb2'] = dJdb2
        self.grads['dJdW1'] = dJdW1
        self.grads['dJdb1'] = dJdb1
        #self.parameters['W1'] = self.parameters['W1'] - dJdW1    
        #self.parameters['W2'] = self.parameters['W2'] - dJdW2    
        #self.parameters['b1'] = self.parameters['b1'] - torch.squeeze(dJdb1)
        #self.parameters['b2'] = self.parameters['b2'] -  torch.squeeze(dJdb2)

    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    mse = torch.mean(torch.square(torch.subtract(y, y_hat)))
    n =  torch.numel(y)
    dJdy_hat = ((y_hat - y) * 2.) / n
    return mse, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    eps = 1.e-12
    J = torch.nn.functional.binary_cross_entropy(y_hat, y)
    n = torch.numel(y)
    dJdy_hat = -1./n * (y/(y_hat+eps) - (1.-y)/(1.-y_hat+eps))

    return J, dJdy_hat












