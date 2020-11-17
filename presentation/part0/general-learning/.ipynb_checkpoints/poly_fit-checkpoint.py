import torch as torch
import numpy as np
import torch.nn as nn


class PolynomialFit:
    def __init__(self, degree:int, init_coeff=None):
        self.degree = degree
        if (init_coeff is None):
            self.coeff = torch.randn(self.degree, requires_grad=True)
        else:
            self.coeff = init_coeff

    def value(self, x):
        x_vals = torch.zeros((1,self.degree))
        for i in range(self.degree):
            x_vals[0,i] = x**i
        print(x_vals)
        print(self.coeff)
        #x_t = torch.from_numpy(x_vals)
        v = x_vals*self.coeff
        return torch.sum(v)


class PolynomialLayer(nn.Module):

    def __init__(self, degree, scale=1.0):
        super().__init__()
        self.degree = degree
        weights = torch.randn(1, self.degree)*scale 
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        x_array = np.zeros((x.shape[0], self.degree))
        for i in range(self.degree):
            x_array[:, i] = x[:]**i
        xv = torch.from_numpy(x_array)*self.weights
        #print(self.weights)
        #print(xv)
        return torch.sum(xv, axis=1)

    def print_weights(self):
        print(self.weights)






