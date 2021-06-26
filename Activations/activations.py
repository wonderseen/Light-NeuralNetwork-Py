##############################################################################################
##############################################################################################
##                                                                                          ##
##  ##         ##   #####   ###     # #####   ###### ######   #####  ##### ##### ###     #  ##
##  ##    #    ##  #######  # ##    # #   ##  #      ##   ## ##      #     #     # ##    #  ##
##  ##   ###   ## ###   ### #  ##   # #    ## #      ##   ## ##      #     #     #  ##   #  ##
##  ##  ## ##  ## ##     ## #   ##  # #    ## ###### ######   #####  ##### ##### #   ##  #  ##
##  ## ##   ## ## ###   ### #    ## # #    ## #      ## ##        ## #     #     #    ## #  ##
##   ####    ####  #######  #     ### #   ##  #      ##  ##       ## #     #     #     ###  ##
##    ##      ##    #####   #      ## #####   ###### ##   ##  #####  ##### ##### #      ##  ##
##                                                                                          ##
##############################################################################################
##############################################################################################
from math import exp
from copy import deepcopy
from typing import List
import numpy as np

class wonder_nn:
    def __init__(self):
        pass
    
    def get_shape(self, x):
        shape = []
        while isinstance(x, List):
            shape.append(len(x))
            x = x[0]
        return shape


class Sigmoid(wonder_nn):
    def __init__(self):
        self.output = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.iterative_sigmoid(x)
        self.output = deepcopy(x)
        return x
    
    def iterative_sigmoid(self, x):
        shape = self.get_shape(x)
        if len(shape) == 0:
            x = 1. / (1. + exp(-x))
            input(x)
        elif len(shape) == 1:
            for i in range(shape[0]):
                x[i] = 1. / (1. + exp(-x[i]))
        elif len(shape) == 2:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    x[i][j] = 1. / (1. + exp(-x[i][j]))
        else:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    x[i][j] = self.forward(x[i][j])
        return x

    def backward(self, d_loss, *params):
        d_loss = deepcopy(d_loss)
        self.grad = self.iterative_backward(self.output)
        self.grad = np.array(self.grad).mean(axis=0)
        d_loss = (self.grad * np.array(d_loss)).tolist()
        return d_loss

    def iterative_backward(self, x):
        shape = self.get_shape(x)
        if len(shape) == 0:
            x = x * (1 - x)
        elif len(shape) == 1:
            for i in range(shape[0]):
                x[i] = x[i] * (1 - x[i])
        elif len(shape) == 2:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    x[i][j] = x[i][j] * (1 - x[i][j])
        else:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    x[i][j] = self.forward(x[i][j])
        return x