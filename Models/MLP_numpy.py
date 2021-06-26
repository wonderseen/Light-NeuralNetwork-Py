
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
from typing import List
from Initializers.Initializer import basic_initializer

from JIT import trans_to_jit
import numpy as np


class Dense2d(object):
    def __init__(
        self,
        w:int, # input_shape
        h:int, # output_shape
        initializer:basic_initializer=None,
        use_bias:bool=True,
    ):
        self.w_dim = w
        self.h_dim = h
        self.use_bias = use_bias

        self.initializer = initializer
        self.init_params()

        self.forward_out = None

    def init_params(self):
        if self.initializer is not None:
            self.weight = np.array(self.initializer.initial([self.h_dim, self.w_dim]))
            # self.bias = self.initializer.initial([self.h_dim])
        else:
            self.weight = np.zeros([self.h_dim, self.w_dim, ])
        
        ## bias is suggested as initialized with zeros in my case
        if self.use_bias:
            self.bias = np.zeros([self.h_dim])
        else:
            self.bias = None
    
    def print_params(self):
        print('weight =')
        for w in self.weight:
            print(w)
        if self.use_bias:
            print('bias =', '\n', self.bias)

    def __call__(self, x):
        return self.forward(x)

    @trans_to_jit
    def forward(self, x):
        if isinstance(x[0], float): # no batch size
            x = [x] # batch_size = 1
        x = np.array(x)
        self.forward_input = x
        return x.dot(self.weight.T) + self.bias

    @trans_to_jit
    def backward(self, d_loss:List[float], lr:float):
        '''
        @d_loss: shape = [w_grad]
        '''
        if isinstance(d_loss, float): d_loss = [d_loss]
        assert isinstance(d_loss[0], float)
        assert len(d_loss) == self.h_dim
        d_loss = np.array(d_loss)

        ## todo: not consider the activation function yet
        # the simpliest is just average forward_input
        self.forward_input = self.forward_input.mean(axis=0)

        #
        if self.use_bias:
            b_grad = d_loss
            self.bias -= lr * b_grad

        ##
        w_grad = d_loss[:,None].dot(self.forward_input[None,])
        self.weight -= lr * w_grad

        #
        new_d_loss = d_loss.dot(self.weight)
        
        return new_d_loss


class MLP(object):
    def __init__(self,
        input_h,
        output_h,
        hidden_size,
        layer_num:int=1,
        initializer:basic_initializer=None
    ):
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.input_h = input_h
        self.output_h = output_h
        self.initializer = initializer

        self.mlp = self.mlp_builder()

    def mlp_builder(self):
        ## define the mlp builder
        mlp = []
        if self.layer_num > 1:
            mlp.append(Dense2d(self.input_h, self.hidden_size, initializer=self.initializer))
            mlp.extend([Dense2d(self.hidden_size, self.hidden_size, initializer=self.initializer) for _ in range(self.layer_num-2)])
            mlp.append(Dense2d(self.hidden_size, self.output_h, initializer=self.initializer))
        else:
            mlp.append(Dense2d(self.input_h, self.output_h, initializer=self.initializer))
        return mlp

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for ds in self.mlp:
            x = ds(x)
        return x
    
    def backward(self, grad, lr):
        for ds in self.mlp[::-1]:
            grad = ds.backward(grad, lr)
        return grad
