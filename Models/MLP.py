
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
from Ops.TensorOps import zeros, reduce_2d_mean
from Ops.MalMul import *
from Ops.Add import *

from JIT import trans_to_jit
from Activations.activations import Sigmoid


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

        self.forward_out = []

        self.w_grad = zeros([self.h_dim, self.w_dim])
        self.new_d_loss = zeros([self.w_dim])

    def init_params(self):
        if self.initializer is not None:
            self.weight = self.initializer.initial([self.h_dim, self.w_dim])
            # self.bias = self.initializer.initial([self.h_dim])
        else:
            self.weight = zeros([self.h_dim, self.w_dim, ])
        
        ## bias is suggested as initialized with zeros in my case
        if self.use_bias:
            self.bias = zeros([self.h_dim])
        else:
            self.bias = None
    
    def print_params(self):
        print('weight =')
        for w in self.weight:
            print(w)
        if self.use_bias:
            print('bias =', '\n', self.bias)

    @trans_to_jit
    def __call__(self, x):
        return self.forward(x)

    @trans_to_jit
    def update_weight(self, grad_w, lr):
        for i in range(self.h_dim):
            for j in range(self.w_dim):
                self.weight[i][j] -= lr * grad_w[i][j]

    @trans_to_jit
    def update_bias(self, grad_b, lr):
        for j in range(self.h_dim):
            self.bias[j] -= lr * grad_b[j]
    
    def forward(self, x):
        if isinstance(x[0], int): # no batch size
            x = [x] # batch_size = 1

        assert len(x[0]) == self.w_dim, (len(x[0]), self.w_dim)
        
        self.forward_input = x

        ## inline
        results = zeros([len(x), len(self.weight)])
        for result, xx in zip(results, x):
            malmut_21d(self.weight, xx, result)
            add_bias(result, self.bias)
        return results

        ## non-inline
        # x = [add_bias(malmut_21d(self.weight, xx), self.bias) for xx in x]
        # return x

    def backward_non_inline(self, d_loss:List[float], lr:float):
        '''
        @d_loss: shape = [w_grad]
        '''
        if isinstance(d_loss, float): d_loss = [d_loss]
        assert isinstance(d_loss[0], float)
        assert len(d_loss) == self.h_dim

        ## todo: not consider the activation function yet
        # the simpliest is just average forward_input
        batch_size = len(self.forward_input)
        for f in self.forward_input[1:]:
            for i in range(self.w_dim):
                self.forward_input[0][i] += f[i]
        for i in range(self.w_dim):
            self.forward_input[0][i] /= float(batch_size)
        self.forward_input = self.forward_input[0]

        #
        if self.use_bias:
            b_grad = d_loss

        ##
        w_grad = malmut_11d(d_loss, self.forward_input)

        #
        d_loss = malmut_12d(d_loss, self.weight)

        ## update here
        self.update_weight(w_grad, lr)
        if self.use_bias: self.update_bias(b_grad, lr)

        return d_loss

    @trans_to_jit
    def backward(self, d_loss:List[float], lr:float):
        '''
        @d_loss: shape = [w_grad]
        '''
        if isinstance(d_loss, float): d_loss = [d_loss]
        assert isinstance(d_loss[0], float)
        assert len(d_loss) == self.h_dim

        ## todo: not consider the activation function yet
        # the simpliest is just average forward_input
        self.forward_input = reduce_2d_mean(self.forward_input)

        #
        if self.use_bias:
            b_grad = d_loss
            self.update_bias(b_grad, lr)

        ##
        for i in range(self.h_dim):
            for j in range(self.w_dim):
                self.w_grad[i][j] = 0.
        malmut_11d(d_loss, self.forward_input, self.w_grad)

        #
        for i in range(self.w_dim):
            self.new_d_loss[i] = 0.
        malmut_12d(d_loss, self.weight, self.new_d_loss)

        ## update here
        self.update_weight(self.w_grad, lr)

        return self.new_d_loss


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


class MLP_sigmoid(object):
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
        self.act = Sigmoid()

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
        x = self.act(x)
        return x
    
    def backward(self, grad, lr):
        for layer in [self.act]+self.mlp[::-1]:
            grad = layer.backward(grad, lr)
        return grad