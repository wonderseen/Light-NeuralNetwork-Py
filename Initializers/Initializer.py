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
import random as rd
from typing import List
from math import sqrt, exp
from abc import abstractmethod


pi = 3.141592

class basic_initializer:
    def __init__(self):
        pass

    @abstractmethod
    def initial(self, name, shape):
        print(f'{name} for a tensor of shape {shape}.')


class trunc_gaussion_initializer(basic_initializer):
    def __init__(self,
        mean: float=0.0,
        std: float=0.1,
        seed: int=0
    ):
        self.name = self.__class__.__name__
        self.mean = mean
        self.std = std
        self.seed = seed
        rd.seed(self.seed)

    def initial(self, shape: List[int]):
        super(trunc_gaussion_initializer, self).initial(self.name, shape)

        if len(shape) == 2:
            tmp = [ [0 for _ in range(shape[1])] for _ in range(shape[0])]
            for i in range(shape[0]):
                for j in range(shape[1]):
                    tmp[i][j] = rd.gauss(self.mean, self.std)
        elif len(shape) == 1:
            tmp = [ 0 for _ in range(shape[0])]
            for j in range(shape[0]):
                tmp[j] = rd.gauss(self.mean, self.std)
        else:
            raise NotImplementedError('')
        ## 
        # mean_tmp = 
        # cov_tmp = 
        # sqrt_tmp = sqrt(cov_tmp)
        # gaussion_standard = [1. / (sqrt(2*pi) * sqrt_tmp) * exp(-(x-mean_tmp) / 2 / cov_tmp) for x in ]
        # wanted_gaussion = [ [x * sqrt_tmp + mean_tmp for x in xx] for xx in gaussion_standard]

        return tmp