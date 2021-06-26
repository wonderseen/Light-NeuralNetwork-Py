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
from JIT import trans_to_jit

def zeros(shape):
    len_shape = len(shape)
    if len_shape == 2:
        return [[0. for _ in range(shape[1])] for _ in range(shape[0])]
    elif len_shape == 1:
        return [0. for _ in range(shape[0])]
    else:
        raise NotImplementedError('')

def random_tensor(shape):
    len_shape = len(shape)
    if len_shape == 2:
        return [[rd.random() for _ in range(shape[1])] for _ in range(shape[0])]
    elif len_shape == 1:
        return [rd.random() for _ in range(shape[0])]
    else:
        raise NotImplementedError('')


@trans_to_jit
def reduce_2d_mean(x):
    sample_num = len(x[0])
    for xx in x[1:]:
        for i, fea in enumerate(xx):
            x[0][i] += fea
    for i in range(len(x[0])):
        x[0][i] / sample_num
    return x[0]

