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
from Ops.TensorOps import zeros
from typing import List
from JIT import trans_to_jit

@trans_to_jit
def malmut_22d(x, y, result:List[List[float]]=None):
    assert len(x[0]) == len(y)
    
    h_x = len(x)
    w_x = len(x[0])

    h_y = w_x
    w_y = len(y[0])

    if result is None: result = zeros([h_x, w_y])
    for i in range(h_x):
        for j in range(w_y):
            for m in range(w_x):
                result[i][j] += x[i][m] * y[m][j]
    return result

@trans_to_jit
def malmut_21d(x, y, result:List[List[float]]=None):
    assert len(x[0]) == len(y)
    
    h_x = len(x)
    w_x = len(x[0])

    h_y = w_x

    if result is None: result = zeros([h_x])
    
    for i in range(h_x):
        for m in range(w_x):
            result[i] += x[i][m] * y[m]
    return result

@trans_to_jit
def malmut_12d(x, y, result:List[List[float]]=None):
    assert len(y) == len(x), (len(y), len(x))
    
    h_y = len(y)
    w_y = len(y[0])

    if result is None: result = zeros([w_y])
    for i in range(w_y):
        for m in range(h_y):
            result[i] += x[m] * y[m][i]
    return result

@trans_to_jit
def malmut_11d(x, y, result:List[List[float]]=None):
    '''
    vector_x = (shape_x, 1)
    vector_y = (shape_y, 1)
    vector_x * vector_y.t = matrix
    '''
    assert isinstance(x[0], float)
    assert isinstance(y[0], float)
    
    h_x = len(x)
    h_y = len(y)

    if result is None: result = zeros([h_x, h_y])
    for i in range(h_x):
        for j in range(h_y):
            result[i][j] = x[i] * y[j]
    return result
