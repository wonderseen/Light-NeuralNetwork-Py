
'''
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
Python implement of naive MLP regression, no other dependency required.
'''
from Initializers.Initializer import trunc_gaussion_initializer
from Models.MLP_numpy import *
from Ops.TensorOps import random_tensor
from LearningSchedule.learning_schedule import cosineLearningSchedule
import random as rd
from copy import deepcopy
from math import cos, pow

from time import time

def test_MLP():
    ## generate data from non-linear numeric expression
    data_num = 200000
    feat_dim = 5
    label_dim = 1 # regression
    data = np.array(random_tensor([data_num, feat_dim]))
    label = np.array([sum([cos(pow(c, i))*c for i, c in enumerate(d)]) / feat_dim for d in data])

    ## split dataset into train and test
    data_label = [(a, b) for a, b in zip(data, label)]
    rd.shuffle(data_label)
    test_rate = 1. / 20.
    test_num = int(data_num*test_rate)
    train_data = data_label[:-test_num]
    test_data = data_label[-test_num:]


    ## define our MLP model
    model = MLP(feat_dim, label_dim, hidden_size=10, layer_num=2, initializer=trunc_gaussion_initializer())


    ## define learning schedule
    lr = cosineLearningSchedule()


    ## train
    @trans_to_jit
    def train():
        iter_num = 50000
        batch_size = 32
        best_loss = 1e9
        for i in range(iter_num):
            if i % 2000 == 0:
                data = np.array([a[0] for a in test_data])
                label = np.array([a[1] for a in test_data])
                predict = model(data)
                loss = np.sum(np.abs(predict - label[:, None])) / label.shape[0]
                best_loss = min(best_loss, loss)
                print('step=%d test_loss=%.04f lr=%.04f label_mean=%.04f' % (i, loss, lr(), sum(label)/len(label)))

            tmp_batch = rd.sample(train_data, batch_size)
            data = np.array([a[0] for a in tmp_batch])
            label = np.array([a[1] for a in tmp_batch])

            predict = model(data)

            ## mse
            # loss = sqrt(sum([pow(p[0]-l, 2) for p, l in zip(predict, label)]) / len(label))
            loss = np.sum(predict - label[:, None]) / label.shape[0]
            model.backward(loss, lr())
            lr.step()

    start_time = time()
    train()
    print(f'time {time()-start_time}' )


    print('best precision = %f' % best_loss)
    for i, layer in enumerate(model.mlp):
        print(f'============= {i}-layer\'s ================')
        layer.print_params()

if __name__ == '__main__':
    test_MLP()

'''
trunc_gaussion_initializer for a tensor of shape [10, 5].
trunc_gaussion_initializer for a tensor of shape [1, 10].
step=0 test_loss=0.4398 lr=0.1000 label_mean=0.3929
step=2000 test_loss=0.0628 lr=0.0990 label_mean=0.3929
step=4000 test_loss=0.0580 lr=0.0965 label_mean=0.3929
step=6000 test_loss=0.0538 lr=0.0935 label_mean=0.3929
step=8000 test_loss=0.0491 lr=0.0910 label_mean=0.3929
step=10000 test_loss=0.0458 lr=0.0900 label_mean=0.3929
step=12000 test_loss=0.0425 lr=0.0900 label_mean=0.3929
step=14000 test_loss=0.0388 lr=0.0900 label_mean=0.3929
step=16000 test_loss=0.0359 lr=0.0900 label_mean=0.3929
step=18000 test_loss=0.0332 lr=0.0900 label_mean=0.3929
step=20000 test_loss=0.0311 lr=0.0900 label_mean=0.3929
step=22000 test_loss=0.0288 lr=0.0900 label_mean=0.3929
step=24000 test_loss=0.0272 lr=0.0900 label_mean=0.3929
step=26000 test_loss=0.0258 lr=0.0900 label_mean=0.3929
step=28000 test_loss=0.0246 lr=0.0900 label_mean=0.3929
step=30000 test_loss=0.0237 lr=0.0900 label_mean=0.3929
step=32000 test_loss=0.0229 lr=0.0900 label_mean=0.3929
step=34000 test_loss=0.0227 lr=0.0900 label_mean=0.3929
step=36000 test_loss=0.0221 lr=0.0900 label_mean=0.3929
step=38000 test_loss=0.0218 lr=0.0900 label_mean=0.3929
step=40000 test_loss=0.0214 lr=0.0900 label_mean=0.3929
step=42000 test_loss=0.0212 lr=0.0900 label_mean=0.3929
step=44000 test_loss=0.0210 lr=0.0900 label_mean=0.3929
step=46000 test_loss=0.0211 lr=0.0900 label_mean=0.3929
step=48000 test_loss=0.0209 lr=0.0900 label_mean=0.3929
time 8.210481643676758
best precision = 0.020894
============= 0-layer's ================
weight =
[ 0.0834953  -0.16337458 -0.0901535   0.03362916 -0.11045916]
[-0.08701969 -0.11220565 -0.20980077 -0.17779642 -0.06450785]
[ 0.14823032  0.01235553  0.04211755  0.19445553 -0.00330278]
[-0.04000425  0.25594209 -0.13778747  0.08776851 -0.18807739]
[-0.17309348 -0.02507557 -0.05000049 -0.16117084 -0.16847536]
[-0.00638287 -0.15180891  0.03036492  0.21531636 -0.14971048]
[-0.14855217  0.1060186   0.06367936  0.21134554  0.03729684]
[ 0.01008784 -0.00131981 -0.08089471  0.07941209 -0.11420149]
[ 0.10127861 -0.02271358  0.02720119 -0.06506367 -0.10893743]
[0.19619066 0.08787332 0.15895527 0.02130802 0.27364967]
bias = 
 [ 0.01930929  0.00816742  0.00248665  0.00811119 -0.01607786  0.00832313
  0.00591696  0.01883192 -0.00292467 -0.04525448]
============= 1-layer's ================
weight =
[-0.10765892 -0.2847183   0.14321854  0.02448184 -0.28945284 -0.07679446
  0.13678503 -0.01691876 -0.0315258   0.20495205]
bias = 
 [0.08298031]
'''