
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
from Dataloader.Buffer import Buffer
from Initializers.Initializer import trunc_gaussion_initializer
from Models.MLP import *
from Ops.TensorOps import random_tensor
from LearningSchedule.learning_schedule import cosineLearningSchedule
import random as rd
from copy import deepcopy
from math import cos, pow, ceil
from Dataloader.base_dataloader import base_dl
from time import time


def test_MLP():
    ## generate data from non-linear numeric expression
    data_num = 20000
    feat_dim = 5
    label_dim = 1 # regression
    data = random_tensor([data_num, feat_dim]) 
    label = [sum([cos(pow(c, i))*c for i, c in enumerate(d)]) / len(data[0]) for d in data]


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
    iter_num = 30000
    batch_size = 16
    best_loss = 1e9
    data_num = len(train_data)
    buffer_size = 500000
    train_data_loader = base_dl(
                            train_data,
                            buffer_size=buffer_size,
                            batch_size=batch_size,
                            drop_last=True,
                            epoch_size=iter_num * batch_size // data_num + 1,
                            shuffle=True)
    ## wait for loading
    while train_data_loader.buffer.get_buffer_num() <= (iter_num * batch_size // data_num + 1) * data_num - batch_size: pass
    
    start_time = time()
    for i in range(iter_num):
        if i % 2000 == 0:
            data = deepcopy([a[0] for a in test_data])
            label = deepcopy([a[1] for a in test_data])
            predict = model(data)
            loss = sum([abs(p[0]-l) for p, l in zip(predict, label)]) / len(label)
            best_loss = min(best_loss, loss)
            print('step=%d test_loss=%.04f lr=%.04f label_mean=%.04f' % (i, loss, lr(), sum(label)/len(label)))
        
        # tmp_batch = rd.sample(train_data, batch_size)
        tmp_batch = train_data_loader.get_next()
        
        data = deepcopy([a[0] for a in tmp_batch])
        label = deepcopy([a[1] for a in tmp_batch])

        predict = model(data)

        ## mse
        # loss = sqrt(sum([pow(p[0]-l, 2) for p, l in zip(predict, label)]) / len(label))
        loss = sum([(p[0]-l) for p, l in zip(predict, label)]) / len(label)

        model.backward(loss, lr())
        lr.step()
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
step=0 test_loss=0.4419 lr=0.1000 label_mean=0.3944
step=2000 test_loss=0.0362 lr=0.0990 label_mean=0.3944
step=4000 test_loss=0.0345 lr=0.0965 label_mean=0.3944
step=6000 test_loss=0.0371 lr=0.0935 label_mean=0.3944
step=8000 test_loss=0.0340 lr=0.0910 label_mean=0.3944
step=10000 test_loss=0.0309 lr=0.0900 label_mean=0.3944
step=12000 test_loss=0.0285 lr=0.0900 label_mean=0.3944
step=14000 test_loss=0.0306 lr=0.0900 label_mean=0.3944
step=16000 test_loss=0.0257 lr=0.0900 label_mean=0.3944
step=18000 test_loss=0.0260 lr=0.0900 label_mean=0.3944
step=20000 test_loss=0.0271 lr=0.0900 label_mean=0.3944
step=22000 test_loss=0.0241 lr=0.0900 label_mean=0.3944
step=24000 test_loss=0.0236 lr=0.0900 label_mean=0.3944
step=26000 test_loss=0.0237 lr=0.0900 label_mean=0.3944
step=28000 test_loss=0.0244 lr=0.0900 label_mean=0.3944
step=30000 test_loss=0.0227 lr=0.0900 label_mean=0.3944
step=32000 test_loss=0.0226 lr=0.0900 label_mean=0.3944
step=34000 test_loss=0.0230 lr=0.0900 label_mean=0.3944
step=36000 test_loss=0.0229 lr=0.0900 label_mean=0.3944
step=38000 test_loss=0.0223 lr=0.0900 label_mean=0.3944
step=40000 test_loss=0.0224 lr=0.0900 label_mean=0.3944
step=42000 test_loss=0.0231 lr=0.0900 label_mean=0.3944
step=44000 test_loss=0.0228 lr=0.0900 label_mean=0.3944
step=46000 test_loss=0.0217 lr=0.0900 label_mean=0.3944
step=48000 test_loss=0.0229 lr=0.0900 label_mean=0.3944
time 38.02636909484863
best precision = 0.021731
============= 0-layer's ================
weight =
[0.07985252894447363, -0.1660243494362199, -0.08052024136039661, 0.02434320579016694, -0.1078799626314696]
[-0.0857223198923299, -0.124417479996275, -0.20570957934899148, -0.19792852137310715, -0.08078142094687014]
[0.14062583830613298, 0.021772878634511834, 0.04654327730126229, 0.19907367933257406, 0.010141218295242272]
[-0.027400004737439098, 0.24619484103852937, -0.11738604798654176, 0.10020104457113864, -0.1720760204078004]
[-0.17271357705738824, -0.035353775594852636, -0.06839175538596459, -0.19285937787380766, -0.1836294814646378]
[-0.012722721427873849, -0.15139722657996377, 0.0256311329067377, 0.1902730014377961, -0.14596391121949726]
[-0.13811506445305197, 0.10340458729231236, 0.06885046130771628, 0.2187803039335599, 0.04299714795657724]
[0.01585380794080505, -0.007199583612676973, -0.06516232048926525, 0.08346560582413054, -0.10547805218308617]
[0.09743056079814565, -0.020710613370406486, 0.02389049618159201, -0.06510539067800235, -0.10336640712322936]
[0.1713069272113096, 0.113478745641865, 0.13135052093483274, 0.018499425011200714, 0.26959475423249457]
bias = 
 [0.05170172045160311, 0.142733312232825, -0.08572426541433485, -0.0059750590434059754, 0.13395367535625805, 0.019147824800879978, -0.061461229155472086, 0.016228508314330467, 0.014385732629500438, -0.14451708260815777]
============= 1-layer's ================
weight =
[-0.08164757433147876, -0.2262591721438437, 0.13947430748287884, 0.00981497086829827, -0.22095836994939044, -0.031411761432041274, 0.10124208606114769, -0.02484094568872506, -0.02847499999507747, 0.23372299499535476]
bias = 
 [0.22153937155307263]
'''