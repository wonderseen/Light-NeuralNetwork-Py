
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
from Models.MLP import *
from Ops.TensorOps import random_tensor
from LearningSchedule.learning_schedule import cosineLearningSchedule
import random as rd
from copy import deepcopy
from math import cos, pow
from Dataloader.base_dataloader import base_dl

from time import time

def test_MLP():
    ## generate data from non-linear numeric expression
    data_num = 200000
    feat_dim = 5
    label_dim = 1 # regression
    data = random_tensor([data_num, feat_dim]) 
    # label = [sum([cos(pow(c, i))*c for i, c in enumerate(d)]) / len(data[0]) for d in data]
    label = [sum([c*c for i, c in enumerate(d)]) / len(data[0]) for d in data]
    max_label = max(label)
    label = [l/max_label for l in label]

    ## split dataset into train and test
    data_label = [(a, b) for a, b in zip(data, label)]
    rd.shuffle(data_label)
    test_rate = 1. / 20.
    test_num = int(data_num*test_rate)
    train_data = data_label[:-test_num]
    test_data = data_label[-test_num:]
    data_num = len(train_data)

    ## define our MLP model
    model = MLP_sigmoid(feat_dim, label_dim, hidden_size=20, layer_num=2, initializer=trunc_gaussion_initializer())


    ## define learning schedule
    lr = cosineLearningSchedule(initial_learning_rate=0.1)


    ## train
    iter_num = 30000
    batch_size = 16
    best_loss = 1e9
    data_num = len(train_data)
    sample_type = 1 # 0 or 1
    if sample_type == 1:
        ## Data loader in a writing-reading threading mode that I simulate the way TF/PY does.
        ## TODO: advance on a multi-reading-threading dataloader
        epoch_size = iter_num * batch_size // data_num + 1
        buffer_size = int(batch_size * iter_num * 1.5)
        train_data_loader = base_dl(
                                train_data,
                                buffer_size=buffer_size,
                                batch_size=batch_size,
                                drop_last=True,
                                epoch_size=epoch_size,
                                shuffle=False)
        while train_data_loader.buffer.get_buffer_num() < min(buffer_size, epoch_size * data_num) - batch_size:
            pass
    
    ## traing
    start_time = time()
    for i in range(iter_num):
        if i % 2000 == 0:
            data = deepcopy([a[0] for a in test_data])
            label = deepcopy([a[1] for a in test_data])
            predict = model(data)
            loss = sum([abs(p[0]-l) for p, l in zip(predict, label)]) / len(label)
            best_loss = min(best_loss, loss)
            print('step=%d test_loss=%.04f lr=%.04f label_mean=%.04f' % (i, loss, lr(), sum(label)/len(label)))

        tmp_batch = train_data_loader.get_next() if sample_type == 1 else rd.sample(train_data, batch_size)
        
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

