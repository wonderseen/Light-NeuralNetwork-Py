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
from copy import deepcopy
from Dataloader.Buffer import Buffer
import random as rd
import threading

OUTOFRANGE_SYMBOL = -1

class base_dl:
    def __init__(self,
        data,
        batch_size:int = 1,
        drop_last:bool = True,
        shuffle:bool = False,
        buffer_size:int = 5000,
        epoch_size:int = 1,
    ):
        self.batch_size = batch_size
        self.data = deepcopy(data)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.data_num = len(data)
        self.buffer = Buffer(buffer_size)
        self.epoch_size = epoch_size

        self.total_step = -1 ## how many times to fetch from the whole buffer
        self.init_buffer_thread()
        
    def init_buffer(self):
        print('**** testing: start a new dataloader initialization...')
        data = []
        for _ in range(self.epoch_size):
            if self.shuffle:
                rd.shuffle(self.data)
            data.extend(deepcopy(self.data))

        if self.drop_last and len(data) % self.batch_size:
            data = data[:-(len(data) % self.batch_size)]
        self.total_step = len(data) // self.batch_size
        assert len(data) > self.batch_size, 'number of data %d should be larger than batch_size %d' % (len(data), self.batch_size)


        ## get single piece of data
        # for d in data:
        #     self.buffer.put(d)

        ## get batch-wise data, more efficient
        for i in range(self.total_step):
            self.buffer.put_batch(data[i*self.batch_size:(i+1)*self.batch_size])
        print('**** testing: end a new dataloader initialization.')

    def init_buffer_thread(self):
        self.th1 = threading.Thread(target=self.init_buffer)
        self.th1.start()
        # self.th2 = threading.Thread(target=get)
        # self.th2.start()

    @property
    def get_batch_size(self):
        return self.batch_size


    def get_next_generator(self):
        while self.total_step:
            batch_data = self.buffer.get_by_number(self.batch_size)
            self.total_step -= 1
            yield batch_data


    def get_next(self):
        ## when all data are popped, buffer is restarted again and again.
        if self.total_step == 0:
            self.th1.join()
            self.init_buffer_thread()
            while self.total_step <= 0:
                pass
        return self.get_next_generator().__next__()


    def get_next_2(self):
        ## when all data are popped, get_next ends.
        if self.total_step == 0:
            self.th1.join()
            # self.th2.join()
            return OUTOFRANGE_SYMBOL
        return self.get_next_generator().__next__()

