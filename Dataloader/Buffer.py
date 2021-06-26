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
import threading 
 
## My implement of Buffer using RingStask
class Buffer:
    def __init__(self, size):
        self.size = size + 1
        self.buffer = [0 for _ in range(self.size)]
        self.start_idx = 0
        self.end_idx = 0

        self.lock = threading.Lock()
        self.has_data = threading.Condition(self.lock) # small sock depand on big sock
        self.has_pos = threading.Condition(self.lock)

    def get_size(self):
        return self.size

    def get_used_buffer_size(self):
        return len(self.buffer)

    def get_buffer_num(self):

        if self.start_idx > self.end_idx:
            return self.end_idx + self.size - self.start_idx
        return self.end_idx - self.start_idx

    def get(self):
        ## only get one piece of data
        with self.has_data:
            while self.get_buffer_num() == 0:
                self.has_data.wait()
            result = self.buffer[self.start_idx]
            self.start_idx = (self.start_idx + 1) % self.size
            self.has_pos.notify_all()
        return result

    def get_by_number(self, number:int = 1):
        ## get @number pieces of data
        with self.has_data:
            while self.get_buffer_num() < number:
                self.has_data.wait()

            ## quicker because of the shifting ops only performing after start_idx on buffer list
            ## but the reading order is not the same as the standard implementation
            if self.start_idx + number >= self.size:
                result = self.buffer[self.start_idx:] + self.buffer[:number+self.start_idx-self.size]
            else:
                result = self.buffer[self.start_idx:self.start_idx+number]
            self.start_idx = (self.start_idx + number) % self.size

            # # slower maybe because of the shifting ops on the whole buffer list
            # result = self.buffer[:number]
            # self.buffer = self.buffer[number:]

            self.has_pos.notify_all()
        return result

    def put(self, data):
        with self.has_pos:
            while self.get_buffer_num() >= self.size - 1:
                self.has_pos.wait()
            # If the length of data bigger than buffer's will wait
            self.buffer[self.end_idx] = data
            self.end_idx = (self.end_idx + 1) % self.size
            # some thread is wait data ,so data need release
            self.has_data.notify_all()
    
    def put_batch(self, data):
        with self.has_pos:
            while self.get_buffer_num() >= self.size - len(data) - 1:
                self.has_pos.wait()

            # If the length of data bigger than buffer's will wait
            for i in range(len(data)):
                self.buffer[(self.end_idx+i) % self.size] = data[i]
            self.end_idx = (self.end_idx + len(data)) % self.size
            
            # some thread is wait data ,so data need release
            self.has_data.notify_all()