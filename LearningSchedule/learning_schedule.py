
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

from abc import abstractmethod
from math import cos
from JIT import trans_to_jit

pi = 3.1415

class BasicLearningSchedule(object):
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self):
        raise NotImplementedError()
        

class cosineLearningSchedule(BasicLearningSchedule):
    def __init__(
        self,
        initial_learning_rate=0.1,
        decay_steps:float=10000,
        alpha:float=0.9,
    ):
        # super(cosineLearningSchedule, self).__init__() 
        self.decay_steps = decay_steps
        self.alpha = min(alpha, 1.0)
        self.initial_learning_rate = initial_learning_rate
        self.global_step = 0

    @trans_to_jit
    def __call__(self):
        step = min(self.global_step, self.decay_steps)
        cosine_decay = 0.5 * (1 + cos(pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return min(self.initial_learning_rate * self.global_step, self.initial_learning_rate * decayed)

    @trans_to_jit
    def step(self):
        self.global_step += 1
