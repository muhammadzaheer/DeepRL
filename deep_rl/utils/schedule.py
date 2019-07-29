#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

class ConstantSchedule:
    def __init__(self, val):
        self.val = val

    def __call__(self, steps=1):
        return self.val

class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        if not int(steps):
            self.inc = 0.0
        else:
            self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val


class LinearScheduleAdam:
    def __init__(self, start, end=None, steps=None):

        self.starts = [0.9, 0.99]
        self.ends = [0.99, 1.0]

        self.inc = (self.ends[0] - self.starts[0]) / float(200000)

        self.current = self.starts[0]
        self.end = self.ends[0]
        self.step = 0


    def __call__(self, steps=1):
        val = self.current
        self.current = min(self.current + self.inc * steps, self.end)

        if self.step == 200000:
            self.inc = (self.ends[1] - self.starts[1]) / float(600000)
            self.end = self.ends[1]

        self.step +=1

        return val
