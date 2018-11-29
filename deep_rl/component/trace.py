#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..utils import *


class Trace:
    def __init__(self, network):
        self._trace = {}
        for name, param in network.named_parameters():
            self._trace[name] = torch.zeros(param.data.size())


    def reset(self):
        for name in self._trace.keys():
            self._trace[name].zero_()

    def mul(self, const):
        for name in self._trace.keys():
            self._trace[name].mul_(float(const))

    def add_network_gradient(self, network):
        for name, param in network.named_parameters():
            if param.grad is not None:
                self._trace[name].add_(param.grad.data)

    def copy_to_network(self, network):
        for name, param in network.named_parameters():
            if param.grad is not None:
                param.grad.data.copy_(self._trace[name])
