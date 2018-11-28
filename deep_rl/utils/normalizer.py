#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
# from baselines.common.running_mean_std import RunningMeanStd

class BaseNormalizer:
    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return

# class MeanStdNormalizer(BaseNormalizer):
#     def __init__(self, read_only=False, clip=10.0, epsilon=1e-8):
#         BaseNormalizer.__init__(self, read_only)
#         self.read_only = read_only
#         self.rms = None
#         self.clip = clip
#         self.epsilon = epsilon
#
#     def __call__(self, x):
#         x = np.asarray(x)
#         if self.rms is None:
#             self.rms = RunningMeanStd(shape=(1, ) + x.shape[1:])
#         if not self.read_only:
#             self.rms.update(x)
#         return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
#                        -self.clip, self.clip)


class MinMaxNormalizer(BaseNormalizer):
    def __init__(self, read_only=False, clip=10.0, epsilon=1e-8):
        BaseNormalizer.__init__(self, read_only)
        self.read_only = read_only
        self.maxes = None
        self.mins = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        x = np.asarray(x)
        if self.maxes is None:
            self.maxes = np.zeros_like(x)
            self.mins = np.zeros_like(x)
        self.maxes = np.maximum(x, self.maxes)
        self.mins = np.minimum(x, self.mins)
        range = self.maxes - self.mins
        return np.true_divide(x, range, out=np.zeros_like(x), where=range != 0, casting='unsafe')


class RescaleNormalizer(BaseNormalizer):
    def __init__(self, coef=1.0):
        BaseNormalizer.__init__(self)
        self.coef = coef

    def __call__(self, x):
        x = np.asarray(x)
        return self.coef * x

class ImageNormalizer(RescaleNormalizer):
    def __init__(self):
        RescaleNormalizer.__init__(self, 1.0 / 255)

class SignNormalizer(BaseNormalizer):
    def __call__(self, x):
        return np.sign(x)