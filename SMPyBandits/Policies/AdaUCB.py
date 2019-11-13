import numpy as np
from .AdaptiveScalingPolicy import AdaptiveScalingPolicy

np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!
__author__ = "Petteri Pulkkinen"
__version__ = "0.9"


class AdaUCB(AdaptiveScalingPolicy):

    def __init__(self, nb_arms, discount):
        super(AdaUCB, self).__init__(nb_arms, discount=discount)

    def computeIndex(self, arm):
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return (self.d_rewards[arm] / self.d_pulls[arm]) + \
                   self.scaling * np.sqrt((2 * np.log(self.d_t) / self.d_pulls[arm]))

    def computeAllIndex(self):
        indexes = (self.d_rewards / self.d_pulls) + self.scaling * np.sqrt((2 * np.log(self.d_t)) / self.d_pulls)
        indexes[self.pulls < 1] = float("+inf")
        self.index[:] = indexes

