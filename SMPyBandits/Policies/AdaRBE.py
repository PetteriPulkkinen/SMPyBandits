import numpy as np
from .AdaptiveScalingPolicy import AdaptiveScalingPolicy

np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!
__author__ = "Petteri Pulkkinen"
__version__ = "0.9"


class AdaRBE(AdaptiveScalingPolicy):

    def __init__(self, nb_arms, discount):
        super(AdaRBE, self).__init__(nb_arms, discount=discount)
        self.from_last = np.zeros(nb_arms)

    def startGame(self):
        super(AdaRBE, self).startGame()
        self.from_last.fill(0)

    def getReward(self, arm, reward):
        super().getReward(arm, reward)
        self.from_last += 1
        self.from_last[arm] = 0

    def computeIndex(self, arm):
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return (self.d_rewards[arm] / self.d_pulls[arm]) + \
                   self.scaling * np.sqrt(np.log(
                        self.d_t / (self.d_t + self.from_last[arm] * (np.exp(-1) - 1))))

    def computeAllIndex(self):
        indexes = (self.d_rewards / self.d_pulls) + \
                  self.scaling * np.sqrt(np.log(
                    self.d_t / (self.d_t + self.from_last * (np.exp(-1) - 1))))
        indexes[self.pulls < 1] = float("+inf")
        self.index[:] = indexes
