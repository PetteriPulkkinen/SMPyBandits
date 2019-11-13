import numpy as np
from .EpsilonGreedy import EpsilonGreedy


np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!
__author__ = "Petteri Pulkkinen"
__version__ = "0.9"


class DEGreedy(EpsilonGreedy):

    def __init__(self, nb_arms, epsilon, discount):
        super(DEGreedy, self).__init__(nb_arms, epsilon=epsilon)
        self.discount = discount

    def getReward(self, arm, reward):
        self.pulls = self.discount * self.pulls
        self.rewards = self.discount * self.rewards
        self.pulls[arm] += 1
        self.rewards[arm] += reward
        self.t += 1