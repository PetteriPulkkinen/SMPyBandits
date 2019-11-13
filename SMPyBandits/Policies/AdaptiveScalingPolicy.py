import numpy as np
from .IndexPolicy import IndexPolicy

__author__ = "Petteri Pulkkinen"
__version__ = "0.9"


class AdaptiveScalingPolicy(IndexPolicy):

    def __init__(self, nb_arms, discount):
        super(AdaptiveScalingPolicy, self).__init__(nb_arms)
        self.scaling = 1
        self.discount = discount
        self.d_t = 0
        self.d_pulls = np.zeros(nb_arms)
        self.d_rewards = np.zeros(nb_arms)

    def startGame(self):
        super(AdaptiveScalingPolicy, self).startGame()
        self.d_pulls.fill(0)
        self.d_rewards.fill(0)
        self.d_t = 0

    def getReward(self, arm, reward):
        super().getReward(arm, reward)

        self.d_pulls = self.discount * self.d_pulls
        self.d_pulls[arm] += 1
        self.d_rewards = self.discount * self.d_rewards
        self.d_rewards[arm] += reward
        self.d_t = self.discount * self.d_t + 1

        nz_idx = np.nonzero(self.d_pulls)
        self.scaling = np.max(self.d_rewards[nz_idx] / self.d_pulls[nz_idx])

    def computeIndex(self, arm):
        raise NotImplementedError("")



