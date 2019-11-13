import numpy as np
from .AdaptiveScalingPolicy import AdaptiveScalingPolicy


np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!
__author__ = "Petteri Pulkkinen"
__version__ = "0.9"


class AdaUCBVT(AdaptiveScalingPolicy):

    def __init__(self, nb_arms, discount):
        super(AdaUCBVT, self).__init__(nb_arms=nb_arms, discount=discount)
        self.rewardsSquared = np.zeros(self.nbArms)

    def startGame(self):
        super(AdaUCBVT, self).startGame()
        self.rewardsSquared.fill(0)

    def getReward(self, arm, reward):
        super(AdaUCBVT, self).getReward(arm, reward)
        self.rewardsSquared = self.discount * self.rewardsSquared
        self.rewardsSquared[arm] = self.rewardsSquared[arm] + reward ** 2

    def computeIndex(self, arm):

        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            mean = self.d_rewards[arm] / self.d_pulls[arm]   # Mean estimate
            variance = np.abs((self.rewardsSquared[arm] / self.d_pulls[arm]) - mean ** 2)  # Variance estimate
            # Correct variance estimate
            variance += np.sqrt(2.0 * np.log(self.d_t) / self.d_pulls[arm])
            return mean + np.sqrt(np.log(self.d_t) * variance / self.d_pulls[arm])

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        means = self.d_rewards / self.d_pulls   # Mean estimate
        variances = np.abs((self.rewardsSquared / self.d_pulls) - means ** 2)  # Variance estimate
        variances += np.sqrt(2.0 * np.log(self.d_t) / self.d_pulls)
        indexes = means + np.sqrt(np.log(self.d_t) * variances / self.d_pulls)
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes
