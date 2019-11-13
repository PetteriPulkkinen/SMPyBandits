import numpy as np
from .AdaptiveScalingPolicy import AdaptiveScalingPolicy


np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!
__author__ = "Petteri Pulkkinen"
__version__ = "0.9"


class AdaKLUCB(AdaptiveScalingPolicy):

    def __init__(self, nb_arms, discount, klucb):
        super(AdaKLUCB, self).__init__(nb_arms, discount=discount)
        self.klucb = klucb
        self.klucb_vect = np.vectorize(klucb)
        self.klucb_vect.__name__ = klucb.__name__
        self.tolerance = 1e-4
        self.c = 1

    def computeIndex(self, arm):
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return self.klucb(self.d_rewards[arm] / self.d_pulls[arm], self.c * np.log(self.d_t) / self.d_pulls[arm],
                              self.tolerance)

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = self.klucb_vect(
            self.d_rewards / self.d_pulls, self.c * np.log(self.d_t) / self.d_pulls, self.tolerance)
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes
