# -*- coding: utf-8 -*-
""" The RBE policy for recency based index exploration.

- Reference: [Oksanen & Koivunen, 2012].
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Petteri Pulkkinen"
__version__ = "0.9"

import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

try:
    from .IndexPolicy import IndexPolicy
except ImportError:
    from IndexPolicy import IndexPolicy


class RBE(IndexPolicy):
    """ The UCB policy for bounded bandits.

    - Reference: [Lai & Robbins, 1985].
    """
    def __init__(self, nbArms, **kwargs):
        super(RBE, self).__init__(nbArms, **kwargs)
        self.last_t = np.zeros(nbArms)

    def startGame(self):
        super(RBE, self).startGame()
        self.last_t.fill(0)

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:

        .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{2 \log(t)}{N_k(t)}}.
        """
        if self.last_t[arm] < 1:
            return float('+inf')
        else:
            return (self.rewards[arm] / self.pulls[arm]) + np.sqrt(np.log(self.t / self.last_t[arm]))

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = (self.rewards / self.pulls) + np.sqrt(np.log(self.t / self.last_t))
        indexes[self.last_t < 1] = float('+inf')
        self.index[:] = indexes

    def getReward(self, arm, reward):
        self.last_t[arm] = self.t
        return super(RBE, self).getReward(arm, reward)

# --- Debugging


if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)

