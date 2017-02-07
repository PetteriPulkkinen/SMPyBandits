# -*- coding: utf-8 -*-
""" The UCB1 (UCB-alpha) index policy, modified to take a random permutation order for the initial exploration of each arm (reduce collisions in the multi-players setting).
Note: using log10(t) and not ln(t) for UCB index.
Reference: [Auer et al. 02].
"""

__author__ = "Lilian Besson"
__version__ = "0.2"

from math import sqrt, log10

from .UCBlog10 import UCBlog10


class UCBlog10alpha(UCBlog10):
    """ The UCB1 (UCB-alpha) index policy, modified to take a random permutation order for the initial exploration of each arm (reduce collisions in the multi-players setting).
    Note: using log10(t) and not ln(t) for UCB index.
    Reference: [Auer et al. 02].
    """

    def __init__(self, nbArms, alpha=4, lower=0., amplitude=1.):
        super(UCBlog10alpha, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        assert alpha > 0, "Error: the alpha parameter for UCBalpha class has to be > 0."
        self.alpha = alpha

    def __str__(self):
        return r"UCB($\alpha={:.3g}$, {})".format(self.alpha, r"$\log_{10}$")

    def computeIndex(self, arm):
        if self.pulls[arm] < 2:
            return float('+inf')
        else:
            mean = self.rewards[arm] / self.pulls[arm]   # Mean estimate
            return mean + sqrt((self.alpha * log10(self.t)) / (2 * self.pulls[arm]))