# -*- coding: utf-8 -*-
""" Base class for any centralized policy, for the multi-players setting."""
from __future__ import division, print_function  # Python 2 compatibility
from .CombinatorialPolicy import CombinatorialPolicy
import numpy as np


__author__ = "Petteri Pulkkinen"
__version__ = "0.6"


class RandomCombinatorialPolicy(CombinatorialPolicy):
    """ Base class for any combinatorial policy, for the combinatorial setting."""

    def choice(self):
        """ Choose an arm."""
        if np.random.rand() < self.mab_algorithm.epsilon:
            idx = np.random.randint(low=0, high=self.opt_algorithm.n_super_arms)
            return self.opt_algorithm.actions[idx], idx
        else:
            belief = self.mab_algorithm.rewards / (1 + self.mab_algorithm.pulls)
            return self.opt_algorithm.find_combination(belief)
