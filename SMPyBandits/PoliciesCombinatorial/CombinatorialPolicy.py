# -*- coding: utf-8 -*-
""" Base class for any centralized policy, for the multi-players setting."""
from __future__ import division, print_function  # Python 2 compatibility
from SMPyBandits.Policies import IndexPolicy
import numpy as np
from typing import Callable

__author__ = "Petteri Pulkkinen"
__version__ = "0.6"


class SelectionAlgorithm(object):
    """Algorithm for combinatorial problems"""

    def find_combination(self, indexes):
        raise NotImplementedError("This method needs to be implemented!")


class ExhaustiveSearch(SelectionAlgorithm):

    def __init__(self, actions: np.array, objective: Callable[[np.array, np.array], float]):
        self.actions = actions
        self.objective = objective

    def find_combination(self, indexes):
        res = self.objective(indexes, self.actions)
        return self.actions[np.argmax(res), :]


class CombinatorialPolicy(object):
    """ Base class for any combinatorial policy, for the combinatorial setting."""

    def __init__(self, cmb_size: int, opt_algorithm: SelectionAlgorithm, mab_algorithm: IndexPolicy):
        self.cmb_size = cmb_size
        self.opt_algorithm = opt_algorithm
        self.mab_algorithm = mab_algorithm

    def startGame(self):
        self.mab_algorithm.startGame()

    def getReward(self, arm, reward):
        self.mab_algorithm.getReward(arm, reward)

    def choice(self):
        """ Choose an arm."""
        self.mab_algorithm.computeAllIndex()
        return self.opt_algorithm.find_combination(self.mab_algorithm.index)
