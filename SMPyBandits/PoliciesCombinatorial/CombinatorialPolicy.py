# -*- coding: utf-8 -*-
""" Base class for any centralized policy, for the multi-players setting."""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Petteri Pulkkinen"
__version__ = "0.6"


class CombinatorialPolicy(object):
    """ Base class for any combinatorial policy, for the combinatorial setting."""

    def __init__(self, opt_algorithm, mab_algorithm):
        self.opt_algorithm = opt_algorithm
        self.mab_algorithm = mab_algorithm

    def startGame(self):
        self.mab_algorithm.startGame()

    def getReward(self, arm, reward):
        self.mab_algorithm.getReward(arm, reward)

    def choice(self):
        """ Choose an arm."""
        raise NotImplementedError('You need to implement this method!')
