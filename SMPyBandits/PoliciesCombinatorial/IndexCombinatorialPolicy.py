# -*- coding: utf-8 -*-
""" Base class for any centralized policy, for the multi-players setting."""
from __future__ import division, print_function  # Python 2 compatibility
from .CombinatorialPolicy import CombinatorialPolicy

__author__ = "Petteri Pulkkinen"
__version__ = "0.6"


class IndexCombinatorialPolicy(CombinatorialPolicy):
    """ Base class for any combinatorial policy, for the combinatorial setting."""

    def choice(self):
        """ Choose an arm."""
        self.mab_algorithm.computeAllIndex()
        return self.opt_algorithm.find_combination(self.mab_algorithm.index)
