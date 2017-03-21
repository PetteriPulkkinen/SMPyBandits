# -*- coding: utf-8 -*-
""" Binomial distributed arm."""

__author__ = "Lilian Besson"
__version__ = "0.5"

from random import random
from numpy.random import binomial as npbinomial

from .Arm import Arm
from .kullback import klBin


def binomial(n, p):
    """ Manual implementation of random sample from Bin(n, p), with random.random() <= p summed n times."""
    return sum(float(random() <= p) for _ in range(n))


class Binomial(Arm):
    """ Binomial distributed arm."""

    def __init__(self, probability, draws=1):
        assert 0 <= probability <= 1, "Error, the parameter probability for Binomial class has to be in [0, 1]."
        assert isinstance(draws, int) and 1 <= draws, "Error, the parameter draws for Binomial class has to be an integer >= 1."
        self.probability = probability
        self.draws = draws
        self.mean = probability * draws

    # --- Random samples

    def draw(self, t=None):
        """ The parameter t is ignored in this Arm."""
        return binomial(self.draws, self.probability)

    def draw_nparray(self, shape=(1,)):
        """ The parameter t is ignored in this Arm."""
        return npbinomial(self.draws, self.probability, shape)

    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def lower_amplitude(self):
        return 0., self.draws

    def __str__(self):
        return "Binomial"

    def __repr__(self):
        return "Bin({:.3g}, {})".format(self.probability, self.draws)

    # --- Lower bound

    def kl(self, x, y):
        return klBin(x, y, self.draws)

    def oneLR(self, mumax, mu):
        """ One term of the Lai & Robbins lower bound for Binomial arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / klBin(mu, mumax, self.draws)