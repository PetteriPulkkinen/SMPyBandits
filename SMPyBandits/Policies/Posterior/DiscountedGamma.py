__author__ = "Lilian Besson"
__version__ = "0.9"

from .Gamma import Gamma


class DiscountedGamma(Gamma):

    def __init__(self, gamma, k=1, lmbda=1):
        super(DiscountedGamma, self).__init__(k, lmbda)
        self.disc = gamma

    def update(self, obs):
        self.discount()
        self.k += obs
        self.lmbda += 1

    def discount(self):
        self.k = self.disc * self.k
        self.lmbda = self.disc * self.lmbda
