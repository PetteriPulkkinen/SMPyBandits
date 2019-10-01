import numpy as np
from typing import Callable

__author__ = "Petteri Pulkkinen"
__version__ = "0.6"


class SelectionAlgorithm(object):
    """Algorithm for combinatorial problems"""

    def __init__(self, n_super_arms: int):
        self.n_super_arms = n_super_arms

    def find_combination(self, indexes):
        raise NotImplementedError("This method needs to be implemented!")


class ExhaustiveSearch(SelectionAlgorithm):

    def __init__(self, actions: np.array, objective: Callable[[np.array, np.array], float]):
        super(ExhaustiveSearch, self).__init__(actions.shape[0])
        self.actions = actions
        self.objective = objective

    def find_combination(self, indexes):
        res = self.objective(indexes, self.actions)
        max_idx = np.random.choice(np.nonzero(res == np.max(res))[0])
        return self.actions[max_idx, :], max_idx
