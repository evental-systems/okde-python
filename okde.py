# Copyright 2017 by Evental Systems LLC

from __future__ import division, print_function, absolute_import

# Standard library imports.
import warnings

# Scipy imports
from scipy import linalg
from numpy import atleast_2d, reshape, zeros, newaxis, dot, exp, pi, sqrt, \
    ravel, power, atleast_1d, squeeze, sum, transpose
import numpy as np
from numpy.random import randint, multivariate_normal

class OnlineGaussianKde(object):
    def __init__(self, dimension):
        self.dimension = dimension
        self.components = []
        self.global_weight = 0.0

    def update(self, point, weight):
        if weight <= 0:
            raise ValueError("`weight` must be positive; was " + weight)
        d = point.shape[0]
        if d != self.dimension:
            raise ValueError("expected `point` to have dimension " + self.dimension + "; was " + d)

        new_component = OneComponentDist(point, zeros((d, d), np.float64), weight)
        self.components.append(new_component)
        self.global_weight += weight


class OneComponentDist(object):
    def __init__(self, mean, covariance, weight):
        self.mean = mean
        self.covariance = covariance
        self.weight = weight

    def global_mean(self):
        return self.mean

    def global_covariance(self):
        return self.covariance
