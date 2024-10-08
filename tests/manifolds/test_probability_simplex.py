# import warnings

import autograd.numpy as np
import pytest
from numpy import testing as np_testing

# import pymanopt
from pymanopt.manifolds import ProbabilitySimplex


# from pymanopt.tools import testing


class TestSphereManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.m = m = 100
        self.manifold = ProbabilitySimplex(m, False)

        # For automatic testing of euclidean_to_riemannian_hessian
        # self.projection = lambda x, u: u - np.tensordot(x, u, np.ndim(u)) * x

    def test_dim(self):
        assert self.manifold.dim == self.m

    def test_typical_dist(self):
        np_testing.assert_almost_equal(
            self.manifold.typical_dist, np.pi * 0.5 * (self.m)
        )

    def test_inner_product(self):
        self.manifold = ProbabilitySimplex(2, False)
        p = np.array([0, 0.5, 0.5])
        X = np.array([0, 1, -1])
        Y = np.array([0, 2, -2])
        assert self.manifold.inner_product(p, X, Y) == 8

    # def test_projection

    # def test_norm

    # def test_random_point

    # def test_random_tangent_vector

    # def test_change_metric_euclidean

    # def test_zero_vector

    # def test_dist

    # def test_euclidean_to_riemannian_gradient

    # def test_euclidean_to_riemannian_hessian

    # def test_exp

    # def test_log

    # def test__normalize

    # def test_retraction
