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
        self.m = m = 2
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
        p = np.array([0, 0.5, 0.5])
        X = np.array([3, 1, -1])
        Y = np.array([2, 2, -2])
        assert self.manifold.inner_product(p, X, Y) == 8

    def test_change_metric_euclidean(self):
        p = np.array([1.0 / 2, 1.0 / 3, 1.0 / 6])
        X = np.array(
            [-0.2965121489310675, 0.018867090771124348, 0.27764505815994317]
        )
        # X2 = X + 10
        # Y = self.manifold.projection(p, X2)
        Y1 = self.manifold.change_metric_euclidean(p, X)
        assert np.allclose(
            Y1,
            np.array(
                [
                    -0.17062114054478128,
                    0.04002429219016789,
                    0.13059684835461377,
                ]
            ),
        )
        # @test isapprox(M, p, X, Y)

    # def test_norm

    # def test_random_point

    # def test_random_tangent_vector

    # def test_projection

    # def test_zero_vector

    # def test_dist

    # def test_euclidean_to_riemannian_gradient

    # def test_euclidean_to_riemannian_hessian

    # def test_exp

    # def test_log

    # def test__normalize

    # def test_retraction
