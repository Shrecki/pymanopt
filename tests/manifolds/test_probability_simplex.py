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

    def test_log(self):
        p = np.array([1.0 / 2, 1.0 / 3, 1.0 / 6])
        q = np.array([0.2, 0.3, 0.5])
        X = self.manifold.log(p, q)
        assert np.allclose(
            X,
            np.array(
                [
                    -0.2965121489310675,
                    0.018867090771124348,
                    0.27764505815994317,
                ]
            ),
        )

        p = np.array([1.0 / 2, 1.0 / 4, 1.0 / 4])
        X = self.manifold.log(p, q)
        assert np.allclose(
            X,
            np.array(
                [-0.3171678647294765, 0.07736015116476168, 0.23980771356471503]
            ),
        )

    def test_exp(self):
        p = np.array([1.0 / 2, 1.0 / 4, 1.0 / 4])
        q = np.array([0.2, 0.3, 0.5])
        X = self.manifold.exp(p, q)
        assert np.allclose(
            X,
            np.array(
                [0.5136415708782431, 0.4830097714918066, 0.7800478959359726]
            ),
        )

    def test_projection(self):
        p = np.array([1 / 2, 1 / 3, 1 / 6])
        q = np.array([0.2, 0.3, 0.5])
        X = self.manifold.log(p, q)
        X2 = X + 10
        Y = self.manifold.projection(p, X2)
        assert np.allclose(X, Y)

    def test_euclidean_to_riemannian_gradient(self):
        M = ProbabilitySimplex(4)  # n=5
        # For f(p) = \sum_i 1/n log(p_i) we know the Euclidean gradient

        def grad_f_eucl(p):
            return 1.0 / (5 * p)

        # but we can also easily derive the Riemannian one

        def grad_f(p):
            return 1.0 / 5 - p

        # We take some point
        p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        Y = grad_f_eucl(p)
        X = grad_f(p)
        # print(M.euclidean_to_riemannian_gradient(p,Y))
        # print(M.projection(p,X))
        # @test isapprox(M, p, X, riemannian_gradient(M, p, Y))
        # Z = zero_vector(M, p)
        # riemannian_gradient!(M, Z, p, Y)
        # @test X == Z
        assert np.allclose(
            M.euclidean_to_riemannian_gradient(p, Y), M.projection(p, X)
        )

    # def test_norm

    # def test_random_point

    # def test_random_tangent_vector

    # def test_zero_vector

    # def test_dist

    # def test_euclidean_to_riemannian_gradient

    # def test_euclidean_to_riemannian_hessian

    # def test__normalize

    # def test_retraction
