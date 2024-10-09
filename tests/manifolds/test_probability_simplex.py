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
        # print(X)
        X2 = X + 10
        # print(X2)
        Y = self.manifold.projection(p, X2)
        assert np.allclose(
            np.array(
                [-5.296512148931068, 0.018867090771124, 5.277645058159942]
            ),
            Y,
        )

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

        p = np.array([0.2, 0.4, 0.1, 0.3])
        grad = np.array([2, -3, 0.5, 4])
        exp_grad = np.array([0.31, -1.38, 0.005, 1.065])
        assert np.allclose(
            M.euclidean_to_riemannian_gradient(p, grad), exp_grad
        )

    def test_dist(self):
        M = ProbabilitySimplex(4)  # n=5
        p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        q = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
        expectedDist = 2 * np.arccos(np.sum(np.sqrt(p * q)))
        actualDist = M.dist(p, q)
        assert np.allclose(expectedDist, actualDist)

    def test_euclidean_to_riemannian_hessian(self):
        M = ProbabilitySimplex(4)  # n=5
        p = np.array([0.2, 0.4, 0.1, 0.3])
        egrad = np.array([2, -3, 0.5, 4])
        ehess = np.array(
            [[2, 0.4, 3, 2], [-1, 10, 9, -2], [11, 20, 0.1, 30], [4, 2, 3, -3]]
        )

        expected = np.array(
            [
                [0.095, -1.1010, -0.267, 0.215],
                [-2.01, 0.638, 0.866, -2.17],
                [0.8725, 1.3345, -0.4985, 2.8325],
                [1.0425, -0.8715, -0.1005, -0.8775],
            ]
        )
        actual = M.euclidean_to_riemannian_hessian(p, egrad, ehess, p)
        assert np.allclose(expected, actual)

        # Similar test, but with n=6
        M = ProbabilitySimplex(5)  # n=5
        p = np.array([0.2, 0.4, 0.1, 0.1, 0.2])
        egrad = np.array([2, -3, 0.5, 4, 3])
        ehess = np.array(
            [
                [2, 0.4, 3, 2, 2],
                [-1, 10, 9, -2, 45],
                [11, 20, 0.1, 30, 28],
                [4, 2, 3, -3, 12],
                [28, 2, -30, 4.4, -54.2],
            ]
        )

        expected = np.array(
            [
                [-0.845, -1.081, 1.073, -0.061, -1.737],
                [-3.89, 0.678, 3.546, -2.722, 12.726],
                [0.4025, 1.3445, 0.1715, 2.6945, 1.6565],
                [-0.1225, -0.2805, 0.6365, -0.4305, 0.2315],
                [4.455, -0.661, -5.427, 0.519, -12.877],
            ]
        )
        actual = M.euclidean_to_riemannian_hessian(p, egrad, ehess, p)
        assert np.allclose(expected, actual)

        # assert False

    # def test_norm

    # def test_random_point

    # def test_random_tangent_vector

    # def test_zero_vector

    # def test_euclidean_to_riemannian_hessian

    # def test__normalize

    # def test_retraction
