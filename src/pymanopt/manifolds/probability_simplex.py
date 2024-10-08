import numpy as np
import scipy

from pymanopt.manifolds.manifold import Manifold


def usinc(theta: float):
    r = 0
    if theta == 0:
        r = 1
    elif np.isinf(theta) or not np.isfinite(theta):
        r = 0
    else:
        r = np.sin(theta) / theta
    return r


class ProbabilitySimplex(Manifold):
    r"""Manifold of probability vectors.

    The (relative interior of) the probability simplex.
    A probability vector is simply a vector with non-negative
    entries summing up to 1.

    Args:
        m: Dimension of the subspace

        boundary_open: If true, boundary is open, otherwise it is false.

    Note:
        A m+1 vector of non-negative coefficients summing to one is fully
        represented in the embedding by m coefficients. Be careful to not
        specify wrongly m to be the dimension of the original vector.
    """

    def __init__(self, m: int, boundary_open: bool = False):
        self._m = m
        self._boundary_open = boundary_open

        name = f"Manifold of {m+1} dimensional probability vectors"
        dimension = int(m)
        super().__init__(name, dimension)

    @property
    def typical_dist(self):
        return np.pi * self._m * 0.5

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        """Inner product between tangent vectors at a point on the manifold.

        This method implements a Riemannian inner product between two tangent vectors
        tangent_vector_a and tangent_vector_b in the tangent space at point.
        When the manifold includes boundary, we skip coordinates where pi is
        equal to zero, see Proposition 2.1 in [AJLS17]_

        Args:
            point: The base point.
            tangent_vector_a: The first tangent vector.
            tangent_vector_b: The second tangent vector.

        Returns:
            out: The inner product between tangent_vector_a and tangent_vector_b
            in the tangent space at point.
        """
        d = 0
        if self._boundary_open:
            d = (tangent_vector_a / point) @ tangent_vector_b
        else:
            mask = point > 0
            d = (tangent_vector_a[mask] / point[mask]) @ tangent_vector_b[mask]
        return d

    def projection(self, point, vector):
        r"""Projects vector from the embedding onto the tangent space at point.

        The formula reads :math: proj_{\nabla^n}(vector, point) =
        Y - \bar{Y}, where \bar{Y} denotes mean of Y.

        Args:
            point : A point on the manifold
            vector : A vector of the ambient space of the tangent space at point.

        Returns:
            out: An element of the tangent space at point closest to vector
            in the ambient space.
        """
        return vector - np.mean(vector)

    to_tangent_space = projection

    def norm(self, point, tangent_vector):
        return np.sqrt(
            self.inner_product(point, tangent_vector, tangent_vector)
        )

    def random_point(self):
        vector = np.random.normal(size=self._m)
        vector /= np.linalg.norm(vector)
        # Could be made faster, see
        # https://stackoverflow.com/questions/30437947/most-memory-efficient-way-to-compute-abs2-of-complex-numpy-ndarray
        return np.abs(vector) ** 2

    def random_tangent_vector(self, point):
        vector = np.random.normal(size=point.shape)
        vector -= np.mean(vector)
        return self.change_metric_euclidean(point, vector)

    def change_metric_euclidean(self, point, X):
        r"""Change point metric to euclidean.

        Map with :math: c: T_p \nabla^n \rightarrow T_p \nabla^n
        such that for all X, Y \in T_p \nabla^n. This can be achieved by rewriting
        representer change in matrix form as
        (diag(point) - point point^T) X and taking the square root of the matrix.

        Args:
            point: A point on the manifold.
            X: A vector on the manifold.

        Returns:
            out: A tangent vector in tangent space.
        """
        d = np.diag(point) - np.outer(point, point)

        # Compute matrix square root!
        return scipy.linalg.sqrtm(d) @ X

    def zero_vector(self, point):
        return np.zeros(point.shape)

    def dist(self, point_a, point_b):
        """The geodesic distance between two points on the manifold.

        For a probability simplex, the formula reads :math: 2arccos(sum sqrt(p_i q_i))

        Args:
            point_a: The first point on the manifold.
            point_b: The second point on the manifold.

        Returns:
            out: The distance between point_a and point_b on the manifold.
        """
        return 2 * np.arccos(np.sqrt(point_a) @ np.sqrt(point_b))

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return (
            point * euclidean_gradient - (point @ euclidean_gradient) * point
        )

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        return self.projection(point, euclidean_hessian)

    def exp(self, point, tangent_vector):
        r"""Computes the exponential map on the manifold.

        In the case of the probability simplex, the formula
        is: :math: exp_p X = 0.5(p + X_p^2 / ||X_p||^2) +
        0.5(p - X_p^2 // || X_p||^2)cos(||X_p||) + 1/||X_p|| sqrt(p)sin(||X_p||)
        X_p = X / sqrt(p) (elem-wise division)

        Args:
            point: A point on the manifold.
            tangent_vector: A tangent vector at point.

        Returns:
            out: The point on the manifold reached by moving away from
            point along a geodesic in the direction of tangent_vector
        """
        s = np.sqrt(point)
        Xs = 0.5 * (tangent_vector / s)
        theta = np.linalg.norm(Xs)
        q = (np.cos(theta) * s + usinc(theta) * tangent_vector) ** 2
        return q

    def log(self, point_a, point_b):
        """Computes the logarithmic map on the manifold.

        The logarithmic map log(point_a, point_b) produces a
        tangent vector in the tangent space at point_a
        that points in the direction of point_b.
        In other words,
        exp(point_a, log(point_a, point_b)) == point_b.
        As such it is the inverse of exp.

        Args:
            point_a: First point on the manifold.
            point_b:Second point on the manifold.

        Returns:
            out:A tangent vector in the tangent space at point_a.
        """
        X = np.zeros((self._m + 1))
        if not np.allclose(point_a, point_b):
            z = self.dist(point_a, point_b)
            s = np.sum(z)
            X = (2 * np.arccos(s) / (1.0 - s**2)) * (z - s * point_a)
        return X

    @staticmethod
    def _normalize(point):
        return point / np.sum(point)

    def retraction(self, point, tangent_vector):
        return self._normalize(point + tangent_vector)
