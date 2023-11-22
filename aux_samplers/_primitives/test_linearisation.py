import jax
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from .linearisation import extended, gauss_hermite, cubature, gauss_hermite_non_additive


@pytest.fixture(scope="module", autouse=True)
def jax_config():
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)


def test_linear():
    A = np.random.randn(2, 4)
    b = np.random.randn(2)

    q = np.random.randn(2, 5)
    Q = q @ q.T

    def mean(x, _):
        return A @ x + b

    def cov(*_):
        return Q

    x_star = np.random.randn(4)
    p_star = np.random.randn(4, 10)
    P_star = p_star @ p_star.T

    F_e, Q_e, b_e = extended(mean, cov, None, x_star, P_star)
    F_gh, Q_gh, b_gh = gauss_hermite(mean, cov, None, x_star, P_star)
    F_c, Q_c, b_c = cubature(mean, cov, None, x_star, P_star)

    np.testing.assert_allclose(F_e, F_gh)
    np.testing.assert_allclose(Q_e, Q_gh)
    np.testing.assert_allclose(b_e, b_gh)

    np.testing.assert_allclose(F_e, F_c)
    np.testing.assert_allclose(Q_e, Q_c)
    np.testing.assert_allclose(b_e, b_c)

    np.testing.assert_allclose(F_e, A)
    np.testing.assert_allclose(Q_e, Q)
    np.testing.assert_allclose(b_e, b)

    # Tests for the non-additive gauss hermite.
    def aug_mean(x, q, _):
        return A @ x + b + jnp.linalg.cholesky(Q) @ q

    F_nagh, Q_nagh, b_nagh = gauss_hermite_non_additive(aug_mean, None, x_star, P_star, 2)
    np.testing.assert_allclose(F_nagh, F_gh)
    np.testing.assert_allclose(b_nagh, b_gh)
    np.testing.assert_allclose(Q_nagh, Q_gh)