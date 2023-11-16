from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from aux_samplers import mvn, extended
from aux_samplers.kalman import get_kernel as get_generic_kernel
from model import get_dynamics, log_potential, grad_log_potential, hess_log_potential


def get_kernel(
    nb_steps: jnp.int64,
    init_dist: Tuple,
    params: jnp.ndarray,
    step_size: jnp.float64,
    parallel: bool = False,
):
    m0, P0 = init_dist
    dim = m0.shape[0]

    eye = jnp.eye(dim)
    zero = jnp.zeros((dim,))

    eyes = jnp.repeat(eye[None, ...], nb_steps, axis=0)
    zeros = jnp.zeros((nb_steps, dim))

    chol_P0 = jnp.linalg.cholesky(P0)

    mean_fcn, covar_fcn = get_dynamics(params, step_size)
    specialized_extended = lambda z: extended(mean_fcn, covar_fcn, None, z, None)

    def dynamics_factory(z):
        Fs, Qs, bs = jax.vmap(specialized_extended)(z[:-1])
        return m0, P0, Fs, Qs, bs

    def _observations_factory(z, v, delta):
        # For a single time step!
        grad = grad_log_potential(z)
        hess = hess_log_potential(z)
        Omega_inv = - hess + 2.0 * eye / delta
        chol_Omega_inv = jnp.linalg.cholesky(Omega_inv)
        Omega = cho_solve((chol_Omega_inv, True), eye)
        aux_y = Omega @ (2.0 * v / delta + grad - hess @ z)
        return aux_y, eye, Omega, zero

    def observations_factory(z, v, delta):
        return jax.vmap(_observations_factory, in_axes=(0, 0, None))(z, v, delta)

    def log_likelihood_fcn(z):
        ms = jax.vmap(mean_fcn)(z[:-1], None)
        Qs = jax.vmap(covar_fcn)(z[:-1], None)
        chol_Qs = jnp.linalg.cholesky(Qs)

        out = mvn.logpdf(z[0], m0, chol_P0)
        out += jnp.sum(mvn.logpdf(z[1:], ms, chol_Qs))
        return out + jnp.sum(jax.vmap(log_potential)(z))

    return get_generic_kernel(dynamics_factory, observations_factory, log_likelihood_fcn, parallel)
