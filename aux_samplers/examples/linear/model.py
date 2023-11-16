import jax

import jax.numpy as jnp
import jax.random as jr
import jax.lax as jl


def ode(
    x: jnp.ndarray, u: jnp.ndarray
) -> jnp.ndarray:

    A = jnp.array(
        [
            [0.0, 1.0],
            [0.0, 0.0]
        ]
    )
    B = jnp.array(
        [
            [0.0],
            [1.0]
        ]
    )
    return A @ x + B @ u


def get_dynamics(
    params: jnp.ndarray,
    step_size: jnp.float64
):
    def mean_fcn(state, _params):
        l, w = params

        x = jnp.atleast_1d(state[:2])
        u = jnp.atleast_1d(state[2:])

        xn = x + step_size * ode(x, u)
        un = jnp.exp(- 1.0 * step_size / l) * u
        return jnp.hstack((xn, un))

    def covar_fcn(state, _params):
        l, w = params  # l, q
        sigma_sqr = (w * l / 2.0) * (1.0 - jnp.exp(-2.0 * step_size / l))
        diag_u = jnp.atleast_1d(sigma_sqr)
        diag_x = jnp.array([1e-4, 5e-4])
        return jnp.diag(jnp.hstack((diag_x, diag_u)))

    return mean_fcn, covar_fcn


@jax.jit
def grad_log_potential(state):
    return jax.grad(log_potential)(state)


@jax.jit
def hess_log_potential(state):
    return jax.hessian(log_potential)(state)


@jax.jit
def log_potential(state):
    goal = jnp.array([0.0, 0.0, 0.0])
    weights = jnp.array([1e1, 1e0, 1e0])
    cost = - 0.5 * jnp.dot(state - goal, weights * (state - goal))
    return 1e-1 * cost


def simulate_trajectory(
    key,
    nb_steps,
    init_mean,
    init_covar,
    trans_mean_fcn,
    trans_covar_fcn,
):

    def body(carry, args):
        key, prev_state = carry
        key, sub_key = jr.split(key, 2)

        trans_covar = trans_covar_fcn(prev_state, None)
        trans_covar_chol = jnp.linalg.cholesky(trans_covar)
        next_state = trans_mean_fcn(prev_state, None) \
                     + trans_covar_chol @ jr.normal(sub_key, shape=(3,))
        return (key, next_state), next_state

    key, sub_key = jr.split(key, 2)

    init_covar_chol = jnp.linalg.cholesky(init_covar)
    init_state = init_mean + init_covar_chol @ jr.normal(sub_key, shape=(3,))
    _, states = jl.scan(body, (key, init_state), (), length=nb_steps - 1)

    states = jnp.insert(states, 0, init_state, 0)
    return states
