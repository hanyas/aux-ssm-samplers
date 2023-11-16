from functools import partial

import jax
import jax.numpy as jnp


def wrap_angle(angle: float) -> float:
    return angle % (2.0 * jnp.pi)


def ode(
    x: jnp.ndarray,
    u: jnp.ndarray,
) -> jnp.ndarray:

    length, mass = 1.0, 1.0
    gravity, damping = 9.81, 1e-3

    q, q_dot = x
    u = jnp.clip(u, -5.0, 5.0)

    return jnp.hstack(
        (
            q_dot,
            - gravity / length * jnp.sin(q)
            + (u - damping * q_dot) / (mass * length**2),
        )
    )


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
    q, q_dot = state[:2]
    u = jnp.atleast_1d(state[2:])

    g0 = jnp.array([jnp.pi, 0.0])

    Q = jnp.diag(jnp.array([1e1, 1e-1]))
    R = jnp.diag(jnp.array([1e-3]))

    xw = jnp.hstack((wrap_angle(q), q_dot))
    cost = (xw - g0).T @ Q @ (xw - g0)
    cost += u.T @ R @ u
    return - 0.5 * 2.0 * cost
