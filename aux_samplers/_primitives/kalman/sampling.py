from functools import partial

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from jax.scipy.linalg import solve

from .base import LGSSM


def sampling(key: PRNGKey, ms: Array, Ps: Array, lgssm: LGSSM, parallel: bool) -> Array:
    """
    Samples from the pathwise smoothing distribution a LGSSM.

    Parameters
    ----------
    key: PRNGKey
        Random number generator key.
    ms: Array
        Filtering means of the LGSSM.
    Ps: Array
        Filtering covariances of the LGSSM.
    lgssm: LGSSM
        LGSSM model to be sampled from.
    parallel: bool
        Whether to run the sampling in _parallel.
    """
    Fs, Qs, bs = lgssm.Fs, lgssm.Qs, lgssm.bs
    mean_incs, cov_incs, gains, sample_incs = \
        _sampling_init(key, ms, Ps, Fs, Qs, bs)  # noqa: bad static type checking
    if parallel:
        means, covs, _, samples = jax.lax.associative_scan(
            jax.vmap(_sampling_op),
            (mean_incs, cov_incs, gains, sample_incs),
            reverse=True
        )
    else:
        def body(carry, inputs):
            carry = _sampling_op(carry, inputs)
            return carry, carry

        _, (means, covs, _, samples) = jax.lax.scan(
            body,
            (mean_incs[-1], cov_incs[-1], gains[-1], sample_incs[-1]),
            (mean_incs[:-1], cov_incs[:-1], gains[:-1], sample_incs[:-1]),
            reverse=True
        )

        means = jnp.append(means, mean_incs[None, -1, ...], 0)
        covs = jnp.append(covs, cov_incs[None, -1, ...], 0)
        samples = jnp.append(samples, sample_incs[None, -1, ...], 0)
    return means, covs, samples


# Operator
def _sampling_op(elem1, elem2):
    m1, P1, G1, e1 = elem1
    m2, P2, G2, e2 = elem2
    return _sampling_op_impl(m1, P1, G1, e1, m2, P2, G2, e2)


@partial(jnp.vectorize, signature='(dx),(dx,dx),(dx,dx),(dx),(dx),(dx,dx),(dx,dx),(dx)->(dx),(dx,dx),(dx,dx),(dx)')
def _sampling_op_impl(m1, P1, G1, e1, m2, P2, G2, e2):
    m = G2 @ m1 + m2
    P = G2 @ P1 @ G2.T + P2
    G = G2 @ G1
    e = G2 @ e1 + e2
    return m, P, G, e


# Initialization

@partial(jnp.vectorize, signature="(dx,dx),(dx,dx),(dx),(dx),(dx,dx)->(dx),(dx,dx),(dx,dx),(dx,dx)")
def mean_and_chol(F, Q, b, m, P):
    """
    Computes the increments means and Cholesky decompositions for the backward sampling steps.

    Parameters
    ----------
    F: Array
        Transition matrix for time t to t+1.
    Q: Array
        Transition covariance matrix for time t to t+1.
    b:
        Transition offset for time t to t+1.
    m: Array
        Filtering mean at time t
    P: Array
        Filtering covariance at time t
    Returns
    -------
    m: Array
        Increment mean to go from time t+1 to t.
    chol: Array
        Cholesky decomposition of the increment covariance to go from time t+1 to t.
    gain: Array
        Gain to go from time t+1 to t.
    """
    dim = m.shape[0]
    S = F @ P @ F.T + Q  # noqa: bad static type checking
    S = 0.5 * (S + S.T)

    if dim == 1:
        gain = P * F / S
    else:
        gain = P @ solve(S, F, assume_a="pos").T

    inc_m = m - gain @ (F @ m + b)
    inc_Sig = P - gain @ S @ gain.T
    inc_Sig = 0.5 * (inc_Sig + inc_Sig.T)

    if dim == 1:
        L = jnp.sqrt(inc_Sig)
    else:
        L = jnp.linalg.cholesky(inc_Sig)
    # When there is 0 uncertainty, the Cholesky decomposition is not defined.
    L = jnp.nan_to_num(L)
    return inc_m, inc_Sig, gain, L


@partial(jnp.vectorize, signature="(dx,dx),(dx,dx),(dx),(dx),(dx,dx),(dx)->(dx),(dx,dx),(dx,dx),(dx)")
def _sampling_init_one(F, Q, b, m, P, eps):
    inc_m, inc_P, gain, L = mean_and_chol(F, Q, b, m, P)
    inc_sample = inc_m + L @ eps
    return inc_m, inc_P, gain, inc_sample


@partial(jnp.vectorize, signature="(dx),(dx,dx),(dx)->(dx),(dx,dx),(dx,dx),(dx)")
def _sample_last_step(m, P, eps):
    if P.shape[0] == 1:
        L = jnp.sqrt(P)
    else:
        L = jnp.linalg.cholesky(P)
    L = jnp.nan_to_num(L)
    last_sample = m + L @ eps
    gain = jnp.zeros_like(P)
    return m, P, gain, last_sample


def _sampling_init(key, ms, Ps, Fs, Qs, bs):
    epsilons = jax.random.normal(key, shape=ms.shape)

    mean_incs, cov_incs, gains, sample_incs = \
        jax.vmap(_sampling_init_one)(Fs, Qs, bs, ms[:-1], Ps[:-1], epsilons[:-1])

    # When we condition on the last step this is 0 and Cholesky ain't liking this.
    last_mean_inc, last_cov_inc, last_gain, last_sample_inc = \
        _sample_last_step(ms[-1], Ps[-1], epsilons[-1])

    mean_incs = jnp.append(mean_incs, last_mean_inc[None, ...], 0)
    cov_incs = jnp.append(cov_incs, last_cov_inc[None, ...], 0)
    gains = jnp.append(gains, last_gain[None, ...], 0)
    sample_incs = jnp.append(sample_incs, last_sample_inc[None, ...], 0)
    return mean_incs, cov_incs, gains, sample_incs
