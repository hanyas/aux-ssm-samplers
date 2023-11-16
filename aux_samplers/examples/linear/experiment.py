import time

import jax
import jax.numpy as jnp
import numpy as np

from jax.tree_util import tree_map
from jax.experimental.host_callback import call

from aux_samplers.common import delta_adaptation
from auxiliary_kalman import get_kernel as get_kalman_kernel
from model import simulate_trajectory, get_dynamics

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


# STATS FN
def stats_fn(x_1, x_2):
    # squared jumping distance averaged across dimensions, and first and second moments
    return (x_2 - x_1) ** 2, x_2, x_2 ** 2


# KERNEL
def loop(key, init_delta, target_alpha, lr, beta, init_state, kernel_fn, delta_fn, n_iter, verbose=False):
    from datetime import datetime
    keys = jax.random.split(key, n_iter)

    print_func = lambda z, *_: print(f"\riteration: {z[0]}, time: {datetime.now().strftime('%H:%M:%S')}, "
                                     f"min_δ: {z[1]:.2e}, max_δ: {z[2]:.2e}, "
                                     f"min_window_accept: {z[3]:.1%},  max_window_accept: {z[4]:.1%}, "
                                     f"min_accept: {z[5]:.1%}, max_accept: {z[6]:.1%}, "
                                     f"min_esjd: {z[7]:.2e}, max_esjd: {z[8]:.2e}", end="\n")

    def body_fn(carry, key_inp):
        from jax.experimental.host_callback import id_tap
        i, stats, state, delta, window_avg_acceptance, avg_acceptance = carry
        if verbose:
            id_tap(print_func, (i,
                                jnp.min(delta), jnp.max(delta),
                                jnp.min(window_avg_acceptance), jnp.max(window_avg_acceptance),
                                jnp.min(avg_acceptance), jnp.max(avg_acceptance),
                                jnp.min(jnp.sum(stats[0], -1)), jnp.max(jnp.sum(stats[0], -1)),
                                ), result=None)

        next_state = kernel_fn(key_inp, state, delta)
        next_stats = stats_fn(state.x, next_state.x)

        # moving average.
        avg_acceptance = (i * avg_acceptance + 1. * next_state.updated) / (i + 1)
        window_avg_acceptance = beta * next_state.updated + (1 - beta) * window_avg_acceptance
        stats = tree_map(lambda u, v: (i * u + v) / (i + 1), stats, next_stats)

        if delta_fn is not None:
            _lr = (n_iter - i) * lr / n_iter
            delta = delta_fn(delta, target_alpha, window_avg_acceptance, _lr)

        carry = i + 1, stats, next_state, delta, window_avg_acceptance, avg_acceptance
        return carry, next_state.x

    init_stats = stats_fn(init_state.x, init_state.x)
    init = 0, init_stats, init_state, init_delta, 1. * init_state.updated, 1. * init_state.updated

    out, samples = jax.lax.scan(body_fn, init, keys)
    return out, samples


NOW = time.time()


def tic_fn(arr):
    time_elapsed = time.time() - NOW
    return np.array(time_elapsed, dtype=arr.dtype), arr


# Experiment arguments
delta_init = 1e-2
target_alpha = 0.5  # 0.75 (second-order)
lr = 0.1
beta = 0.01

key = jax.random.PRNGKey(321)
burnin_key, sample_key, init_key = jax.random.split(key, 3)

# KERNEL
nb_steps = 51
step_size = 0.1

l, w = 0.5, 5.0
params = jnp.array([l, w])

init_mean = jnp.array([1.0, 2.0, 0.0])
init_covar = jnp.diag(
    jnp.array([1e-8, 1e-8, l * w / 2.0])
)
init_dist = (init_mean, init_covar)

init_fn, kernel_fn = get_kalman_kernel(
    nb_steps,
    init_dist,
    params,
    step_size,
)

trans_mean_fcn, trans_covar_fcn = get_dynamics(params, step_size)

xs_init = simulate_trajectory(
    init_key,
    nb_steps,
    init_mean,
    init_covar,
    trans_mean_fcn,
    trans_covar_fcn
)
# xs_init = 1e-4 * jax.random.normal(init_key, shape=(nb_steps, 3))
# xs_init = xs_init.at[0, :].set(init_mean)

init_state = init_fn(xs_init)

# BURNIN
(_, _, burnin_state, burnin_delta, burnin_avg_acceptance, _), burnin_samples = \
    loop(
        burnin_key,
        delta_init,
        target_alpha,
        lr,
        beta,
        init_state,
        kernel_fn,
        delta_adaptation,
        1_000,
        True
    )


output_shape = (
    jax.ShapeDtypeStruct((), burnin_delta.dtype),
    jax.ShapeDtypeStruct(burnin_delta.shape, burnin_delta.dtype)
)

tic, burnin_delta = call(
    tic_fn, burnin_delta, result_shape=output_shape
)

(_, stats, _, out_delta, _, pct_accepted), samples = \
    loop(
        sample_key,
        burnin_delta,
        target_alpha,
        lr,
        beta,
        burnin_state,
        kernel_fn,
        None,
        1000,
        True
    )

output_shape = (
    jax.ShapeDtypeStruct((), pct_accepted.dtype),
    jax.ShapeDtypeStruct(pct_accepted.shape, pct_accepted.dtype)
)

toc, _ = call(tic_fn, pct_accepted,
              result_shape=output_shape)

plt.plot(samples.mean(axis=0))
plt.show()
