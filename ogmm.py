"""Online EM for Gaussian Mixtures."""
from functools import partial

import jax.numpy as jnp
from jax import jit, tree_map, vmap
from jax.lax import cond, fori_loop
from jax.scipy.linalg import eigh
from jax.scipy.special import logsumexp

from jax_tqdm import loop_tqdm

import numpy as np

from sklearn.mixture import GaussianMixture


@partial(vmap, in_axes=(0, None, None))
def posterior(y, params, n_features):
    precisions = jnp.linalg.inv(params['cov'])
    log_det_precisions = jnp.log(jnp.linalg.det(precisions))
    norm = 0.5 * (log_det_precisions - n_features * jnp.log(2 * jnp.pi))
    diff_tmp = y - params['mu']
    dot_tmp = vmap(jnp.dot, (0, 0))
    log_prob = dot_tmp(diff_tmp, dot_tmp(precisions, diff_tmp))
    weighted_log_prob = norm - 0.5 * log_prob + jnp.log(params['pi'])
    log_prob_norm = logsumexp(weighted_log_prob)
    log_resp = weighted_log_prob - log_prob_norm
    return jnp.exp(log_resp)


@partial(vmap, in_axes=(0, None, None))
def log_prob(y, params, n_features):
    precisions = jnp.linalg.inv(params['cov'])
    log_det_precisions = jnp.log(jnp.linalg.det(precisions))
    norm = 0.5 * (log_det_precisions - n_features * jnp.log(2 * jnp.pi))
    diff_tmp = y - params['mu']
    dot_tmp = vmap(jnp.dot, (0, 0))
    log_prob = dot_tmp(diff_tmp, dot_tmp(precisions, diff_tmp))
    weighted_log_prob = norm - 0.5 * log_prob + jnp.log(params['pi'])
    return logsumexp(weighted_log_prob)


def gamma(k):
    return (1 - 10e-10) * (k + 2) ** (-6/10)


@partial(vmap, in_axes=(0, 0, None, None))
def update_stat(y, t, stat, gam):
    stat['s0'] = gam * t + (1 - gam) * stat['s0']
    stat['s1'] = gam * jnp.einsum('i,k->ik', t, y) + (1 - gam) * stat['s1']
    yyT = jnp.einsum('i,j->ij', y, y)
    stat['S2'] = gam * jnp.einsum('k,ij->kij', t, yyT) + (1 - gam) * stat['S2']
    return stat


def fill_diagonal(a, val):
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)


def update_params(stat, params, N, k, polyak):
    fact_polyak = 1 / (N - polyak + 1)

    pi = stat['s0'] / stat['s0'].sum()
    params['pi'] = cond(k >= polyak,
                        lambda x: (1 - fact_polyak) * params['pi'] +
                        fact_polyak * x, lambda x: x, pi)

    mu = jnp.einsum('kj,k->kj', stat['s1'], 1/stat['s0'])
    params['mu'] = cond(k >= polyak,
                        lambda x: (1 - fact_polyak) * params['mu'] +
                        fact_polyak * x, lambda x: x, mu)

    cov = jnp.einsum('kij,k->kij', stat['S2'], 1/stat['s0']) - \
        jnp.einsum('ki,kj->kij', params['mu'], params['mu'])
    cov = fill_diagonal(cov, vmap(jnp.diagonal, in_axes=(0))(cov) + 1e-6)
    params['cov'] = cond(k >= polyak,
                         lambda x: (1 - fact_polyak) * params['cov'] +
                         fact_polyak * x, lambda x: x, cov)
    return params


def _initialization(X, n_components, n_first=1000):
    gmm = GaussianMixture(n_components, max_iter=1)
    gmm.fit(X[:n_first])
    params = {}
    params['pi'] = jnp.array(gmm.weights_)
    params['mu'] = jnp.array(gmm.means_)
    params['cov'] = jnp.linalg.inv(gmm.precisions_)
    return params


def _fit(X, X_idx, N, M, polyak, params):
    n_components, n_features = params['mu'].shape
    stat = {}
    stat['s0'] = jnp.zeros(n_components)
    stat['s1'] = jnp.zeros(params['mu'].shape)
    stat['S2'] = jnp.stack([jnp.diag(jnp.ones(n_features))] * n_components)

    @loop_tqdm(2 * M)
    def warmup_step(k, val):
        stat, params = val
        y_idx = jnp.take(X_idx, k, axis=0)
        Y = jnp.take(X, y_idx, axis=0)

        gam = gamma(k)
        t = posterior(Y, params, n_features)
        stat = update_stat(Y, t, stat, gam)
        stat = tree_map(lambda x: x.mean(axis=0), stat)

        return stat, params

    init_val = (stat, params)
    stat, _ = fori_loop(0, 2 * M, warmup_step, init_val)

    @loop_tqdm(N)
    def training_step(k, val):
        stat, params = val
        y_idx = jnp.take(X_idx, k, axis=0)
        Y = jnp.take(X, y_idx, axis=0)

        gam = gamma(k)
        t = posterior(Y, params, n_features)
        stat = update_stat(Y, t, stat, gam)
        stat = tree_map(lambda x: x.mean(axis=0), stat)

        params = update_params(stat, params, N, k, polyak)

        return stat, params

    init_val = (stat, params)
    _, params = fori_loop(0, N, training_step, init_val)

    return params


def fit(X, n_components, M, N, batch_size, polyak):
    params = _initialization(X, n_components)
    X_idx = jnp.array(np.random.randint(len(X), size=(N, batch_size, )))
    params = _fit(X, X_idx, N, M, polyak, params)
    return params


@jit
def predict(X, params):
    _, n_features = params['mu'].shape
    t = posterior(X, params, n_features)
    return jnp.argmax(t, axis=-1)


@jit
def log_like(X, weights, means, covariances):
    _, n_features = means.shape
    return log_prob(X, weights, means, covariances, n_features)


@partial(vmap, in_axes=(0, None, None, 0))
def _weights_gmm(y, means, covariances, t):
    A, D = eigh(covariances)
    delta = A / jnp.einsum('kij,ki->kj', D, y - means) ** 2
    return jnp.einsum('k,ki->i', t, delta).max()


@jit
def weights_gmm(X, weights, means, covariances):
    n_features = X.shape[-1]
    t = posterior(X, weights, means, covariances, n_features)
    return _weights_gmm(X, means, covariances, t)


def BIC(X, weights, means, covariances):
    N = X.shape[0]
    n_components, n_features = means.shape
    L = log_like(X, weights, means, covariances).sum()
    p = n_features * n_components + n_components * (n_components + 1) / 2
    return - 2 * L + p * jnp.log(N)
