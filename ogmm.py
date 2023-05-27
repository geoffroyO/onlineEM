import jax.numpy as jnp
from jax import vmap, jit
from jax.scipy.special import logsumexp
from jax.scipy.linalg import eigh
from jax.lax import fori_loop 

from sklearn.mixture import GaussianMixture

from functools import partial

@partial(vmap, in_axes=(0, None, None, None, None))
def posterior(y, weights, means, covariances, n_features):
    precisions = jnp.linalg.inv(covariances)
    log_det_precisions = jnp.log(jnp.linalg.det(precisions))
    norm = 0.5 * (log_det_precisions - n_features * jnp.log(2 * jnp.pi))
    diff_tmp = y - means
    dot_tmp = vmap(jnp.dot, (0, 0))
    log_prob = dot_tmp(diff_tmp, dot_tmp(precisions, diff_tmp))
    weighted_log_prob = norm - 0.5 * log_prob + jnp.log(weights)
    log_prob_norm = logsumexp(weighted_log_prob)
    log_resp = weighted_log_prob - log_prob_norm
    return jnp.exp(log_resp)

@partial(vmap, in_axes=(0, None, None, None, None))
def log_prob(y, weights, means, covariances, n_features):
    precisions = jnp.linalg.inv(covariances)
    log_det_precisions = jnp.log(jnp.linalg.det(precisions))
    norm = 0.5 * (log_det_precisions - n_features * jnp.log(2 * jnp.pi))
    diff_tmp = y - means
    dot_tmp = vmap(jnp.dot, (0, 0))
    log_prob = dot_tmp(diff_tmp, dot_tmp(precisions, diff_tmp))
    weighted_log_prob = norm - 0.5 * log_prob + jnp.log(weights)
    return logsumexp(weighted_log_prob)

def batch_data(X, batch_size):
    N, _ = X.shape
    X_batch = []
    for k in range(N // batch_size):
        X_batch.append(jnp.array(X[k * batch_size:(k + 1) * batch_size]))
    return jnp.array(X_batch)

def gamma(k):
    return (1 - 10e-10) * (k + 2) **(-6/10)

@partial(vmap, in_axes=(0, 0, None, None, None, None))
def update_stat(y, t, s0, s1, S2, gam):
    s0 = gam * t + (1 - gam) * s0
    s1 = gam * jnp.einsum('i,k->ik', t, y) + (1 - gam) * s1
    yyT = y.reshape(-1, 1) @ y.reshape(1, -1)
    S2 = gam * jnp.einsum('k,ij->kij', t, yyT) + (1 - gam) * S2
    return s0, s1, S2

def fill_diagonal(a, val):
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)

def update_params(s0, s1, S2):
    weights = s0 / s0.sum()
    means = s1 / s0[:, jnp.newaxis]
    covariances = S2 / s0[:, jnp.newaxis, jnp.newaxis] - jnp.einsum('ki,kj->kij', means, means)
    covariances = fill_diagonal(covariances, vmap(jnp.diagonal, in_axes=(0))(covariances) + 1e-6)
    return weights, means, covariances

def _initialization(X, n_components, batch_size, n_first=1000):
    gmm = GaussianMixture(n_components, max_iter=1)
    gmm.fit(X[:n_first])
    weights = jnp.array(gmm.weights_)
    means = jnp.array(gmm.means_)
    covariances = jnp.linalg.inv(gmm.precisions_)
    X_batch = batch_data(X, batch_size)
    return X_batch, weights, means, covariances

@jit
def _fit(X_batch, weights, means, covariances):
    N = X_batch.shape[0]
    n_components, n_features = means.shape

    s0 = jnp.zeros(n_components)
    s1 = jnp.zeros(means.shape)
    S2 = jnp.stack([jnp.diag(jnp.ones(n_features))] * n_components)

    # Warm-up
    def warmup_step(k, val):
        X, s0, s1, S2, weights, means, covariances, n_features = val
        y = jnp.take(X, k, axis=0)

        # Update statistics
        gam = gamma(k)
        t = posterior(y, weights, means, covariances, n_features)
        s0, s1, S2 = update_stat(y, t, s0, s1, S2, gam)
        s0, s1, S2 = s0.mean(axis=0), s1.mean(axis=0), S2.mean(axis=0)

        return X, s0, s1, S2, weights, means, covariances, n_features

    init_val = (X_batch, s0, s1, S2, weights, means, covariances, n_features)
    _, s0, s1, S2, _, _, _, _ = fori_loop(0, 200, warmup_step, init_val)
    
    # Training
    def training_step(k, val):
        X, s0, s1, S2, weights, means, covariances, n_features = val
        y = jnp.take(X, k, axis=0)

        # Update statistics
        gam = gamma(k)
        t = posterior(y, weights, means, covariances, n_features)
        s0, s1, S2 = update_stat(y, t, s0, s1, S2, gam)
        s0, s1, S2 = s0.mean(axis=0), s1.mean(axis=0), S2.mean(axis=0)

        # Update parameters
        weights, means, covariances = update_params(s0, s1, S2)

        return X, s0, s1, S2, weights, means, covariances, n_features

    init_val = (X_batch, s0, s1, S2, weights, means, covariances, n_features)
    _, s0, s1, S2, weights, means, covariances, _ = fori_loop(200, N, training_step, init_val)

    return weights, means, covariances

@jit
def predict(X, weights, means, covariances):
    _, n_features = means.shape
    t = posterior(X, weights, means, covariances, n_features)
    return jnp.argmax(t, axis=-1)

@jit
def log_like(X, weights, means, covariances):
    _, n_features = means.shape
    return log_prob(X, weights, means, covariances, n_features)

@partial(vmap, in_axes=(0, None, None, 0))
def _weights_gmm(y, means, covariances, t):
    A, D = eigh(covariances)
    delta =  A / jnp.einsum('kij,ki->kj', D, y - means) ** 2
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

def fit(X, n_components, batch_size):
    X_batch, weights, means, covariances = _initialization(X, n_components, batch_size)
    weights, means, covariances = _fit(X_batch, weights, means, covariances)
    return {'weights': weights, 'means': means, 'covariances': covariances}
