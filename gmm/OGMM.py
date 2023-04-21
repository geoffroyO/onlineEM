import jax.numpy as jnp
from jax import vmap, jit
from jax.scipy.special import logsumexp

from jax.lax import fori_loop 

from sklearn.mixture import GaussianMixture

from functools import partial

@partial(vmap, in_axes=(0, None, None, None, None))
def posterior(y, pi, means, covariances, n_features):
    precisions, log_det_precisions = jnp.linalg.inv(covariances), jnp.log(jnp.linalg.det(covariances))
    norm = 0.5 * (log_det_precisions - n_features * jnp.log(2 * jnp.pi))
    diff_tmp = y - means
    dot_tmp = vmap(jnp.dot, (0, 0))
    log_prob = dot_tmp(diff_tmp, dot_tmp(precisions, diff_tmp))
    weighted_log_prob = norm - 0.5 * log_prob + jnp.log(pi)
    log_prob_norm = logsumexp(weighted_log_prob)
    log_resp = weighted_log_prob - log_prob_norm
    return jnp.exp(log_resp)

@partial(vmap, in_axes=(0, None, None, None, None))
def log_prob(y, pi, means, covariances, n_features):
    precisions, log_det_precisions = jnp.linalg.inv(covariances), jnp.log(jnp.linalg.det(covariances))
    norm = 0.5 * (log_det_precisions - n_features * jnp.log(2 * jnp.pi))
    diff_tmp = y - means
    dot_tmp = vmap(jnp.dot, (0, 0))
    log_prob = dot_tmp(diff_tmp, dot_tmp(precisions, diff_tmp))
    weighted_log_prob = norm - 0.5 * log_prob + jnp.log(pi)
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

def update_params(s0, s1, S2):
    pi = s0 / s0.sum()
    mu = s1 / s0[:, jnp.newaxis]
    sigma = S2 / s0[:, jnp.newaxis, jnp.newaxis] - jnp.einsum('ki,kj->kij', mu, mu)
    return pi, mu, sigma

def _initialization(X, n_components, batch_size, n_first=1000):
    gmm = GaussianMixture(n_components, max_iter=1)
    gmm.fit(X[:n_first])
    pi = jnp.array(gmm.weights_)
    mu = jnp.array(gmm.means_)
    sigma = jnp.linalg.inv(gmm.precisions_)
    X_batch = batch_data(X, batch_size)
    return X_batch, pi, mu, sigma
@jit
def _fit(X_batch, pi, mu, sigma):
    N = X_batch.shape[0]
    n_components, n_features = mu.shape

    s0 = jnp.zeros(n_components)
    s1 = jnp.zeros(mu.shape)
    S2 = jnp.stack([jnp.diag(jnp.ones(n_features))] * n_components)

    # Warm-up
    def warmup_step(k, val):
        X, s0, s1, S2, pi, mu, sigma, n_features = val
        y = jnp.take(X, k, axis=0)

        # Update statistics
        gam = gamma(k)
        t = posterior(y, pi, mu, sigma, n_features)
        s0, s1, S2 = update_stat(y, t, s0, s1, S2, gam)
        s0, s1, S2 = s0.mean(axis=0), s1.mean(axis=0), S2.mean(axis=0)

        return X, s0, s1, S2, pi, mu, sigma, n_features

    init_val = (X_batch, s0, s1, S2, pi, mu, sigma, n_features)
    _, s0, s1, S2, _, _, _, _ = fori_loop(0, 200, warmup_step, init_val)

    # Training
    def training_step(k, val):
        X, s0, s1, S2, pi, mu, sigma, n_features = val
        y = jnp.take(X, k, axis=0)

        # Update statistics
        gam = gamma(k)
        t = posterior(y, pi, mu, sigma, n_features)
        s0, s1, S2 = update_stat(y, t, s0, s1, S2, gam)
        s0, s1, S2 = s0.mean(axis=0), s1.mean(axis=0), S2.mean(axis=0)

        # Update parameters
        pi, mu, sigma = update_params(s0, s1, S2)

        return X, s0, s1, S2, pi, mu, sigma, n_features

    init_val = (X_batch, s0, s1, S2, pi, mu, sigma, n_features)
    _, s0, s1, S2, pi, mu, sigma, _ = fori_loop(200, N, training_step, init_val)

    return pi, mu, sigma

@jit
def predict(X, pi, mu, sigma):
    _, n_features = mu.shape
    t = posterior(X, pi, mu, sigma, n_features)
    return jnp.argmax(t, axis=-1)

@jit
def log_like(X, pi, mu, sigma):
    _, n_features = mu.shape
    return log_prob(X, pi, mu, sigma, n_features)

def fit(X, n_components, batch_size):
    X_batch, pi, mu, sigma = _initialization(X, n_components, batch_size)
    return _fit(X_batch, pi, mu, sigma)
