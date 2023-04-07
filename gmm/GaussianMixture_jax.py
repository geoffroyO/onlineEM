from gmm.gmm import GMM
from gmm.gmm import posterior
from jax import vmap, jit
import jax.numpy as jnp
from jax.lax import fori_loop

from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import numpy as np


def batch_data(X, batch_size):
    N, _ = X.shape
    X_batch = []
    for k in range(N // batch_size):
        X_batch.append(jnp.array(X[k * batch_size:(k + 1) * batch_size]))
    if (N // batch_size) * batch_size != len(X):
        X_batch.append(jnp.array(X[(N // batch_size) * batch_size:]))
    return X_batch


@jit
def update_s01(y, t, s0, s1, gam):
    s0 = gam * t + (1 - gam) * s0
    s1 = gam * jnp.einsum('i,k->ik', t, y) + (1 - gam) * s1
    return s0, s1


@jit
def init_S2(y, t, inv_S2, log_det_inv_S2, gam, batch_size):
    S2y = jnp.einsum('kij,j->ki', inv_S2, y)
    square_S2y = jnp.einsum('ki,kj->kij', S2y, S2y)

    t_square_S2y = jnp.einsum('k,kij->kij', t, square_S2y)
    yS2y = jnp.einsum('j,kj->k', y, S2y)
    t_yS2y = t * yS2y

    tmp_den = 1 + gam * t_yS2y / (batch_size * (1 - gam))
    tmp_num = gam * t_square_S2y / (batch_size * (1 - gam) ** 2)
    tmp = tmp_num / tmp_den[:, jnp.newaxis, jnp.newaxis]

    inv_S2 = inv_S2 / (1 - gam) - tmp

    tmp = -jnp.log(1 + gam * t_yS2y / (batch_size * (1 - gam)))
    tmp -= len(y) * jnp.log(1 - gam)
    log_det_inv_S2 = tmp + log_det_inv_S2

    return inv_S2, log_det_inv_S2


@jit
def update_S2(i, val):
    y_batch, t_batch, inv_S2, log_det_inv_S2, gam, batch_size = val
    y, t = jnp.take(y_batch, i, axis=0), jnp.take(t_batch, i, axis=0)
    S2y = jnp.einsum('kij,j->ki', inv_S2, y)
    square_S2y = jnp.einsum('ki,kj->kij', S2y, S2y)

    t_square_S2y = jnp.einsum('k,kij->kij', t, square_S2y)
    yS2y = jnp.einsum('j,kj->k', y, S2y)
    t_yS2y = t * yS2y

    tmp_den = 1 + gam * t_yS2y / batch_size
    tmp_num = gam * t_square_S2y / batch_size
    tmp = tmp_num / tmp_den[:, jnp.newaxis, jnp.newaxis]

    inv_S2 = inv_S2 - tmp

    tmp = -jnp.log(1 + gam * t_yS2y / batch_size)
    log_det_inv_S2 = tmp + log_det_inv_S2

    return y_batch, t_batch, inv_S2, log_det_inv_S2, gam, batch_size


@jit
def _update_pi(s0):
    return s0 / s0.sum()


@jit
def _update_mu(s0, s1):
    return s1 / s0[:, jnp.newaxis]


@jit
def _update_prec(s0, s1, inv_S2, log_det_inv_S2):
    S2s1 = jnp.einsum('kij,kj->ki', inv_S2, s1)
    square_S2s1 = jnp.einsum('ki,kj->kij', S2s1, S2s1)

    s1S2s1 = jnp.einsum('kj,kj->k', s1, S2s1)
    s0_s1S2s1 = 1 - s1S2s1 / s0

    tmp = square_S2s1 / s0_s1S2s1[:, jnp.newaxis, jnp.newaxis]

    s0S2 = jnp.einsum('k,kij->kij', s0, inv_S2)
    prec = s0S2 + tmp

    log_det_prec = inv_S2.shape[1] * jnp.log(s0)
    log_det_prec -= jnp.log(s0_s1S2s1)
    log_det_prec += log_det_inv_S2

    return prec, log_det_prec


class GMMOEM:
    def __init__(self, n_components, random_state=42):
        self.n_components = n_components
        self.pi, self.mu, self.prec, self.log_det_prec = None, None, None, None
        self.s0, self.s1, self.inv_S2, self.log_det_inv_S2 = None, None, None, None

        self.random_state = random_state

    def _initialization(self, X, n_first=1000):
        gmm = GaussianMixture(self.n_components, max_iter=1, random_state=self.random_state)
        gmm.fit(X[:n_first])

        self.pi = jnp.array(gmm.weights_)
        self.mu = jnp.array(gmm.means_)
        self.prec = jnp.array(gmm.precisions_)
        self.log_det_prec = jnp.log(jnp.linalg.det(self.prec))

    def fit(self, X, batch_size):
        self._initialization(X)
        gamma = iter(
            (1 - 10e-10) * np.array([k for k in range(2, 10 * 2 * (len(X) // batch_size + 1) + 2)]) ** (-6 / 10))
        n_comp, n_features = self.mu.shape

        X_batch = batch_data(X, batch_size)
        del X

        self.s0 = jnp.zeros(n_comp)
        self.s1 = jnp.zeros(self.mu.shape)
        self.inv_S2 = jnp.stack([jnp.diag(jnp.ones(n_features))] * n_comp)
        self.log_det_inv_S2 = jnp.log(jnp.linalg.det(self.inv_S2))

        # vmap
        _update_s01 = vmap(update_s01, in_axes=(0, 0, None, None, None))

        # Warm-up
        # posterior = jit(vmap(GMM(self.pi, self.mu, self.prec, self.log_det_prec).posterior))
        # posterior(y, pi, means, precisions, log_det_precisions, n_features)
        jit_posterior = jit(vmap(posterior, in_axes=(0, None, None, None, None, None)))
        print('Warm-up...')
        for batch in tqdm(X_batch[:2]):
            gam = next(gamma)
            t = jit_posterior(batch, self.pi, self.mu, self.prec, self.log_det_prec, n_features)
            s0, s1 = _update_s01(batch, t, self.s0, self.s1, gam)
            self.s0 = s0.mean(axis=0)
            self.s1 = s1.mean(axis=0)
            inv_S2, log_det_inv_S2 = init_S2(jnp.take(batch, 0, axis=0), jnp.take(t, 0, axis=0), self.inv_S2,
                                             self.log_det_inv_S2, gam, batch_size)
            lower, upper = 1, batch_size
            val = (batch, t, inv_S2, log_det_inv_S2, gam, batch_size)
            _, _, self.inv_S2, self.log_det_inv_S2, _, _ = fori_loop(lower, upper, update_S2, val)

        print('Training...')
        for batch in tqdm(X_batch[2:]):
            # posterior = jit(vmap(GMM(self.pi, self.mu, self.prec, self.log_det_prec).posterior))
            t = jit_posterior(batch, self.pi, self.mu, self.prec, self.log_det_prec, n_features)
            gam = next(gamma)

            s0, s1 = _update_s01(batch, t, self.s0, self.s1, gam)
            self.s0 = s0.mean(axis=0)
            self.s1 = s1.mean(axis=0)
            inv_S2, log_det_inv_S2 = init_S2(jnp.take(batch, 0, axis=0), jnp.take(t, 0, axis=0), self.inv_S2,
                                             self.log_det_inv_S2, gam, batch_size)
            lower, upper = 1, batch_size
            val = (batch, t, inv_S2, log_det_inv_S2, gam, batch_size)
            _, _, self.inv_S2, self.log_det_inv_S2, _, _ = fori_loop(lower, upper, update_S2, val)

            self.pi = _update_pi(self.s0)
            self.mu = _update_mu(self.s0, self.s1)
            self.inv_S2, self.log_det_inv_S2 = _update_prec(self.s0, self.s1, self.inv_S2, self.log_det_inv_S2)

    def predict(self, X):
        posterior = jit(vmap(GMM(self.pi, self.mu, self.prec, self.log_det_prec).posterior))
        t = posterior(X)
        cluster_lab = jnp.argmax(t, axis=-1)
        return cluster_lab

    def log_like(self, X):
        logpdf = jit(vmap(GMM(self.pi, self.mu, self.prec, self.log_det_prec).logpdf))
        res = logpdf(X)
        return res.sum()

    def bic(self, X):
        log_like = self.log_like(X)
        n, p = X.shape
        return -2 * log_like + self.n_components * (1 + p * (1 + (p + 1) / 2)) * np.log(n)
