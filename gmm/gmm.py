from jax import vmap
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def _estimate_log_gaussian_prob(y, means, precisions, log_det_precisions):
    n_features = len(y)

    norm = 0.5 * (log_det_precisions - n_features * jnp.log(2 * jnp.pi))
    diff_tmp = y - means
    dot_tmp = vmap(jnp.dot, (0, 0))
    log_prob = dot_tmp(diff_tmp, dot_tmp(precisions, diff_tmp))
    return norm - 0.5 * log_prob


class GMM:
    def __init__(self, weights, means, precisions, log_det_precisions):
        self.weights_ = weights
        self.means_ = means
        self.precisions_ = precisions
        self.log_det_precisions_ = log_det_precisions

    def _estimate_log_prob(self, y):
        return _estimate_log_gaussian_prob(y, self.means_, self.precisions_, self.log_det_precisions_)

    def _estimate_log_weights(self):
        return jnp.log(self.weights_)

    def _estimate_weighted_log_prob(self, y):
        return self._estimate_log_prob(y) + self._estimate_log_weights()

    def _estimate_log_prob_resp(self, y):
        weighted_log_prob = self._estimate_weighted_log_prob(y)
        log_prob_norm = logsumexp(weighted_log_prob)
        log_resp = weighted_log_prob - log_prob_norm
        return log_prob_norm, log_resp

    def posterior(self, y):
        _, log_resp = self._estimate_log_prob_resp(y)
        return jnp.exp(log_resp)

    def logpdf(self, y):
        log_prob_norm, _ = self._estimate_log_prob_resp(y)
        return log_prob_norm


def posterior(y, pi, means, precisions, log_det_precisions, n_features):
    norm = 0.5 * (log_det_precisions - n_features * jnp.log(2 * jnp.pi))
    diff_tmp = y - means
    dot_tmp = vmap(jnp.dot, (0, 0))
    log_prob = dot_tmp(diff_tmp, dot_tmp(precisions, diff_tmp))
    weighted_log_prob = norm - 0.5 * log_prob + jnp.log(pi)
    log_prob_norm = logsumexp(weighted_log_prob)
    log_resp = weighted_log_prob - log_prob_norm

    return jnp.exp(log_resp)

