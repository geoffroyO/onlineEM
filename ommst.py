"""Online EM for Multiple-Scale t-distribution."""
import copy
import warnings
from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, tree_map, vmap
from jax.lax import cond, fori_loop, while_loop
from jax.scipy.special import digamma, gammaln, logsumexp

from jax_tqdm import loop_tqdm

import numpy as np


from onlineEM.manifold import (beta_polak_ribiere,
                               inner_product,
                               line_search,
                               norm,
                               riemannian_gradient,
                               transport)
from onlineEM.root_finding import brentq

from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")


def gamma(k):
    return (1 - 10e-10) * (k + 1) ** (-6/10)


def mst_log_pdf(y, params):
    th2 = params['A'] * params['nu']
    th1 = jnp.log(1 + jnp.einsum('kji,kj->ki',
                                 params['D'],
                                 y-params['mu']) ** 2 / th2)
    exponent = - (params['nu'] + 1) / 2

    main = exponent * th1

    gam1 = gammaln((params['nu'] + 1) / 2)
    gam2 = gammaln(params['nu'] / 2)
    th2 = gam1 - (gam2 + 0.5 * jnp.log(jnp.pi * th2))

    main += th2

    return main.sum(1)


def mmst_logpdf(y, params):
    return jnp.log(params['pi']) + mst_log_pdf(y, params)


@partial(vmap, in_axes=(0, None))
def posterior(y, params):
    tmp = mmst_logpdf(y, params)
    return jnp.exp(tmp - logsumexp(tmp, axis=0))


def compute_alpha_beta(y, params):
    tmp = params['nu'] / 2
    alpha = tmp + 0.5
    beta = tmp + jnp.einsum('kji,kj->ki',
                            params['D'],
                            y-params['mu']) ** 2 / (2 * params['A'])
    return alpha, beta


def _u(alpha, beta):
    return alpha / beta


def _u_tilde(alpha, beta):
    return digamma(alpha) - jnp.log(beta)


def fill_diagonal(a, val):
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)


@partial(vmap, in_axes=(0, 0, None, None, None))
def update_stat(y, t, params, stat, gam):
    alpha, beta = compute_alpha_beta(y, params)
    u, u_tilde = _u(alpha, beta), _u_tilde(alpha, beta)

    t_u = jnp.einsum('k,kj->kj', t, u)
    t_u_tilde = jnp.einsum('k,kj->kj', t, u_tilde)

    y_mat = jnp.einsum('i,j->ij', y, y)

    stat['s0'] = gam * t + (1 - gam) * stat['s0']
    stat['s1'] = gam * jnp.einsum('ij,k->ijk', t_u, y) + (1 - gam) * stat['s1']
    stat['S2'] = gam * (jnp.einsum('ij,kl->ijkl', t_u, y_mat)) + \
        (1 - gam) * stat['S2']
    stat['s3'] = gam * t_u + (1 - gam) * stat['s3']
    stat['s4'] = gam * t_u_tilde + (1 - gam) * stat['s4']

    return stat


def update_pi(stat):
    return stat['s0'] / stat['s0'].sum()


def update_mu(params, stat):
    v = jnp.einsum('kim,kmi->km', params['D'], stat['s1'])
    tmp = jnp.einsum('kij,kj->kij', params['D'], 1 / stat['s3'])
    mu = jnp.einsum('kij,kj->ki', tmp, v)
    return mu, v


def update_A(v, params, stat):
    tmp = jnp.einsum('kjm,kmji->kmi', params['D'], stat['S2'])
    tmp = jnp.einsum('kmi,kim->km', tmp, params['D'])
    return tmp - v ** 2 / stat['s3'] + 1e-6


@partial(vmap, in_axes=({'s0': None, 's1': None, 'S2': None, 's3': 0,
                         's4': 0}, ))
@partial(vmap, in_axes=({'s0': None, 's1': None, 'S2': None, 's3': 0,
                         's4': 0}, ))
def update_nu(stat):
    solver = brentq(lambda x, s3km, s4km: s4km - s3km -
                    digamma(x / 2) + jnp.log(x / 2) + 1)
    return solver(0.001, 100, (stat['s3'], stat['s4']))


def cost_D(D, stat):
    tmp = jnp.einsum('ij,i->ij', stat['s1'], 1/stat['s3'])
    tmp = stat['S2'] - jnp.einsum('mi,mj->mij', stat['s1'], tmp)
    tmp = jnp.einsum('mij,jm->mi', tmp, D)
    cost = jnp.einsum('mi,im->', tmp, D)
    return cost


grad_cost_D = grad(cost_D)


@partial(vmap, in_axes=({'pi': None, 'D': 0, 'A': None,
                         'mu': None, 'nu': None},
                        {'s0': None, 's1': 0, 'S2': 0, 's3': 0,
                        's4': None}))
def update_D(params, stat):
    x = params['D']
    _cost_D = partial(cost_D, stat=stat)
    _grad_cost_D = partial(grad_cost_D, stat=stat)

    cost = _cost_D(x)
    grad = riemannian_gradient(x, _grad_cost_D(x))
    gradient_norm = norm(grad)
    Pgrad = grad
    gradPgrad = inner_product(grad, Pgrad)
    oldalpha = -1
    descent_direction = -Pgrad

    cost_evaluations = 0

    def body(val):
        x, cost, grad, Pgrad, gradient_norm, gradPgrad, \
                descent_direction, oldalpha, cost_evaluations = val
        df0 = inner_product(grad, descent_direction)
        descent_direction = jnp.where(df0 >= 0, -Pgrad, descent_direction)
        df0 = jnp.where(df0 >= 0, -gradPgrad, df0)

        newx, oldalpha = line_search(_cost_D, x, descent_direction, cost,
                                     df0, oldalpha)

        newcost = _cost_D(newx)
        newgrad = riemannian_gradient(newx, _grad_cost_D(newx))
        newgradient_norm = norm(newgrad)
        Pnewgrad = newgrad
        newgradPnewgrad = inner_product(newgrad, Pnewgrad)

        oldgrad = transport(newx, grad)

        descent_direction = transport(newx, descent_direction)
        beta = beta_polak_ribiere(newgrad, Pnewgrad, gradPgrad, oldgrad)
        descent_direction = -Pnewgrad + beta * descent_direction

        x = newx
        cost = newcost
        grad = newgrad
        Pgrad = Pnewgrad
        gradient_norm = newgradient_norm
        gradPgrad = newgradPnewgrad
        cost_evaluations += 1

        return (x, cost, grad, Pgrad,
                gradient_norm, gradPgrad,
                descent_direction, oldalpha,
                cost_evaluations)

    def cond(val):
        _, _, _, _, gradient_norm, _, _, _, cost_evaluations = val
        return (gradient_norm >= 1e-6) * (cost_evaluations <= 350)

    val_init = (x, cost, grad, Pgrad, gradient_norm, gradPgrad,
                descent_direction, oldalpha, cost_evaluations)
    x, _, _, _, gradient_norm, _, _, _, _ = while_loop(cond, body, val_init)
    return x


def update_params(params, stat):
    stat_updt = {}
    stat_updt['s0'] = stat['s0']
    stat_updt['s1'] = jnp.einsum('kij,k->kij', stat['s1'], 1/stat['s0'])
    stat_updt['S2'] = jnp.einsum('kmij,k->kmij', stat['S2'], 1/stat['s0'])
    stat_updt['s3'] = jnp.einsum('ki,k->ki', stat['s3'], 1/stat['s0'])
    stat_updt['s4'] = jnp.einsum('ki,k->ki', stat['s4'], 1/stat['s0'])

    params['pi'] = update_pi(stat_updt)
    params['D'] = update_D(params, stat_updt)
    params['mu'], v = update_mu(params, stat_updt)
    params['A'] = update_A(v, params, stat_updt)
    params['nu'] = update_nu(stat_updt)

    return params


def _initialization(X, n_components, n_first=1000):
    gmm = GaussianMixture(n_components, max_iter=5)
    gmm.fit(X[:n_first])
    pi = jnp.array(gmm.weights_).astype(jnp.float64)
    mu = jnp.array(gmm.means_).astype(jnp.float64)
    covariances = gmm.covariances_
    A, D = vmap(jnp.linalg.eig, in_axes=(0))(covariances)
    A = A.astype(jnp.float64) + 1e-6
    D = D.astype(jnp.float64)
    nu = jnp.full(A.shape, 10.).astype(jnp.float64)

    params = {'pi': pi, 'mu': mu, 'A': A, 'D': D, 'nu': nu}
    params_polyak = tree_map(lambda x: copy.deepcopy(x), params)
    return params, params_polyak


def _fit(X, X_idx, M, N, polyak, params, params_polyak):
    stat = {}
    stat['s0'] = jnp.zeros(params['pi'].shape)
    stat['s1'] = jnp.zeros(params['D'].shape)
    stat['S2'] = jnp.zeros((*params['D'].shape, params['mu'].shape[-1]))
    stat['s3'], stat['s4'] = jnp.zeros(params['A'].shape), \
        jnp.zeros(params['A'].shape)

    # Warm-up
    @loop_tqdm(2 * M)
    def warmup_step(k, val):
        params, stat = val
        y_idx = jnp.take(X_idx, k, axis=0)
        Y = jnp.take(X, y_idx, axis=0)

        # Update statistics
        gam = gamma(k)
        t = posterior(Y, params)
        stat = tree_map(lambda s: s.mean(axis=0),
                            update_stat(Y, t, params, stat, gam))
        return params, stat
    init_val = (params, stat)
    _, stat = fori_loop(0, 2 * M, warmup_step, init_val)

    # Training
    @loop_tqdm(N)
    def training_step(k, val):
        params, params_polyak, stat = val
        y_idx = jnp.take(X_idx, k, axis=0)
        Y = jnp.take(X, y_idx, axis=0)

        # Update statistics
        gam = gamma(k)
        t = posterior(Y, params)
        stat = tree_map(lambda s: s.mean(axis=0),
                            update_stat(Y, t, params, stat, gam))

        # Update parameters
        params = update_params(params, stat)
        params_polyak = cond(k > polyak,
                             lambda X, Y: tree_map(lambda x, y: x+y, X, Y),
                             lambda X, _: tree_map(lambda x: copy.deepcopy(x), X),
                             params, params_polyak)
        return params, params_polyak, stat

    val = (params, params_polyak,  stat)
    _, params_polyak, _ = fori_loop(0, N, training_step, val)
    params_polyak = tree_map(lambda x: x / (N - polyak + 1), params_polyak)
    return params_polyak


def fit(X, n_components, M, N, batch_size, polyak):
    params, params_polyak = _initialization(X, n_components)
    X_idx = jnp.array(np.random.randint(len(X), size=(N, batch_size, )))
    params_polyak = _fit(X, X_idx, M, N, polyak, params, params_polyak)
    return params_polyak


@jit
def predict(X, params):
    t = posterior(X, params)
    return jnp.argmax(t, axis=-1)


@jit
@partial(vmap, in_axes=(0, None))
def log_like(X, params):
    return logsumexp(mmst_logpdf(X, params['pi'],
                                 params['mu'],
                                 params['A'],
                                 params['D'],
                                 params['nu']))


@jit
@partial(vmap, in_axes=(0, None))
def weights_mmst(X, params):
    alpha, beta = compute_alpha_beta(X, params['mu'],
                                     params['A'],
                                     params['D'],
                                     params['nu'])
    u = _u(alpha, beta)
    tmp = mmst_logpdf(X, params['pi'], params['mu'],
                      params['A'], params['D'], params['nu'])
    t = jnp.exp(tmp - logsumexp(tmp, axis=0))
    w = jnp.einsum('k,ki->i', t, u)
    return jnp.max(w)


def BIC(X, params):
    N = X.shape[0]
    n_components, n_features = params['mu'].shape
    L = log_like(X, params['pi'], params['mu'],
                 params['A'], params['D'], params['nu']).sum()
    p = n_components * (1 + (n_features * (n_features + 5)) / 2) - 1
    return - 2 * L + p * jnp.log(N)
