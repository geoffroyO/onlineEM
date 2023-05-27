import warnings
warnings.filterwarnings("ignore")

import jax
from jax import vmap, jit
from jax.scipy.special import digamma, gammaln, logsumexp
import jax.numpy as jnp
from jax.lax import fori_loop, while_loop

from manifold import norm, inner_product, transport, beta_polak_ribiere, riemannian_gradient, line_search

from root_finding import brentq

from sklearn.mixture import GaussianMixture

from functools import partial

def batch_data(X, batch_size):
    N, _ = X.shape
    X_batch = []
    for k in range(N // batch_size):
        X_batch.append(jnp.array(X[k * batch_size:(k + 1) * batch_size]))
    return jnp.array(X_batch)

def gamma(k):
    return (1 - 10e-10) * (k + 1) **(-6/10)

def mst_log_pdf(y, mu, A, D, nu):
    th2 = A * nu
    th1 = jnp.log(1 + jnp.einsum('kji,kj->ki', D, y-mu) ** 2 / th2)
    exponent = - (nu + 1) / 2

    main = exponent * th1

    gam1 = gammaln((nu + 1) / 2)
    gam2 = gammaln(nu / 2)
    th2 = gam1 - (gam2 + 0.5 * jnp.log(jnp.pi * th2))

    main += th2

    return main.sum(1)

def mmst_logpdf(y, pi, mu, A, D, nu):
    return jnp.log(pi) + mst_log_pdf(y, mu, A, D, nu)

@partial(vmap, in_axes=(0, None, None, None, None, None))
def posterior(y, pi, mu, A, D, nu):
    tmp = mmst_logpdf(y, pi, mu, A, D, nu)
    return jnp.exp(tmp - logsumexp(tmp, axis=0))


def compute_alpha_beta(y, mu, A, D, nu):
    tmp = nu / 2
    alpha = tmp + 0.5
    beta = tmp + jnp.einsum('kji,kj->ki', D, y-mu) ** 2 / (2 * A)
    return alpha, beta

def _u(alpha, beta):
    return alpha / beta

def _u_tilde(alpha, beta):
    return digamma(alpha) - jnp.log(beta)

@partial(vmap, in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None))
def update_stat(y, t, mu, A, D, nu, s0, s1, S2, s3, s4, gam):
    alpha, beta = compute_alpha_beta(y, mu, A, D, nu)
    u, u_tilde = _u(alpha, beta), _u_tilde(alpha, beta)

    t_u = jnp.einsum('k,kj->kj', t, u)
    t_u_tilde = jnp.einsum('k,kj->kj', t, u_tilde)

    y_mat = jnp.einsum('i,j->ij', y, y)

    s0 = gam * t + (1 - gam) * s0
    s1 = gam * jnp.einsum('ij,k->ijk', t_u, y) + (1 - gam) * s1
    S2 = gam * jnp.einsum('ij,kl->ijkl', t_u, y_mat) + (1 - gam) * S2
    s3 = gam * t_u + (1 - gam) * s3
    s4 = gam * t_u_tilde + (1 - gam) * s4

    return s0, s1, S2, s3, s4

def update_pi(s0):
    return s0 / s0.sum()

def update_mu(D, s1, s3):
    vmap_diag = vmap(jnp.diag, in_axes=(0))

    S3_inv = vmap_diag(1 / s3)
    v = jnp.diagonal(jnp.einsum('kij,kni->kjn', D, s1), 0, -2, -1)

    tmp = jnp.einsum('kji,kj->ki', S3_inv, v)
    mu = jnp.einsum('kij,kj->ki', D, tmp)
    return mu, v

def update_A(v, D, S2, s3):
    tmp = jnp.swapaxes(D[:, None, ...], -2, -1) @ S2
    tmp = tmp @ D[:, None, ...]
    tmp = jnp.diagonal(tmp, 0, -2, -1)
    return jnp.diagonal(tmp, 0, -2, -1) - v ** 2 / s3

def update_nu(s3, s4):
    solver = brentq(lambda x, s3km, s4km: s4km - s3km - digamma(x / 2) + jnp.log(x / 2) + 1)
    return solver(0.001, 100, (s3, s4))

def _compute_matQuad(s1, S2, s3):
        tmp = jnp.einsum('kij,ki->kij', s1, 1/s3)
        return S2 - jnp.einsum('kmi,kmj->kmij', s1, tmp)  

@partial(vmap, in_axes=(1, 0))
def quadForm(dk, mat_quad_km):
    return dk.T @ mat_quad_km @ dk

@partial(vmap, in_axes=(1, 0))
def _num_grad(dk, mat_quad_km):
    return 2 * mat_quad_km @ dk 

def find_cost(mat_quad_k):
    @jit
    def cost(Dk):
        return jnp.log(quadForm(Dk, mat_quad_k)).sum()  
    @jit
    def grad(Dk):  
        return jnp.einsum('ki,k->ik',_num_grad(Dk, mat_quad_k), 1 / quadForm(Dk, mat_quad_k))
    return cost, grad

def update_D(x, mat_quad_k):
    objective, egrad = find_cost(mat_quad_k)
    cost = objective(x)
    grad = riemannian_gradient(x, egrad(x))
    gradient_norm = norm(grad)
    Pgrad = grad
    gradPgrad = inner_product(grad, Pgrad)
    oldalpha = -1
    descent_direction = -Pgrad
    
    cost_evaluations = 0
    def body(val):
        x, cost, grad, Pgrad, gradient_norm, gradPgrad, descent_direction, oldalpha, cost_evaluations = val
        df0 = inner_product(grad, descent_direction)
        descent_direction = jnp.where(df0 >= 0, -Pgrad, descent_direction)
        df0 = jnp.where(df0 >= 0, -gradPgrad, df0)
        
        newx, oldalpha = line_search(objective, x, descent_direction, cost, df0, oldalpha)
        
        newcost = objective(newx)
        newgrad = riemannian_gradient(newx, egrad(newx))
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
        
        return x, cost, grad, Pgrad, gradient_norm, gradPgrad, descent_direction, oldalpha, cost_evaluations
    
    def cond(val):
        _, _, _, _, gradient_norm, _, _, _, cost_evaluations = val
        return (gradient_norm >= 1e-6) * (cost_evaluations <= 350) 
    
    val_init = (x, cost, grad, Pgrad, gradient_norm, gradPgrad, descent_direction, oldalpha, cost_evaluations)
    x, _, _, _, gradient_norm, _, _, _, _ = while_loop(cond, body, val_init)
    return x

def update_params(s0, s1, S2, s3, s4, D):
    s10 = jnp.einsum('kij,k->kij', s1, 1/s0)
    S20 = jnp.einsum('kmij,k->kmij', S2, 1/s0)
    s30 = jnp.einsum('ki,k->ki', s3, 1/s0)
    s40 = jnp.einsum('ki,k->ki', s4, 1/s0)


    pi = update_pi(s0)
    mat_quad = _compute_matQuad(s10, S20, s30)
    n_comp = mat_quad.shape[0]
    
    def body_D(k, D):
        tmp = update_D(D[k], mat_quad[k])
        D = D.at[k].set(tmp)
        return D
    D = fori_loop(0, n_comp, body_D, D)

    mu, v = update_mu(D, s10, s30)
    A = update_A(v, D, S20, s30)
    
    nu = vmap(vmap(update_nu, in_axes=(0, 0)), in_axes=(0, 0))(s30, s40)
    return pi, mu, A, D, nu

def _initialization(X, n_components, batch_size, n_first=1000):
    gmm = GaussianMixture(n_components, max_iter=5)
    gmm.fit(X[:n_first])
    pi = jnp.array(gmm.weights_)
    mu = jnp.array(gmm.means_)
    covariances = gmm.covariances_
    A, D = vmap(jnp.linalg.eig, in_axes=(0))(covariances)
    A = A.astype(jnp.float32)
    D = D.astype(jnp.float32)
    nu = jnp.full(A.shape, 10.)
    X_batch = batch_data(X, batch_size)
    return X_batch, pi, mu, A, D, nu

@jit 
def _fit(X_batch, pi, mu, A, D, nu):
    N = X_batch.shape[0]

    s0 = jnp.zeros(pi.shape)
    s1 = jnp.zeros(D.shape)
    S2 = jnp.zeros((*D.shape, mu.shape[-1]))
    s3, s4 = jnp.zeros(A.shape), jnp.zeros(A.shape)

    # Warm-up
    def warmup_step(k, val):
        X, s0, s1, S2, s3, s4,  pi, mu, A, D, nu = val
        y = jnp.take(X, k, axis=0)

        # Update statistics
        gam = gamma(k)
        t = posterior(y, pi, mu, A, D, nu)
        s0, s1, S2, s3, s4 = update_stat(y, t, mu, A, D, nu, s0, s1, S2, s3, s4, gam)
        s0, s1, S2, s3, s4 = s0.mean(axis=0), s1.mean(axis=0), S2.mean(axis=0), s3.mean(axis=0), s4.mean(axis=0)
        return X, s0, s1, S2, s3, s4,  pi, mu, A, D, nu
    
    init_val = (X_batch, s0, s1, S2, s3, s4,  pi, mu, A, D, nu)
    _, s0, s1, S2, s3, s4, _, _, _, _ , _ = fori_loop(0, 200, warmup_step, init_val)
    
    # Training
    def training_step(k, val):
        X, s0, s1, S2, s3, s4,  pi, mu, A, D, nu = val
        y = jnp.take(X, k, axis=0)

        # Update statistics
        gam = gamma(k)
        t = posterior(y, pi, mu, A, D, nu)
        s0, s1, S2, s3, s4 = update_stat(y, t, mu, A, D, nu, s0, s1, S2, s3, s4, gam)
        s0, s1, S2, s3, s4 = s0.mean(axis=0), s1.mean(axis=0), S2.mean(axis=0), s3.mean(axis=0), s4.mean(axis=0)

        # Update parameters
        pi, mu, A, D, nu = update_params(s0, s1, S2, s3, s4, D)

        return X, s0, s1, S2, s3, s4,  pi, mu, A, D, nu

    val = (X_batch, s0, s1, S2, s3, s4,  pi, mu, A, D, nu)
    _, _, _, _, _, _,  pi, mu, A, D, nu = fori_loop(0, N, training_step, val)
      
    return pi, mu, A, D, nu

def fit(X, n_components, batch_size):
    import numpy as np
    X_batch, pi, mu, A, D, nu = _initialization(X, n_components, batch_size)
    # def rot(theta):
    #     return jnp.array([[1, 0, 0], [0, jnp.cos(theta), jnp.sin(theta)], [0, -jnp.sin(theta), jnp.cos(theta)]])
    # pi = jnp.array([.1, .4, .5]).astype(jnp.float64)
    # mu = jnp.array([[0, 0, 0, -5], [1, 3, 1, 2], [2, 2, 2, 4]]).astype(jnp.float64)
    # A = jnp.array([[1, 2, 5, 10], [1, 3, 1, 2], [2, 2, 2, 5]]).astype(jnp.float64)
    # D = jnp.array([np.linalg.qr(np.random.normal(size=(4, 4)))[0]]*3).astype(jnp.float64)
    # nu = jnp.array([[5, 5, 5, 5], [5, 5, 5, 2], [5, 5, 5, 10]]).astype(jnp.float64)
    pi, mu, A, D, nu = _fit( X_batch, pi, mu, A, D, nu)
    return {'weights': pi, 'means': mu, 'A': A, 'D': D, 'nu': nu}

@jit
def predict(X, pi, mu, A, D, nu):
    t = posterior(X, pi, mu, A, D, nu)
    return jnp.argmax(t, axis=-1)

@jit
@partial(vmap, in_axes=(0, None, None, None, None, None))
def log_like(y, pi, mu, A, D, nu):
    return logsumexp(mmst_logpdf(y, pi, mu, A, D, nu))

@jit
@partial(vmap, in_axes=(0, None, None, None, None, None))
def weights_mmst(y, pi, mu, A, D, nu):
    alpha, beta = compute_alpha_beta(y, mu, A, D, nu)
    u = _u(alpha, beta)
    tmp = mmst_logpdf(y, pi, mu, A, D, nu)
    t = jnp.exp(tmp - logsumexp(tmp, axis=0))
    w = jnp.einsum('k,ki->i', t, u)
    return jnp.max(w)

def BIC(X, pi, mu, A, D, nu):
    N = X.shape[0]
    n_components, n_features = mu.shape
    L = log_like(X, pi, mu, A, D, nu).sum()
    p = n_components * (1 + (n_features * (n_features + 5)) / 2) - 1
    return - 2 * L + p * jnp.log(N)

def sample_mst(N, muk, Ak, Dk, nuk, key):
    M = nuk.shape[0]
    X = jax.random.normal(key, shape=(N, M))
    W = jax.random.gamma(key, nuk, shape=(N, M))
    X /= jnp.sqrt(W)
    matAk = jnp.diag(jnp.sqrt(Ak))
    coef = Dk@matAk
    return muk + jnp.einsum('ij,kj->ki', coef, X)

def sample_mmst(N, pi, mu, A, D, nu, key):
    samples = []
    cluster = []
    for k in range(len(pi)):
        N_tmp = int(N*pi[k])
        samples.append(sample_mst(N_tmp, mu[k], A[k], D[k], nu[k], key))
        cluster += [k] * N_tmp
    samples = jnp.concatenate(samples)
    shuffle = jax.random.permutation(key, N)
    return samples[shuffle], jnp.array(cluster)[shuffle]

