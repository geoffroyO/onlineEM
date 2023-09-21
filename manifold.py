"""Manifold operations."""
import jax.numpy as jnp
from jax.lax import while_loop


def norm(tangent_vector):
    return jnp.linalg.norm(tangent_vector)


def retraction(point, tangent_vector):
    a = point + tangent_vector
    q, r = jnp.linalg.qr(a)
    s = jnp.diagonal(r, axis1=-2, axis2=-1)
    s = s + jnp.where(s == 0, 1, 0)
    s = s / jnp.abs(s)
    q = q * s.T
    return q


def inner_product(tangent_vector_a, tangent_vector_b):
    return jnp.trace(tangent_vector_a @ tangent_vector_b.T)


def projection(point, tangent_vector):
    tmp = point.T @ tangent_vector
    return tangent_vector - point @ (tmp + tmp.T) / 2


def transport(point, tangent_vector):
    return projection(point, tangent_vector)


def riemannian_gradient(point, euclidian_gradient):
    return projection(point, euclidian_gradient)


def beta_polak_ribiere(newgrad, Pnewgrad, gradPgrad, oldgrad): 
    ip_diff = inner_product(Pnewgrad, newgrad - oldgrad)
    return jnp.where(ip_diff / gradPgrad > 0, ip_diff / gradPgrad, 0)


def line_search(objective, x, d, f0, df0, oldalpha):
    norm_d = norm(d)

    alpha = jnp.where(oldalpha == -1, 1 / norm_d, oldalpha)
    newx = retraction(x, alpha * d)
    newf = objective(newx)
    cost_evaluations = 1

    def body(val):
        alpha, newx, newf, cost_evaluations = val
        alpha *= .5
        newx = retraction(x, alpha * d)
        newf = objective(newx)
        cost_evaluations += 1
        return alpha, newx, newf, cost_evaluations

    def cond(val):
        alpha, _, newf, cost_evaluations = val
        return (newf > f0 + 0.5 * alpha * df0) * (cost_evaluations <= 10)

    val_init = alpha, newx, newf, cost_evaluations
    alpha, newx, newf, cost_evaluations = while_loop(cond, body, val_init)
    alpha = jnp.where(newf > f0, 0., alpha)
    newx = jnp.where(newf > f0, x,  newx)

    oldalpha = jnp.where(cost_evaluations == 2, alpha, 2 * alpha)

    return newx, oldalpha
