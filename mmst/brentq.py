import jax
import jax.numpy as jnp

ECONVERGED = 0
ECONVERR = -1

def jax_brentq(f):
  rtol = 4 * jnp.finfo(jnp.float64).eps

  def x(a, b, args=(), xtol=2e-14, maxiter=200):
    xpre = a * 1.0
    xcur = b * 1.0
    fpre = f(xpre, *args)
    fcur = f(xcur, *args)

    root = jnp.where(fpre == 0, xpre, 0.)
    status = jnp.where(fpre == 0, ECONVERGED, ECONVERR)
    root = jnp.where(fcur == 0, xcur, root)
    status = jnp.where(fcur == 0, ECONVERGED, status)

    def _f1(x):
      x['xblk'] = x['xpre']
      x['fblk'] = x['fpre']
      x['spre'] = x['xcur'] - x['xpre']
      x['scur'] = x['xcur'] - x['xpre']
      return x

    def _f2(x):
      x['xpre'] = x['xcur']
      x['xcur'] = x['xblk']
      x['xblk'] = x['xpre']
      x['fpre'] = x['fcur']
      x['fcur'] = x['fblk']
      x['fblk'] = x['fpre']
      return x

    def _f5(x):
      x['stry'] = -x['fcur'] * (x['xcur'] - x['xpre']) / (x['fcur'] - x['fpre'])
      return x

    def _f6(x):
      x['dpre'] = (x['fpre'] - x['fcur']) / (x['xpre'] - x['xcur'])
      dblk = (x['fblk'] - x['fcur']) / (x['xblk'] - x['xcur'])
      _tmp = dblk * x['dpre'] * (x['fblk'] - x['fpre'])
      x['stry'] = -x['fcur'] * (x['fblk'] * dblk - x['fpre'] * x['dpre']) / _tmp
      return x

    def _f3(x):
      x = jax.lax.cond(x['xpre'] == x['xblk'], _f5, _f6, x)
      k = jnp.min(jnp.array([abs(x['spre']), 3 * abs(x['sbis']) - x['delta']]))
      j = 2 * abs(x['stry']) < k
      x['spre'] = jnp.where(j, x['scur'], x['sbis'])
      x['scur'] = jnp.where(j, x['stry'], x['sbis'])
      return x

    def _f4(x): 
      x['spre'] = x['sbis']
      x['scur'] = x['sbis']
      return x

    def body_fun(x):
      x['itr'] += 1
      x = jax.lax.cond(x['fpre'] * x['fcur'] < 0, _f1, lambda a: a, x)
      x = jax.lax.cond(abs(x['fblk']) < abs(x['fcur']), _f2, lambda a: a, x)
      x['delta'] = (xtol + rtol * abs(x['xcur'])) / 2
      x['sbis'] = (x['xblk'] - x['xcur']) / 2
      j = jnp.logical_or(x['fcur'] == 0, abs(x['sbis']) < x['delta'])
      x['status'] = jnp.where(j, ECONVERGED, x['status'])
      x['root'] = jnp.where(j, x['xcur'], x['root'])
      x = jax.lax.cond(jnp.logical_and(abs(x['spre']) > x['delta'], abs(x['fcur']) < abs(x['fpre'])),
                       _f3, _f4, x)
      x['xpre'] = x['xcur']
      x['fpre'] = x['fcur']
      x['xcur'] += jnp.where(abs(x['scur']) > x['delta'],
                          x['scur'], jnp.where(x['sbis'] > 0, x['delta'], -x['delta']))
      x['fcur'] = f(x['xcur'], *args)
      x['funcalls'] += 1
      return x

    def cond_fun(R):
      return jnp.logical_and(R['status'] != ECONVERGED, R['itr'] <= maxiter)

    R = dict(root=root, status=status, xpre=xpre, xcur=xcur, fpre=fpre, fcur=fcur,
             itr=0, funcalls=2, xblk=xpre, fblk=fpre,
             sbis=(xpre - xcur) / 2,
             delta=(xtol + rtol * abs(xcur)) / 2,
             stry=-fcur * (xcur - xpre) / (fcur - fpre),
             scur=xcur - xpre, spre=xcur - xpre,
             dpre=(fpre - fcur) / (xpre - xcur))
    R = jax.lax.cond(status == ECONVERGED,
                     lambda x: x,
                     lambda x: jax.lax.while_loop(cond_fun, body_fun, x),
                     R)
    return R['root']

  return x
