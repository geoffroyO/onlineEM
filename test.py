from gmm.GaussianMixture_jax import GMMOEM
import numpy as np
import jax

n_components = 5
batch_size = 256
n_first = 1000
X = np.random.normal(size=(256*1000, 10))
gmm = GMMOEM(n_components)
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    gmm.fit(X, batch_size)
for k in range(10):
    printopp(k)

