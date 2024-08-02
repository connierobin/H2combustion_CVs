import jax
import jax.numpy as jnp
import numpy as np
import time

@jax.jit
def jax_gauss(x, sigma):
    return sigma * jnp.exp(-x**2/2)

def easy_gauss_and_grad(x, sigma):
    x_jnp = jnp.array(x)
    sig_jnp = jnp.array(sigma)
    jax_gauss_and_grad = jax.value_and_grad(jax_gauss)
    jax_gauss_and_grad_jit = jax.jit(jax_gauss_and_grad)
    val, grad = jax_gauss_and_grad_jit(x_jnp, sig_jnp)

    return np.array(val), np.array(grad)

t1 = time.time()
easy_gauss_and_grad(np.array(0.), np.array(1.))
t2 = time.time()
easy_gauss_and_grad(np.array(1.), np.array(0.))
t3 = time.time()
easy_gauss_and_grad(np.array(1.), np.array(1.))
t4 = time.time()

print(f'First: {t2-t1}')
print(f'Second: {t3-t2}')
print(f'Third: {t4-t3}')

print(easy_gauss_and_grad(np.array(1.), np.array(1.)))