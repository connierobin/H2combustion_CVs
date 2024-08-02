import jax.numpy as jnp
from jax import vmap

def SumGaussianPW_single(pw_x, pw_center, eigenvectors, orth_eigenvectors, h, sigma, env_sigma):
    pw_x_minus_center = pw_x - pw_center  # D
    pw_x_projected = jnp.matmul(pw_x_minus_center, eigenvectors)  # k
    pw_x_projected_sq_sum = jnp.sum(pw_x_projected**2)  # scalar

    exps = h * jnp.exp(-pw_x_projected_sq_sum / (2 * sigma**2))  # scalar

    pairdists_projected_orth = jnp.matmul(pw_x_minus_center, orth_eigenvectors) * orth_eigenvectors  # D * k
    pairdists_projected_sum = jnp.sum(pairdists_projected_orth, axis=-1)  # D
    pairdists_remainder = pw_x_minus_center - pairdists_projected_sum  # D

    pairdists_envelope_sq_sum = jnp.sum(pairdists_remainder**2)  # scalar

    envelope_exps = h * jnp.exp(-pairdists_envelope_sq_sum / (2 * env_sigma**2))  # scalar

    return exps * envelope_exps  # scalar

def SumGaussianPW(pw_x, pw_centers, eigenvectors, orth_eigenvectors, h, sigma, unsummed=False):
    env_sigma = sigma

    # Vectorize the single computation over the batch dimension N
    vmap_sum_gaussian_pw = vmap(SumGaussianPW_single, in_axes=(0, 0, 0, 0, None, None, None))

    total_bias = vmap_sum_gaussian_pw(pw_x, pw_centers, eigenvectors, orth_eigenvectors, h, sigma, env_sigma)  # N

    if unsummed:
        return total_bias
    else:
        return jnp.sum(total_bias)  # scalar

# Example usage with JAX arrays
import jax

N, D, k = 100, 50, 10  # Example dimensions
pw_x = jax.random.normal(jax.random.PRNGKey(0), (N, D))
pw_centers = jax.random.normal(jax.random.PRNGKey(1), (N, D))
eigenvectors = jax.random.normal(jax.random.PRNGKey(2), (N, D, k))
orth_eigenvectors = jax.random.normal(jax.random.PRNGKey(3), (N, D, k))
h = 1.0
sigma = 0.5

result = SumGaussianPW(pw_x, pw_centers, eigenvectors, orth_eigenvectors, h, sigma)
print(result)
