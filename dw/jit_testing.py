import time
import jax
import jax.numpy as jnp
from jax import vmap
import haiku as hk
import optax
from jax import random
from functools import partial

# Global dimensionality of the system
n = 8

def potential(qx, qy, qn):
    V = 0.1*(qy +0.1*qx**3)**2 + 2*jnp.exp(-qx**2) + (qx**2+qy**2)/36 + jnp.sum(qn**2)/36
    return V

def gradV(q):
    qx = q[:, 0:1]
    qy = q[:, 1:2]
    qn = q[:, 2:]
    Vx = 0.1*2*(qy +0.1*qx**3)*3*0.1*qx**2 - 2*qx*2*jnp.exp(-qx**2) + 2*qx/36
    Vy = 0.1*2*(qy +0.1*qx**3) + 2*qy/36
    Vn = 2*qn/36
    return jnp.concatenate((Vx, Vy, Vn), axis=1)

# Your previously defined functions (potential, gradV, etc.)

def autoencoder_fn(x, is_training=False):
    input_dim = x.shape[-1]
    intermediate_dim = 64
    encoding_dim = 3

    x = hk.Linear(intermediate_dim)(x)
    x = jax.nn.relu(x)
    x = hk.Linear(intermediate_dim)(x)
    x = jax.nn.relu(x)
    encoded = hk.Linear(encoding_dim)(x)

    x = hk.Linear(intermediate_dim)(encoded)
    x = jax.nn.relu(x)
    x = hk.Linear(intermediate_dim)(x)
    x = jax.nn.relu(x)
    decoded = hk.Linear(input_dim)(x)
    return decoded, encoded

autoencoder = hk.without_apply_rng(hk.transform(autoencoder_fn))

def initialize_autoencoder(rng, sample):
    params = autoencoder.init(rng, sample, is_training=True)
    return params

# Returns a None placeholder that could contain auxiliary information
def mse_loss(params, x, is_training):
    decoded, encoded = autoencoder.apply(params, x, is_training=is_training)
    loss = jnp.mean((x - decoded) ** 2)
    return loss, None

@partial(jax.jit, static_argnums=(3, 4))
def train_step(params, x, opt_state, update_fn, is_training):
    (loss, grads) = jax.value_and_grad(mse_loss, has_aux=True)(params, x, is_training)
    updates, opt_state = update_fn(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

@partial(jax.jit, static_argnums=(3, 4))
def train_epoch(params, data, opt_state, update_fn, batch_size):
    def body_fun(i, carry):
        params, opt_state, loss = carry
        start = i * batch_size
        batch = jax.lax.dynamic_slice(data, (start, 0), (batch_size, data.shape[1]))
        params, opt_state, batch_loss = train_step(params, batch, opt_state, update_fn, is_training=True)
        loss += batch_loss[0]  # Extract the actual loss value
        return params, opt_state, loss

    num_batches = len(data) // batch_size
    params, opt_state, loss = jax.lax.fori_loop(0, num_batches, body_fun, (params, opt_state, 0.0))
    return params, opt_state, loss

def train_autoencoder(data, params, opt_state, optimizer, epochs=300, batch_size=32):
    update_fn = optimizer.update
    for epoch in range(epochs):
        data = jax.random.permutation(jax.random.PRNGKey(epoch), data)
        params, opt_state, loss = train_epoch(params, data, opt_state, update_fn, batch_size)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    return params, opt_state

def AE(data, params, opt_state, optimizer):
    params, opt_state = train_autoencoder(data, params, opt_state, optimizer)
    return params, opt_state

def encode(params, x):
    _, encoded = autoencoder.apply(params, x, is_training=False)
    return encoded

@jax.jit
def SumGaussian_single(x, center, encoder_params, h, sigma):
    x_minus_center = x - center
    x_projected = encode(encoder_params, x_minus_center)
    x_projected_sq_sum = jnp.sum(x_projected**2)
    exps = h * jnp.exp(-x_projected_sq_sum / (2 * sigma**2))
    return exps

@jax.jit
def JSumGaussian(x, centers, encoder_params_list, h, sigma):
    def batch_params(params):
        return params

    encoder_params_batched = batch_params(encoder_params_list[0])
    
    vmap_sum_gaussian = vmap(SumGaussian_single, in_axes=(0, None, None, None, None))
    total_bias = vmap_sum_gaussian(x, centers, encoder_params_batched, h, sigma)
    return total_bias

@jax.jit
def GradGaussian(x, centers, encoder_params_list, h, sigma):
    x_jnp = jnp.array(x)
    centers_jnp = jnp.array(centers)
    encoder_params_list_jnp = jax.tree_map(lambda x: jnp.array(x), encoder_params_list)
    grad = jax.grad(lambda x: jnp.sum(JSumGaussian(x, centers_jnp, encoder_params_list_jnp, h, sigma)))(x_jnp)
    return grad

@jax.jit
def next_step(qnow, qs, encoder_params_list, height, sigma, dt=1e-3, beta=1.0):
    if qs is None:
        qnext = qnow + (- gradV(qnow)) * dt + jnp.sqrt(2 * dt / beta) * jax.random.normal(jax.random.PRNGKey(0), shape=qnow.shape)
    else:
        qnext = qnow + (- (gradV(qnow) + GradGaussian(qnow, qs, encoder_params_list, height, sigma))) * dt + jnp.sqrt(2 * dt / beta) * jax.random.normal(jax.random.PRNGKey(0), shape=qnow.shape)
    return qnext

def test_new_version():
    rng = jax.random.PRNGKey(42)
    sample = jnp.ones((1, 2 + n))
    params = initialize_autoencoder(rng, sample)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    data = jax.random.normal(rng, (100, 2 + n))

    start_time = time.time()
    params, opt_state = train_autoencoder(data, params, opt_state, optimizer, epochs=10, batch_size=10)
    end_time = time.time()
    print(f"Time taken for training (10 epochs): {end_time - start_time:.4f} seconds")

    q0 = jax.random.normal(rng, (1, 2 + n))
    qs = jax.random.normal(rng, (10, 2 + n))
    encoder_params_list = [params]

    start_time = time.time()
    for _ in range(1000):
        q0 = next_step(q0, qs, encoder_params_list, height=0.25, sigma=1.25, dt=1e-2, beta=4)
    end_time = time.time()
    print(f"Time taken for 1000 next_step calls: {end_time - start_time:.4f} seconds")
    jax.clear_caches()

# Run test for the new version
test_new_version()
