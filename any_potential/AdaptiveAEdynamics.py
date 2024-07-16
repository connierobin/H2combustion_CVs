import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py
import json
from tqdm import tqdm
import numpy.linalg as la
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import vmap
from functools import partial
import haiku as hk
import optax
import random
import time

# Global dimensionality of the system
n = 0

def potential(qx, qy, qn):
    V = 10 * (qx**4 + qy**4 - 2 * qx**2 - 4 * qy**2 + qx * qy + 0.2 * qx + 0.1 * qy + jnp.sum(qn**2))
    return V


def gradV(q):
    qx = q[:, 0:1]
    qy = q[:, 1:2]
    qn = q[:, 2:]
    Vx = 10 * (4 * qx**3 - 4 * qx + qy + 0.2)
    Vy = 10 * (4 * qy**3 - 8 * qy + qx + 0.1)
    Vn = 10 * 2*qn
    grad = jnp.concatenate((Vx, Vy, Vn), axis=1)
    return grad


# Define the autoencoder
def autoencoder_fn(x, is_training=False):
    input_dim = x.shape[-1]
    intermediate_dim = 64
    encoding_dim = 2

    x = hk.Linear(intermediate_dim)(x)
    x = jax.nn.leaky_relu(x)
    x = hk.Linear(intermediate_dim)(x)
    x = jax.nn.leaky_relu(x)
    encoded = hk.Linear(encoding_dim)(x)

    x = hk.Linear(intermediate_dim)(encoded)
    x = jax.nn.leaky_relu(x)
    x = hk.Linear(intermediate_dim)(x)
    x = jax.nn.leaky_relu(x)
    # decoded = jax.numpy.abs(hk.Linear(input_dim)(x))  # This led to large loss values
    decoded = hk.Linear(input_dim)(x)
    return decoded, encoded


# Define the autoencoder globally
autoencoder = hk.without_apply_rng(hk.transform(autoencoder_fn))


def initialize_autoencoder(rng, sample):
    params = autoencoder.init(rng, sample, is_training=True)
    return params


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
        print(f'batch.shape: {batch.shape}')
        params, opt_state, batch_loss = train_step(params, batch, opt_state, update_fn, is_training=True)
        print(f'batch_loss: {batch_loss[0]}')
        loss += batch_loss[0]  # Extract the actual loss value
        return params, opt_state, loss

    num_batches = len(data) // batch_size
    loss = 0.0
    params, opt_state, loss = jax.lax.fori_loop(0, num_batches, body_fun, (params, opt_state, 0.0))
    return params, opt_state, loss


def train_autoencoder(data, params, opt_state, optimizer, epochs=300, batch_size=4):
    update_fn = optimizer.update
    for epoch in range(epochs):
        # data = jax.random.permutation(jax.random.PRNGKey(epoch), data)
        # params, opt_state, loss = train_epoch(params, data, opt_state, update_fn, batch_size)
        data_scrambled = jax.random.permutation(jax.random.PRNGKey(epoch), data)
        params, opt_state, loss = train_epoch(params, data_scrambled, opt_state, update_fn, batch_size)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    # Calculate encoded values after training
    encoded_values = encode(params, data)
    decoded_values = decode(params, data)
    # encoded_spread = jnp.std(encoded_values, axis=0)

    return params, opt_state, encoded_values, decoded_values

def AE(data, params, opt_state, optimizer):
    params, opt_state, encoded_values, decoded_values = train_autoencoder(data, params, opt_state, optimizer)

    return params, opt_state, encoded_values, decoded_values


# def train_step(params, x, opt_state, optimizer, is_training):
#     (loss, grads) = jax.value_and_grad(mse_loss, has_aux=True)(params, x, is_training)
#     updates, opt_state = optimizer.update(grads, opt_state, params)
#     new_params = optax.apply_updates(params, updates)
#     return new_params, opt_state, loss

# def train_autoencoder(data, params, opt_state, optimizer, epochs=300, batch_size=32):
#     for epoch in range(epochs):
#         data = jax.random.permutation(jax.random.PRNGKey(epoch), data)
#         for i in range(0, len(data), batch_size):
#             batch = data[i:i + batch_size]
#             params, opt_state, loss = train_step(params, batch, opt_state, optimizer, is_training=True)
#         if epoch % 10 == 0:
#             print(f'Epoch {epoch}, Loss: {loss}')
#     return params, opt_state


# def AE(data, params, opt_state, optimizer):
#     params, opt_state = train_autoencoder(data, params, opt_state, optimizer)
#     return params, opt_state


# Returns a None placeholder that could contain auxiliary information
def mse_loss(params, x, is_training):
    decoded, encoded = autoencoder.apply(params, x, is_training=is_training)
    loss = jnp.mean((x - decoded) ** 2)
    return loss, None


def encode(params, x):
    _, encoded = autoencoder.apply(params, x, is_training=False)
    return encoded


def decode(params, x):
    decoded, _ = autoencoder.apply(params, x, is_training=False)
    return decoded


@jax.jit
def SumGaussian_single(x, center, scale_factor, h, sigmas, ep_0b, ep_0w, ep_1b, ep_1w, ep_2b, ep_2w, ep_3b, ep_3w, ep_4b, ep_4w, ep_5b, ep_5w):
    encoder_params = {'linear': {'b': ep_0b, 'w': ep_0w},
                        'linear_1': {'b': ep_1b, 'w': ep_1w},
                        'linear_2': {'b': ep_2b, 'w': ep_2w},
                        'linear_3': {'b': ep_3b, 'w': ep_3w},
                        'linear_4': {'b': ep_4b, 'w': ep_4w},
                        'linear_5': {'b': ep_5b, 'w': ep_5w},}

    # K = latent space dimension
    # N = number of points passed in via x

    x_encoded = encode(encoder_params, x)               # N * K
    center_encoded = encode(encoder_params, center)     # K
    x_projected = x_encoded - center_encoded            # N * K

    x_projected_sq_sums = x_projected**2    # N * K
    N = len(x)
    K = len(sigmas)

    # Vectorized function to calculate exponent
    def calc_exp(x_projected_sq_sum, sigma):
        return scale_factor * h * jnp.exp(-x_projected_sq_sum / (2 * sigma**2))

    # Apply vmap over K (latent dimension)
    calc_exp_vmap = vmap(calc_exp, in_axes=(0, 0))

    # Apply the previous vmap over N (data points)
    vmap_calc_exp_over_n = vmap(calc_exp_vmap, in_axes=(0, None))

    # Compute the exps
    exps = vmap_calc_exp_over_n(x_projected_sq_sums, sigmas)


    do_print = False
    if do_print:
        print(f'sigmas: {sigmas.shape}')
        print(f'x_encoded: {x_encoded.shape}')
        print(f'center_encoded: {center_encoded.shape}')
        print(f'x_projected: {x_projected.shape}')
        print(f'x_projected_sq_sums: {x_projected_sq_sums.shape}')
        print(f'exps: {exps.shape}')

    return exps


# UNSUMMED
@jax.jit
def JSumGaussian(x, centers, encoder_params_list, scale_factors, h, sigma_list):
    # x: 1 * M
    # centers: N * M

    x_jnp = jnp.array(x)
    centers_jnp = jnp.array(centers)
    scale_factors_jnp = jnp.array(scale_factors)
    sigma_list_jnp = jnp.array(sigma_list)
    # encoder_params_list_jnp = jnp.array(encoder_params_list)

    ep_0b = jnp.stack([elem['linear']['b'] for elem in encoder_params_list])
    ep_0w = jnp.stack([elem['linear']['w'] for elem in encoder_params_list])
    ep_1b = jnp.stack([elem['linear_1']['b'] for elem in encoder_params_list])
    ep_1w = jnp.stack([elem['linear_1']['w'] for elem in encoder_params_list])
    ep_2b = jnp.stack([elem['linear_2']['b'] for elem in encoder_params_list])
    ep_2w = jnp.stack([elem['linear_2']['w'] for elem in encoder_params_list])
    ep_3b = jnp.stack([elem['linear_3']['b'] for elem in encoder_params_list])
    ep_3w = jnp.stack([elem['linear_3']['w'] for elem in encoder_params_list])
    ep_4b = jnp.stack([elem['linear_4']['b'] for elem in encoder_params_list])
    ep_4w = jnp.stack([elem['linear_4']['w'] for elem in encoder_params_list])
    ep_5b = jnp.stack([elem['linear_5']['b'] for elem in encoder_params_list])
    ep_5w = jnp.stack([elem['linear_5']['w'] for elem in encoder_params_list])
    
    # Vectorize the single computation over the batch dimension N
    vmap_sum_gaussian = vmap(SumGaussian_single, in_axes=(None, 0, 0, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    total_bias = vmap_sum_gaussian(x_jnp, centers_jnp, scale_factors_jnp, h, sigma_list_jnp, ep_0b, ep_0w, ep_1b, ep_1w, ep_2b, ep_2w, ep_3b, ep_3w, ep_4b, ep_4w, ep_5b, ep_5w)  # N

    print(f'total_bias.shape: {total_bias.shape}')

    total_bias = jnp.sum(total_bias, axis=0)    # N * K -> N

    print(f'total_bias.shape: {total_bias.shape}')

    # TODO??: Normalize AND plot the size of normalization factor
    # Track the new sigma values that we calculate and use that for all calcs

    # TODO: variable sigma's dependent on the size of the eigenvalue. Larger eigenvalue = larger Gaussian
    # NEED that as it might potentially help the AE specifically

    return total_bias  # scalar


jax_SumGaussian = jax.grad(JSumGaussian)
jax_SumGaussian_jit = jax.jit(jax_SumGaussian)

@jax.jit
def GradGaussian(x_in, centers, encoder_params_list, scale_factors, h, sigma_list):
    print(f'jax_SumGaussian_jit._cache_size: {jax_SumGaussian_jit._cache_size()}')
    x_jnp = jnp.array(x_in)
    centers_jnp = jnp.array(centers)        # N * D
    encoder_params_list_jnp = jax.tree_map(lambda x: jnp.array(x), encoder_params_list)
    scale_factors_jnp = jnp.array(scale_factors)        # N * D
    sigma_list_jnp = jnp.array(sigma_list)
    # grad = jax_SumGaussian_jit(x_jnp, centers_jnp, encoder_params_list_jnp, h, sigma)
    grad = jax.grad(lambda x: jnp.sum(JSumGaussian(x, centers_jnp, encoder_params_list_jnp, scale_factors_jnp, h, sigma_list_jnp)))(x_jnp)
    # return jnp.zeros(grad.shape)
    print(f'gradients.shape: {(jnp.sum(grad, axis=0)).shape}')
    return jnp.sum(grad, axis=0)


def calculate_scale_factor(center, encoder_params, h, sigmas):
    # x_projected = encode(encoder_params, center - center)
    # x_projected = jnp.zeros(center.shape)     # encode(center) - encode(center) = 0
    # x_projected_sq_sum = jnp.sum(x_projected**2)
    # gauss_value = h * jnp.exp(-x_projected_sq_sum / (2 * sigmas**2))
    gauss_value = h
    print(f'gauss_value: {gauss_value}')
    # desired_value = h * (1 / (jnp.sqrt(2 * jnp.pi * sigma**2)))
    desired_value = h
    print(f'desired_value: {desired_value}')
    print(f'scale factor: {desired_value / gauss_value}')
    return desired_value / gauss_value


@jax.jit
def next_step(qnow, qs, encoder_params_list, scale_factors, height, sigma_list, dt=1e-3, beta=1.0, step_cap=0.3):
    seed = int(time.time() * 1e6)  # Generate a seed based on the current time in microseconds
    rng_key = jax.random.PRNGKey(seed)
    rng_key, subkey = jax.random.split(rng_key)

    if qs is None:
        step = (- gradV(qnow)) * dt + jnp.sqrt(2 * dt / beta) * jax.random.normal(subkey, shape=qnow.shape)
        step_size = jnp.linalg.norm(step)
        capped_step = jnp.where(step_size > step_cap, step * (step_cap / step_size), step)
        qnext = qnow + capped_step
    else:
        step = (- (gradV(qnow) + GradGaussian(qnow, qs, encoder_params_list, scale_factors, height, sigma_list))) * dt + jnp.sqrt(
            2 * dt / beta) * jax.random.normal(subkey, shape=qnow.shape)
        # step = (- gradV(qnow)) * dt + jnp.sqrt(2 * dt / beta) * jax.random.normal(subkey, shape=qnow.shape)
        step_size = jnp.linalg.norm(step)
        capped_step = jnp.where(step_size > step_cap, step * (step_cap / step_size), step)
        qnext = qnow + capped_step
        # grad = jax.grad(lambda x: jnp.sum(JSumGaussian(x, qs, encoder_params_list, scale_factors, height, sigma)))(qnow)
        # print(f"GradGaussian: {grad}")  # Debug print
        # sum = jnp.sum(JSumGaussian(qnow, qs, encoder_params_list, scale_factors, height, sigma))
        # print(f"SumGaussian: {sum}")  # Debug print
    # print(f"qnow: {qnow}, qnext: {qnext}")  # Debug print
    return qnext


def GradAtCenter(centers, encoder_params_list, scale_factors, h, sigma_list):
    gradients = []  # N * K
    for center, encoder_params, scale_factor, sigmas in zip(centers, encoder_params_list, scale_factors, sigma_list):
        grad = GradGaussian([center], [center], [encoder_params], [scale_factor], h, [sigmas])
        gradients.append(grad)
    print(f'GradAtCenter gradients.shape: {jnp.array(gradients).shape}')
    # return jnp.sum(jnp.array(gradients), axis=1)
    return jnp.array(gradients)


# @jax.jit
# def GradAtCenter(centers, encoder_params_list, scale_factors, h, sigma):
#     def grad_center(center, encoder_params, scale_factor):
#         grad = jax_SumGaussian_jit(center, centers, encoder_params, scale_factors, h, sigma)
#         return grad
    
#     vmap_grad_center = vmap(grad_center, in_axes=(0, None))
#     return vmap_grad_center(centers, encoder_params_list, scale_factors)


def MD(q0, T, Tdeposite, height, sigma_factor, rng, dt=1e-3, beta=1.0, n=0):
    Nsteps = int(T / dt)
    NstepsDeposite = int(Tdeposite / dt)
    Ncenter = int(NstepsDeposite / 2)
    trajectories = np.zeros((Nsteps + 1, q0.shape[0], q0.shape[1]))

    # variance = 0.7  # Threshhold for choosing number of eigenvectors
    q = q0
    qs = None


    # Sample data for initialization
    sample = jnp.ones((1, 2 + n))

    # Initialize the autoencoder
    params = initialize_autoencoder(rng, sample)
    encoder_params_list = [params]

    sigma_list = []
    scale_factors = [1.0]  # Initialize scale factors
    encoded_values_list = []
    decoded_values_list = []
    
    # Initialize the optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)


    for i in tqdm(range(Nsteps)):

        trajectories[i, :] = q
        q = next_step(q, qs, encoder_params_list, scale_factors, height, sigma_list, dt, beta)

        if (i + 1) % NstepsDeposite == 0:

            if qs is None:

                data = trajectories[:NstepsDeposite]  # (N_steps, 1, 2)
                data = np.squeeze(data, axis=1)  # (100, 2)
                mean_vector = np.mean(data, axis=0, keepdims=True)
                center_vector = np.array([data[Ncenter]])
                params, opt_state, encoded_values, decoded_values = AE(data, params, opt_state, optimizer)
                encoder_params_list[0] = params     # overwrite initializer values
                # qs = mean_vector                #data[-2:-1]#mean_vector
                qs = center_vector
                # sigma = max(jnp.std(encoded_values, axis=0)) * sigma_factor
                sigmas = jnp.std(encoded_values, axis=0) * sigma_factor

                # sigma_list.append(sigma)
                sigma_list.append(sigmas)

                # Calculate scale factor for the new center
                new_scale_factor = calculate_scale_factor(center_vector, params, height, sigmas)
                scale_factors[0] = new_scale_factor

            else:
                data = trajectories[i - NstepsDeposite + 1:i + 1]
                data = np.squeeze(data, axis=1)  # (100, 2)
                # mean_vector = np.mean(data, axis=0, keepdims=True)
                center_vector = np.array([data[Ncenter]])
                params, opt_state, encoded_values, decoded_values = AE(data, params, opt_state, optimizer)
                encoder_params_list.append(params)
                # qs = np.concatenate([qs, mean_vector], axis=0)
                qs = np.concatenate([qs, center_vector], axis=0)
                # sigma = max(jnp.std(encoded_values, axis=0)) * sigma_factor
                sigmas = jnp.std(encoded_values, axis=0) * sigma_factor

                # sigma_list.append(sigma)
                sigma_list.append(sigmas)
                # Calculate scale factor for the new center
                new_scale_factor = calculate_scale_factor(center_vector, params, height, sigmas)
                scale_factors.append(new_scale_factor)

            # Store encoded values for visualization
            encoded_values_list.append(encoded_values)
            decoded_values_list.append(decoded_values)

    trajectories[Nsteps, :] = q

    # Calculate gradient directions at each center
    gradient_directions = GradAtCenter(qs, encoder_params_list, scale_factors, height, sigma_list)

    return trajectories, qs, encoder_params_list, scale_factors, gradient_directions, encoded_values_list, decoded_values_list, sigma_list


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return json.dumps(obj.tolist())


def run(filename=None, T=4):
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int, default=0)
    args = parser.parse_args()

    # # number of extra dimensions
    # n = 8

    # T = 100
    dt = 1e-2
    beta = 50
    Tdeposite = 1
    height = 4.
    sigma_factor = 3.
    ic_method = 'AE'
    potential = 'wolfeschlegel'
    # random_seed = 1234
    random_seed = None
    if random_seed == None:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = random_seed
    rng = jax.random.PRNGKey(seed)

    max_qn_val = 20     # size of the relevant part of the potential surface

    q0 = np.concatenate((np.array([[-2.0, 2.0]]), np.array([np.random.rand(n)*(2*max_qn_val) - max_qn_val])), axis=1)
    trajectory, qs, encoder_params_list, scale_factors, gradient_directions, encoded_values_list, decoded_values_list, sigma_list = MD(q0, T, Tdeposite=Tdeposite, height=height, sigma_factor=sigma_factor, rng=rng, dt=dt, beta=beta, n=n)  # (steps, bs, dim)
    
    # TODO: add in parameter for which autoencoder to use -- to change the number of layers, the activations, whether the params are reset, etc.
    simulation_settings = {
        'n': n,
        'T': T,
        'dt': dt,
        'beta': beta,
        'Tdeposite': Tdeposite,
        'height': height,
        'sigma_factor': sigma_factor,
        'ic_method': ic_method,
        'potential': potential,
        'max_qn_val': max_qn_val,
        'random_seed': seed,
    }

    encoder_params_json_strings = [json.dumps(d, default=convert_to_serializable) for d in encoder_params_list]

    with h5py.File('results/run_2.h5', 'w') as h5file:
        dt = h5py.special_dtype(vlen=str)
        h5file.create_dataset('trajectory', data=trajectory)
        h5file.create_dataset('qs', data=qs)
        h5file.create_dataset('encoder_params_list', data=encoder_params_json_strings, dtype=dt)
        h5file.create_dataset('scale_factors', data=scale_factors)
        h5file.create_dataset('gradient_directions', data=gradient_directions)
        h5file.create_dataset('encoded_values_list', data=encoded_values_list)
        h5file.create_dataset('decoded_values_list', data=decoded_values_list)
        h5file.create_dataset('sigma_list', data=sigma_list)

        for key, value in simulation_settings.items():
            h5file.attrs[key] = value




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int, default=0)
    args = parser.parse_args()

    for _ in range(1):
        run(T=100)




    
