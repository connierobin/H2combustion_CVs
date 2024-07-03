import numpy as np
import matplotlib.pyplot as plt
import argparse
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
    x = jax.nn.hard_tanh(x)
    x = hk.Linear(intermediate_dim)(x)
    x = jax.nn.hard_tanh(x)
    encoded = hk.Linear(encoding_dim)(x)

    x = hk.Linear(intermediate_dim)(encoded)
    x = jax.nn.hard_tanh(x)
    x = hk.Linear(intermediate_dim)(x)
    x = jax.nn.hard_tanh(x)
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
    # print('GAUSSIANS RIGHT AFTER TRAINING')
    # print('Encoded values')
    # print(encoded_values)
    # print('Decoded values')
    # print(decoded_values)
    # print('Center')
    # center = np.mean(data, axis=0, keepdims=True)
    # print(center)
    # print('Encoded Center')
    # center_encoded = encode(params, center)
    # print(center_encoded)
    # print('Encoded Minus Encoded Center')
    # x_projected = encoded_values - center_encoded
    # print(x_projected)
    # print('Squared')
    # x_projected_sq_sum = jnp.sum(x_projected**2)
    # print(x_projected_sq_sum)
    # print('Gaussian (no scale factor or height factor, sigma = 1)')
    # scale_factor = 1
    # h = 1
    # sigma = 1
    # exps = scale_factor * h * jnp.exp(-x_projected_sq_sum / (2 * sigma**2))
    # print(exps)
    # print('New exps test: proj squared sum')
    # x_projected_sq = jnp.sum(x_projected**2, axis=1)
    # print(x_projected_sq)
    # print('New exps test: exps')
    # exps = scale_factor * h * jnp.exp(-x_projected_sq / (2 * sigma**2))
    # print(exps)
    # print('Gaussian result')
    # exps_sum = jnp.sum(exps)
    # print(exps_sum)

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


# @jax.jit
# def JSumGaussian(x, centers, encoder_params_list, h, sigma):
#     # x: 1 * M
#     # centers: N * M

#     # Ensure that all parameters in encoder_params_list are correctly batched
#     def batch_params(params):
#         return params

#     encoder_params_batched = batch_params(encoder_params_list[0])
    
#     # Vectorize the single computation over the batch dimension N
#     vmap_sum_gaussian = vmap(SumGaussian_single, in_axes=(0, None, None, None, None))
#     total_bias = vmap_sum_gaussian(x, centers, encoder_params_batched, h, sigma)  # N


#     # TODO??: Normalize AND plot the size of normalization factor
#     # Track the new sigma values that we calculate and use that for all calcs

#     # TODO: variable sigma's dependent on the size of the eigenvalue. Larger eigenvalue = larger Gaussian
#     # NEED that as it might potentially help the AE specifically

#     return jnp.sum(total_bias)  # scalar


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


def MD(q0, T, Tdeposite, height, sigma_factor, dt=1e-3, beta=1.0, n=0):
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
    # rng = jax.random.PRNGKey(42)
    seed = random.randint(0, 2**32 - 1)
    rng = jax.random.PRNGKey(seed)
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

    # Calculate the spread of encoded values
    # encoded_values = encode(params, data)
    # encoded_spread = jnp.std(encoded_values, axis=0)
    # print(f'Encoded spread: {encoded_spread}')
    
    # Adjust sigma value
    # suggested_sigma = jnp.mean(encoded_spread)
    # print(f'suggested sigma: {suggested_sigma}    current sigma: {sigma}')
    # if sigma < 0.1 * suggested_sigma or sigma > 10 * suggested_sigma:
    #     print(f"Warning: Sigma value {sigma} may not be appropriate. Suggested value: {suggested_sigma}")

    return trajectories, qs, encoder_params_list, scale_factors, gradient_directions, encoded_values_list, decoded_values_list, sigma_list


def findTSTime(trajectory):
    x_dimension = trajectory[:, 0, 0]
    y_dimension = trajectory[:, 0, 1]

    first_occurrence_index_1 = -1
    first_occurrence_index_2 = -1
    first_occurrence_index_3 = -1

    # Find the indices where the first dimension is greater than 0
    indices_1 = np.where(x_dimension > 0)[0]
    # Check if any such indices exist
    if indices_1.size > 0:
        # Get the first occurrence
        first_occurrence_index_1 = indices_1[0]
        print(f"The first time step where the first dimension is greater than 0 is: {first_occurrence_index_1}")
    else:
        print("There are no time steps where the first dimension is greater than 0.")

    # Find the indices where the second dimension is greater than 0
    indices_2 = np.where(y_dimension < 0)[0]
    # Check if any such indices exist
    if indices_2.size > 0:
        # Get the first occurrence
        first_occurrence_index_2 = indices_2[0]
        print(f"The first time step where the second dimension is less than 0 is: {first_occurrence_index_2}")
    else:
        print("There are no time steps where the second dimension is less than 0.")

    # Find the indices where the second dimension is greater than 0
    indices_3 = np.where((x_dimension > 0) & (y_dimension < 0))[0]
    # Check if any such indices exist
    if indices_3.size > 0:
        # Get the first occurrence
        first_occurrence_index_3 = indices_3[0]
        print(f"The first time step where the first dimension is greater than zero and the second dimension is less than 0 is: {first_occurrence_index_3}")
    else:
        print("There are no time steps where the first dimension is greater than zero and the second dimension is less than 0.")

    return first_occurrence_index_1, first_occurrence_index_2, first_occurrence_index_3


def visualize_encoded_data(data, params):
    encoded_data = encode(params, data)
    return encoded_data

def plot_encoded_data(encoded_data, gaussian_values):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=gaussian_values, cmap=plt.get_cmap('plasma'), label='Encoded Data with Gaussian Value')
    plt.colorbar(scatter, label='Gaussian Value')

    annotate = False
    if annotate:
        for i in range(len(encoded_data)):
            plt.annotate(f'{i}, {encoded_data[i]}', (encoded_data[i, 0], encoded_data[i, 1]))

    plt.xlabel('Encoded Dimension 1')
    plt.ylabel('Encoded Dimension 2')
    plt.title('Encoded Data in 2D Space')
    plt.legend()
    plt.show()

def visualize_decoded_data(data, params, num_samples=5):
    encoded_data = encode(params, data)
    decoded_data = autoencoder.apply(params, encoded_data, is_training=False)[0]
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))
    for i in range(num_samples):
        original = data[i]
        decoded = decoded_data[i]
        
        axes[i, 0].plot(original, 'b-', label='Original')
        axes[i, 0].set_title(f'Original Sample {i+1}')
        axes[i, 1].plot(decoded, 'r-', label='Decoded')
        axes[i, 1].set_title(f'Decoded Sample {i+1}')
    
    plt.show()


def main_plot(trajectory, qs, encoder_params_list, scale_factors, gradient_directions, encoded_values_list, decoded_values_list, sigma_list, height, NstepsDeposite, T):
    filename = None

    cmap = plt.get_cmap('plasma')

    xx = np.linspace(-3, 3, 200)
    yy = np.linspace(-5, 5, 200)
    [X, Y] = np.meshgrid(xx, yy)  # 100*100
    W = potential(X, Y, np.zeros(n))
    W1 = W.copy()
    W1 = W1.at[W > 300].set(float('nan'))  # Use JAX .at[] method

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    contourf_ = ax1.contourf(X, Y, W1, levels=29)
    plt.colorbar(contourf_)

    indices = np.arange(trajectory.shape[0])
    ax1.scatter(trajectory[:, 0, 0], trajectory[:, 0, 1], c=indices, cmap=cmap)

    num_points = X.shape[0] * X.shape[1]

    # Initialize an empty list to store the results
    results = []

    # Gs = JSumGaussian(jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1), jnp.zeros((num_points, n))], axis=1), qs, encoder_params_list, scale_factors, h=height, sigma=sigma)
    points = jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1), jnp.zeros((num_points, n))], axis=1)

    # Gaussian values not summed over time steps:
    for i, (encoder_params, scale_factor, center, sigmas) in enumerate(zip(encoder_params_list, scale_factors, qs, sigma_list)):
        result = JSumGaussian(points, [center], [encoder_params], [scale_factor], h=height, sigma_list=[sigmas])
        results.append(result)
    results = jnp.array(results)
    results = jnp.sum(results, axis=-1)         # TODO: check if this is correct
    Gs = jnp.sum(results, axis=0)
    Gs = Gs.reshape(X.shape)
    Sum = Gs + W1

    ax2 = fig.add_subplot(1, 3, 2)
    cnf2 = ax2.contourf(X, Y, Gs.reshape(X.shape[0], X.shape[1]), levels=29)
    plt.colorbar(cnf2)
    indices = np.arange(qs.shape[0])
    ax2.scatter(qs[:, 0], qs[:, 1], c=indices, cmap=cmap)
    # ax2.quiver(qs[:, 0], qs[:, 1])
    ax2.quiver(qs[:, 0], qs[:, 1], gradient_directions[:, 0], gradient_directions[:, 1])
    ax2.axis('equal')
    indices = np.arange(trajectory.shape[0])
    # ax2.scatter(trajectory[:, 0, 0], trajectory[:, 0, 1], c=indices, cmap=cmap, alpha=0.1)

    # ax2.scatter(trajectory[:, 0], trajectory[:, 1], c=indices, cmap=cmap)
    ax3 = fig.add_subplot(1, 3, 3)
    cnf3 = ax3.contourf(X, Y, Sum, levels=29)

    # fig.colorbar(contourf_)
    plt.title('Local AE dynamics')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

    ####################
    # PLOT PERFORMANCE #
    ####################
    fig_performance, ax_performance = plt.subplots()
    ax_performance.contourf(X, Y, W1, levels=29)
    plt.colorbar(contourf_, ax=ax_performance)

    colors = [
    'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
    'magenta', 'lime', 'indigo', 'violet', 'gold', 'coral', 'teal', 'navy', 'maroon', 'turquoise']

    # NstepsDeposite = int(Tdeposite / dt)
    reshaped_trajectory = trajectory[:-1].reshape(T, NstepsDeposite, 1, 2 + n)  # Assuming 10 iterations and calculating steps
    m = 5   # only show every 5th data point for visual clarity

    for i, (data, params, decoded_values) in enumerate(zip(reshaped_trajectory, encoder_params_list, decoded_values_list)):
        color = colors[i % len(colors)]
        ax_performance.scatter(data[::m, :, 0], data[::m, :, 1], c=color, marker='o', label=f'Original Data {i+1}')
        ax_performance.scatter(decoded_values[::m, 0], decoded_values[::m, 1], c=color, marker='x', label=f'Decoded Data {i+1}')

    ax_performance.legend()
    plt.title('Autoencoder Performance')
    plt.show()

    ###############
    # PLOT SIGMAS #
    ###############
    fig_sigma, ax_sigma = plt.subplots()
    np_sigma_list = np.array(sigma_list)
    N, K = np_sigma_list.shape
    indices = np.arange(N)
    for k in range(K):
        ax_sigma.scatter(indices, np_sigma_list[:, k], label=f'K={k}')
    plt.xlabel('Iteration')
    plt.ylabel('Sigma value')
    plt.title('Gaussian sigma values / Average encoding variance')
    plt.show()


    ##################
    # PLOT GAUSSIANS #
    ##################
    Ncenter = int(NstepsDeposite / 2)
    # Plot the individual Gaussians
    for i in range(len(results)):
        fig_gaussian, ax_gaussian = plt.subplots()
        ax_gaussian.contourf(X, Y, W1, levels=29)
        cnf = ax_gaussian.contourf(X, Y, results[i].reshape(X.shape[0], X.shape[1]), levels=29)
        plt.colorbar(cnf, ax=ax_gaussian)

        indices = np.arange(reshaped_trajectory.shape[1])
        ax_gaussian.scatter(reshaped_trajectory[i, :, 0, 0], reshaped_trajectory[i, :, 0, 1], c=indices, cmap=cmap)

        ax_gaussian.scatter(qs[i, 0], qs[i, 1], label='Gaussian center', color='pink')

        ax_gaussian.legend()
        plt.title(f'Gaussian {i}')
        plt.show()

        if True:
            gaussian_values = JSumGaussian(reshaped_trajectory[i, :, 0, :], [qs[i]], [encoder_params_list[i]], [scale_factors[i]], h=height, sigma_list=[sigma_list[i]])
            gaussian_values = jnp.sum(gaussian_values, axis=-1)
            plot_encoded_data(encoded_values_list[i], gaussian_values)

# def plot_performance(X, Y, W1, trajectory, qs, encoder_params_list, scale_factors, gradient_directions, encoded_values_list, decoded_values_list, sigma_list):
    




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
    # sigma = 1.25
    # sigma = 0.2       # typical spread of values in latent space seems to be from -8 to 2, although individual ones seem to range over ~2
    sigma_factor = 3.
    ic_method = 'AE'

    

    max_qn_val = 20

    q0 = np.concatenate((np.array([[-2.0, 2.0]]), np.array([np.random.rand(n)*(2*max_qn_val) - max_qn_val])), axis=1)
    trajectory, qs, encoder_params_list, scale_factors, gradient_directions, encoded_values_list, decoded_values_list, sigma_list = MD(q0, T, Tdeposite=Tdeposite, height=height, sigma_factor=sigma_factor, dt=dt, beta=beta, n=n)  # (steps, bs, dim)
    
    first_occurrence_index_1, first_occurrence_index_2, first_occurrence_index_3 = findTSTime(trajectory)
    


    savename = 'results/T{}_Tdeposite{}_dt{}_height{}_sigmafactor{}_beta{}_ic{}'.format(T, Tdeposite, dt, height, sigma_factor, beta, ic_method)
    np.savez(savename, trajectory=trajectory, qs=qs, encoder_params_list=encoder_params_list)

    NstepsDeposite = int(Tdeposite / dt)
    # main_plot(trajectory, qs, encoder_params_list, scale_factors, gradient_directions, encoded_values_list, decoded_values_list, sigma_list, height, NstepsDeposite, T)
    
    return first_occurrence_index_1, first_occurrence_index_2, first_occurrence_index_3, np.mean(sigma_list)
    


    # Visualize encoded data
    # encoded_data_0 = encode(encoder_params_list[0], trajectory)
    # print(encoded_data_0[0:200])
    # plot_encoded_data(encoded_data_0)
    # encoded_data_1 = encode(encoder_params_list[1], trajectory)
    # print(encoded_data_1[0:200])
    # plot_encoded_data(encoded_data_1)
    # encoded_data_2 = encode(encoder_params_list[2], trajectory)
    # print(encoded_data_2[0:200])
    # plot_encoded_data(encoded_data_2)

    # Visualize decoded data
    # visualize_decoded_data(data, params, num_samples=5)


if __name__ == '__main__':
    # i_1, i_2, i_3, sigma = run(T=2)

    results = []

    for _ in range(10):
        i_1, i_2, i_3, sigma = run(T=100)
        results.append((i_1, i_2, i_3, sigma))

    # Print results in a way that's easy to copy and paste
    for result in results:
        print(f"{result[0]}\t{result[1]}\t{result[2]}\t{result[3]}")
        

    # run(filename='T40', T=40)






    