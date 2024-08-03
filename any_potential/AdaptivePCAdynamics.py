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

def potential(qx, qy, qn, pot_name):
    if pot_name == 'rosenbrock':
        return rosenbrock_potential(qx, qy, qn)
    elif pot_name == 'wolfeschlegel':
        return wolfeschlegel_potential(qx, qy, qn)
    return wolfeschlegel_potential(qx, qy, qn)

def gradV(q, pot_name):
    if pot_name == 'rosenbrock':
        return grad_rosenbrock(q)
    elif pot_name == 'wolfeschlegel':
        return grad_wolfeschlegel(q)
    return grad_wolfeschlegel(q)

def wolfeschlegel_potential(qx, qy, qn):
    V = 10 * (qx**4 + qy**4 - 2 * qx**2 - 4 * qy**2 + qx * qy + 0.2 * qx + 0.1 * qy + 10 * jnp.sum(qn**2))
    return V

def grad_wolfeschlegel(q):
    qx = q[:, 0:1]
    qy = q[:, 1:2]
    qn = q[:, 2:]
    Vx = 10 * (4 * qx**3 - 4 * qx + qy + 0.2)
    Vy = 10 * (4 * qy**3 - 8 * qy + qx + 0.1)
    Vn = 100 * 2*qn
    grad = jnp.concatenate((Vx, Vy, Vn), axis=1)
    return grad

def rosenbrock_potential(qx, qy, qn):
    V = 100 * (qy - qx**2)**2 + (1 - qx)**2 + 100 * jnp.sum(qn**2)
    return V

def grad_rosenbrock(q):
    qx = q[:, 0:1]
    qy = q[:, 1:2]
    qn = q[:, 2:]
    Vx = -400 * qx * (qy - qx**2) - 2 * (1 - qx)
    Vy = 200 * (qy - qx**2)
    Vn = 100 * 2*qn
    grad = jnp.concatenate((Vx, Vy, Vn), axis=1)
    return grad


def PCA(data):  # datasize: N * dim
    # Step 4.1: Compute the mean of the data
    mean_vector = np.mean(data, axis=0, keepdims=True)

    # Step 4.2: Center the data by subtracting the mean
    centered_data = data - mean_vector

    # Step 4.3: Compute the covariance matrix of the centered data
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Step 4.4: Perform eigendecomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 4.5: Sort the eigenvectors based on eigenvalues (descending order)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 4.6: Choose the number of components (optional)
    # k = 1  # Set the desired number of components

    # Step 4.7: Retain the top k components
    selected_eigenvectors = eigenvectors[:, :]

    # Step 4.8: Transform your data to the new lower-dimensional space
    transformed_data = np.dot(centered_data, selected_eigenvectors)

    # ##### using the last configuration
    # mean_vector = data[-1:]

    # print(np.dot(centered_data, selected_eigenvectors)-np.matmul(centered_data, selected_eigenvectors))
    # print(eigenvalues, eigenvalues.shape)
    return selected_eigenvectors, eigenvalues, transformed_data

# NOTE: this function is currently NOT IN USE it is mainly for plotting
def GaussiansPCA(q, qs, eigenvectors, choose_eigenvalue, height, sigma):
    choose_eigenvalue_ = np.expand_dims(choose_eigenvalue, axis=1)
    # TODO: put in adjustable sigma values for PCA as well
    # TODO: put in envelope/decay Gaussians

    V = np.empty((q.shape[0], 1))
    for i in range(q.shape[0]):
        x_minus_centers = q[i:i + 1] - qs  # N * M
        x_minus_centers = np.expand_dims(x_minus_centers, axis=1)  # N * 1 * M
        x_projected = np.matmul(x_minus_centers, eigenvectors)
        x_projected_ = x_projected * choose_eigenvalue_


        x_projected_sq_sum = np.sum((x_projected_) ** 2, axis=(-2, -1))  # N

        V[i] = np.sum(height * np.exp(-np.expand_dims(x_projected_sq_sum, axis=1) / 2 / sigma ** 2), axis=0)

    return V

# # This function is used for taking derivatives with jax -- outdated version with only one sigma value
# def SumGaussiansPCA(q, qs, eigenvectors, choose_eigenvalue, height, sigma):
#     choose_eigenvalue_ = jnp.expand_dims(choose_eigenvalue, axis=1)

#     # V = np.empty((q.shape[0], 1))
#     V = 0.0
#     for i in range(q.shape[0]):
#         x_minus_centers = q[i:i + 1] - qs  # N * M
#         x_minus_centers = jnp.expand_dims(x_minus_centers, axis=1)  # N * 1 * M
#         x_projected = jnp.matmul(x_minus_centers, eigenvectors)
#         x_projected_ = x_projected * choose_eigenvalue_
#         x_projected_sq_sum = jnp.sum((x_projected_) ** 2, axis=(-2, -1))  # N

#         V += jnp.sum(height * jnp.exp(-jnp.expand_dims(x_projected_sq_sum, axis=1) / 2 / sigma ** 2), axis=0)

#     return V

# This function is used for taking derivatives with jax
def SumGaussiansPCA(q, qs, eigenvectors, choose_eigenvalue, height, sigma_list):
    choose_eigenvalue_ = jnp.expand_dims(choose_eigenvalue, axis=1)

    V = 0.0
    for i in range(q.shape[0]):
        x_minus_centers = q[i:i + 1] - qs  # N * M
        x_minus_centers = jnp.expand_dims(x_minus_centers, axis=1)  # N * 1 * M
        x_projected = jnp.matmul(x_minus_centers, eigenvectors)
        x_projected_ = x_projected * choose_eigenvalue_
        x_projected_sq_sum = jnp.sum((x_projected_) ** 2, axis=(-2, -1))  # N

        x_projected_sq = jnp.sum((x_projected_) ** 2, axis=(1))
        another_exponent = - x_projected_sq / 2 / sigma_list ** 2
        V += jnp.sum(height * jnp.exp(another_exponent), axis=0)

    return V

jax_PCASumGaussian = jax.grad(SumGaussiansPCA)
jax_PCASumGaussian_jit = jax.jit(jax_PCASumGaussian)

def PCAGradGaussians(q, qs, eigenvectors, choose_eigenvalue, height, sigma_list):
    # print(f'jax_PCASumGaussian_jit._cache_size: {jax_PCASumGaussian_jit._cache_size()}')
    # scale_factors_jnp = jnp.array(scale_factors)        # N * D
    # sigma_list_jnp = jnp.array(sigma_list)

    q_jnp = jnp.array(q)
    qs_jnp = jnp.array(qs)
    eigenvectors_jnp = jnp.array(eigenvectors)
    choose_eigenvalue_jnp = jnp.array(choose_eigenvalue)
    sigma_list_jnp = jnp.array(sigma_list)

    grad = jax.grad(lambda x: jnp.sum(SumGaussiansPCA(x, qs_jnp, eigenvectors_jnp, choose_eigenvalue_jnp, height, sigma_list_jnp)))(q_jnp)
    # return jnp.zeros(grad.shape)
    # print(f'gradients.shape: {(jnp.sum(grad, axis=0)).shape}')
    return jnp.sum(grad, axis=0)


# def gradGaussians(q, qs, eigenvectors, choose_eigenvalue, height, sigma):
#     # print(q.shape, qs.shape, choose_eigenvalue.shape)
#     # x: 1 * M
#     # centers: N * M
#     # eigenvectors: N * M * k
#     # choose_eigenvalue: N*M

#     choose_eigenvalue_ = np.expand_dims(choose_eigenvalue, axis=1) # N*1*M

#     x_minus_centers = q - qs  # N * M
#     x_minus_centers = np.expand_dims(x_minus_centers, axis=1)  # N * 1 * M
#     x_projected = np.matmul(x_minus_centers, eigenvectors)  # N * 1 * k

#     x_projected_ = x_projected * choose_eigenvalue_
#     eigenvectors_ = eigenvectors * choose_eigenvalue_

#     x_projected_sq_sum = np.sum((x_projected_)** 2, axis=(-2, -1))  # N
#     exps = -height / sigma ** 2 * np.exp(-np.expand_dims(x_projected_sq_sum, axis=1) / 2 / sigma ** 2)  # N * 1
#     PTPx = np.matmul(eigenvectors_, np.transpose(x_projected_, axes=(0, 2, 1)))  # N * M * 1
#     PTPx = np.squeeze(PTPx, axis=2)  # N * M
#     grad = np.sum(exps * PTPx, axis=0, keepdims=True)  # 1 * M

#     # Gs_x = height * np.sum(
#     #     eigenvectors.T[:, 0:1] * (-np.sum((q - qs) * eigenvectors.T, axis=1, keepdims=True)) / sigma ** 2 * (
#     #         np.exp(-np.sum((q - qs) * eigenvectors.T, axis=1, keepdims=True) ** 2 / 2 / sigma ** 2)), axis=0,
#     #     keepdims=True)
#     # Gs_y = height * np.sum(
#     #     eigenvectors.T[:, 1:2] * (-np.sum((q - qs) * eigenvectors.T, axis=1, keepdims=True)) / sigma ** 2 * (
#     #         np.exp(-np.sum((q - qs) * eigenvectors.T, axis=1, keepdims=True) ** 2 / 2 / sigma ** 2)), axis=0,
#     #     keepdims=True)

#     return grad#np.concatenate((Gs_x, Gs_y), axis=1)


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


# @jax.jit
def next_step(qnow, qs, eigenvectors, choose_eigenvalue, height, sigma_list, pot_name, dt=1e-3, beta=1.0, step_cap=0.3):
    seed = int(time.time() * 1e6)  # Generate a seed based on the current time in microseconds
    rng_key = jax.random.PRNGKey(seed)
    rng_key, subkey = jax.random.split(rng_key)


    if qs is None:
        step = (- gradV(qnow, pot_name)) * dt
        step_size = jnp.linalg.norm(step)
        capped_step = jnp.where(step_size > step_cap, step * (step_cap / step_size), step)
        nudged_step = capped_step + jnp.sqrt(2 * dt / beta) * jax.random.normal(subkey, shape=qnow.shape)
        qnext = qnow + nudged_step
    else:
        # matrix derivative [[-1.29296676e-19  7.31946182e-20  8.68774077e-18]]
        # auto derivative [-1.2929659e-19  7.3194569e-20  8.6877349e-18]
        # print('TEST GRADIENTS')
        # test derivative
        # eps = 0.0001
        # gradGaussians is the matrix way
        # print('matrix derivative', gradGaussians(qnow, qs, eigenvectors, choose_eigenvalue, height, sigma))
        # print('auto derivative',PCAGradGaussians(qnow, qs, eigenvectors, choose_eigenvalue, height, sigma))
        # # The following code throws an error, but I'm fine with it because the above code works correctly
        # V0 = GaussiansPCA(qnow, qs, eigenvectors, choose_eigenvalue, height, sigma)
        # for i in range(2):
        #     q = qnow.copy()
        #     q.at[0,i] +=  eps
        #     print(qnow, q)
        #     print(str(i) + ' compoenent dev: ', (GaussiansPCA(q, qs, eigenvectors, choose_eigenvalue, height, sigma) - V0)/eps)
        #     print(f'Sum of above: {np.sum((GaussiansPCA(q, qs, eigenvectors, choose_eigenvalue, height, sigma) - V0)/eps)}')


        step = (- (gradV(qnow, pot_name) + PCAGradGaussians(qnow, qs, eigenvectors, choose_eigenvalue, height, sigma_list))) * dt
        # step = (- gradV(qnow)) * dt + jnp.sqrt(2 * dt / beta) * jax.random.normal(subkey, shape=qnow.shape)
        step_size = jnp.linalg.norm(step)
        capped_step = jnp.where(step_size > step_cap, step * (step_cap / step_size), step)
        nudged_step = capped_step + jnp.sqrt(2 * dt / beta) * jax.random.normal(subkey, shape=qnow.shape)
        qnext = qnow + nudged_step
        # grad = jax.grad(lambda x: jnp.sum(JSumGaussian(x, qs, encoder_params_list, scale_factors, height, sigma)))(qnow)
        # print(f"GradGaussian: {grad}")  # Debug print
        # sum = jnp.sum(JSumGaussian(qnow, qs, encoder_params_list, scale_factors, height, sigma))
        # print(f"SumGaussian: {sum}")  # Debug print
    # print(f"qnow: {qnow}, qnext: {qnext}")  # Debug print
    return qnext

def GradAtCenter(centers, eigenvectors_list, choose_eigenvalue_list, h, sigma):
    # gradients = []  # N * K
    # for center, eigenvectors, choose_eigenvalue in zip(centers, eigenvectors_list, choose_eigenvalue_list):
    #     grad = - gradV([center]) + PCAGradGaussians([center], [center], eigenvectors, choose_eigenvalue, h, sigma)
    #     gradients.append(grad)
    # print(f'GradAtCenter gradients.shape: {jnp.array(gradients).shape}')
    # # return jnp.sum(jnp.array(gradients), axis=1)
    # return jnp.array(gradients)
    return np.zeros(centers.shape[0])


# @jax.jit
# def GradAtCenter(centers, encoder_params_list, scale_factors, h, sigma):
#     def grad_center(center, encoder_params, scale_factor):
#         grad = jax_SumGaussian_jit(center, centers, encoder_params, scale_factors, h, sigma)
#         return grad
    
#     vmap_grad_center = vmap(grad_center, in_axes=(0, None))
#     return vmap_grad_center(centers, encoder_params_list, scale_factors)


def MD(q0, T, Tdeposite, height, sigma, sigma_factor, rng, decay_sigma, pot_name, dt=1e-3, beta=1.0, n=0, threshold=-1., reset_params=False):
    Nsteps = int(T / dt)
    NstepsDeposite = int(Tdeposite / dt)
    Ncenter = int(NstepsDeposite / 2)
    trajectories = np.zeros((Nsteps + 1, q0.shape[0], q0.shape[1]))

    variance = 0.01  # Threshhold for choosing number of eigenvectors in PCA
    q = q0
    qs = None


    # TODO: put these back in later
    # TODO: also delete the 'threshold' parameter
    sigma_list = []
    scale_factors = [1.0]  # Initialize scale factors

    qs = None
    eigenvectors = None
    save_eigenvalues = None
    choose_eigenvalue = None

    for i in tqdm(range(Nsteps)):

        trajectories[i, :] = q
        q = next_step(q, qs, eigenvectors, choose_eigenvalue, height, sigma_list, pot_name, dt, beta)

        if (i + 1) % NstepsDeposite == 0:

            if qs is None:

                data = trajectories[:NstepsDeposite]  # (N_steps, 1, 2)
                data = np.squeeze(data, axis=1)  # (100, 2)
                mean_vector = np.mean(data, axis=0, keepdims=True)

                selected_eigenvectors, eigenvalues, transformed_data = PCA(data)

                qs = mean_vector                #data[-2:-1]#mean_vector
                # qs = center_vector
                eigenvectors = np.expand_dims(selected_eigenvectors, axis=0)
                save_eigenvalues = np.expand_dims(eigenvalues, axis=0)

                eigenvalues = np.expand_dims(eigenvalues, axis=0)
                choose_eigenvalue_tmp = np.zeros((1, 2 + n))
                cumsum = np.cumsum(eigenvalues, axis=1)
                var_ratio = cumsum / np.sum(save_eigenvalues)
                idx = np.argmax(var_ratio > variance)

                for s in range(idx + 1):
                    choose_eigenvalue_tmp[0, s] = 1
                choose_eigenvalue = choose_eigenvalue_tmp

                # TODO: put this back in later
                # sigma = max(jnp.std(encoded_values, axis=0)) * sigma_factor
                # sigmas = jnp.std(encoded_values, axis=0) * sigma_factor
                sigmas = jnp.std(transformed_data, axis=0) * sigma_factor

                # sigma_list.append(sigma)
                sigma_list.append(sigmas)

                # TODO: put this back in later?
                # Calculate scale factor for the new center
                # new_scale_factor = calculate_scale_factor(center_vector, params, height, sigmas)
                # scale_factors[0] = new_scale_factor

            else:
                data = trajectories[i - NstepsDeposite + 1:i + 1]
                data = np.squeeze(data, axis=1)  # (100, 2)
                mean_vector = np.mean(data, axis=0, keepdims=True)

                selected_eigenvectors, eigenvalues, transformed_data = PCA(data)

                qs = np.concatenate([qs, mean_vector], axis=0)
                # qs = np.concatenate([qs, center_vector], axis=0)

                eigenvectors = np.concatenate([eigenvectors, np.expand_dims(selected_eigenvectors, axis=0)], axis=0)
                save_eigenvalues = np.concatenate([save_eigenvalues, np.expand_dims(eigenvalues, axis=0)], axis=0)

                eigenvalues = np.expand_dims(eigenvalues, axis=0)
                choose_eigenvalue_tmp = np.zeros((1, 2 + n))
                cumsum = np.cumsum(eigenvalues, axis=1)
                var_ratio = cumsum / np.sum(eigenvalues)
                idx = np.argmax(var_ratio > variance)

                for s in range(idx + 1):
                    choose_eigenvalue_tmp[0, s] = 1
                choose_eigenvalue = np.concatenate([choose_eigenvalue, choose_eigenvalue_tmp], axis=0)

                # TODO: put these back later
                # sigma = max(jnp.std(encoded_values, axis=0)) * sigma_factor
                # sigmas = jnp.std(encoded_values, axis=0) * sigma_factor
                sigmas = jnp.std(transformed_data, axis=0) * sigma_factor

                # sigma_list.append(sigma)
                sigma_list.append(sigmas)
                # Calculate scale factor for the new center
                # new_scale_factor = calculate_scale_factor(center_vector, params, height, sigmas)
                # scale_factors.append(new_scale_factor)


    trajectories[Nsteps, :] = q

    # Calculate gradient directions at each center
    gradient_directions = GradAtCenter(qs, eigenvectors, choose_eigenvalue, height, sigma)

    return trajectories, qs, eigenvectors, eigenvalues, choose_eigenvalue, gradient_directions, sigma_list


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return json.dumps(obj.tolist())


def run(filename=None, T=4, potential='wolfeschlegel'):
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
    sigma = 0.7
    sigma_factor = 3.
    decay_sigma = 0.7
    reset_params = True
    threshold = 0.003
    ic_method = 'PCA'
    # potential = 'wolfeschlegel'
    # potential = 'rosenbrock'
    # random_seed = 1234
    random_seed = None
    if random_seed == None:
        seed = random.randint(0, 2**32 - 1)
    else:
        seed = random_seed
    rng = jax.random.PRNGKey(seed)

    max_qn_val = 20     # size of the relevant part of the potential surface

    q0 = np.concatenate((np.array([[-2.0, 2.0]]), np.array([np.random.rand(n)*(2*max_qn_val) - max_qn_val])), axis=1)
    trajectory, qs, eigenvectors, eigenvalues, choose_eigenvalue, gradient_directions, sigma_list = MD(q0, T, Tdeposite=Tdeposite, height=height, sigma_factor=sigma_factor, sigma=sigma, decay_sigma=decay_sigma, rng=rng, pot_name=potential, dt=dt, beta=beta, n=n, threshold=threshold, reset_params=False)  # (steps, bs, dim)

    # TODO: add in parameter for which autoencoder to use -- to change the number of layers, the activations, whether the params are reset, etc.
    simulation_settings = {
        'n': n,
        'T': T,
        'dt': dt,
        'beta': beta,
        'Tdeposite': Tdeposite,
        'height': height,
        'sigma': sigma,
        'sigma_factor': sigma_factor,
        'decay_sigma': decay_sigma,
        'threshold': threshold,
        'reset_params': reset_params,
        'ic_method': ic_method,
        'potential': potential,
        'max_qn_val': max_qn_val,
        'random_seed': seed,
    }

    with h5py.File(filename, 'w') as h5file:
        dt = h5py.special_dtype(vlen=str)
        h5file.create_dataset('trajectory', data=trajectory)
        h5file.create_dataset('qs', data=qs)
        h5file.create_dataset('eigenvectors', data=eigenvectors)
        h5file.create_dataset('eigenvalues', data=eigenvalues)
        h5file.create_dataset('choose_eigenvalue', data=choose_eigenvalue)
        # h5file.create_dataset('scale_factors', data=scale_factors)
        h5file.create_dataset('gradient_directions', data=gradient_directions)
        h5file.create_dataset('sigma_list', data=sigma_list)

        for key, value in simulation_settings.items():
            h5file.attrs[key] = value




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int, default=0)
    args = parser.parse_args()

    potential='wolfeschlegel'

    for i in range(10):
        name = f'PCA_results/run_ws_n0_k1_{i+11}.h5'
        run(filename=name, T=100, potential=potential)




    
