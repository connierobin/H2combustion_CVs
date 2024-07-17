import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import jax
from jax import vmap
import jax.numpy as jnp
import haiku as hk
import re


###############
# AUTOENCODER #
###############

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

def encode(params, x):
    _, encoded = autoencoder.apply(params, x, is_training=False)
    return encoded

# Define the autoencoder globally
autoencoder = hk.without_apply_rng(hk.transform(autoencoder_fn))


#####################
# SUMMING GAUSSIANS #
#####################

# @jax.jit
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
def JSumGaussian(x, centers, encoder_params_list, scale_factors, h, sigma_list):
    # x: 1 * M
    # centers: N * M

    x_jnp = jnp.array(x)
    centers_jnp = jnp.array(centers)
    scale_factors_jnp = jnp.array(scale_factors)
    sigma_list_jnp = jnp.array(sigma_list)
    # encoder_params_list_jnp = jnp.array(encoder_params_list)

    ep_0b = jnp.stack([np.array(elem['linear']['b']) for elem in encoder_params_list])
    ep_0w = jnp.stack([np.array(elem['linear']['w']) for elem in encoder_params_list])
    ep_1b = jnp.stack([np.array(elem['linear_1']['b']) for elem in encoder_params_list])
    ep_1w = jnp.stack([np.array(elem['linear_1']['w']) for elem in encoder_params_list])
    ep_2b = jnp.stack([np.array(elem['linear_2']['b']) for elem in encoder_params_list])
    ep_2w = jnp.stack([np.array(elem['linear_2']['w']) for elem in encoder_params_list])
    ep_3b = jnp.stack([np.array(elem['linear_3']['b']) for elem in encoder_params_list])
    ep_3w = jnp.stack([np.array(elem['linear_3']['w']) for elem in encoder_params_list])
    ep_4b = jnp.stack([np.array(elem['linear_4']['b']) for elem in encoder_params_list])
    ep_4w = jnp.stack([np.array(elem['linear_4']['w']) for elem in encoder_params_list])
    ep_5b = jnp.stack([np.array(elem['linear_5']['b']) for elem in encoder_params_list])
    ep_5w = jnp.stack([np.array(elem['linear_5']['w']) for elem in encoder_params_list])
    
    # Vectorize the single computation over the batch dimension N
    vmap_sum_gaussian = vmap(SumGaussian_single, in_axes=(None, 0, 0, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    total_bias = vmap_sum_gaussian(x_jnp, centers_jnp, scale_factors_jnp, h, sigma_list_jnp, ep_0b, ep_0w, ep_1b, ep_1w, ep_2b, ep_2w, ep_3b, ep_3w, ep_4b, ep_4w, ep_5b, ep_5w)  # N

    # print(f'total_bias.shape: {total_bias.shape}')

    total_bias = jnp.sum(total_bias, axis=0)    # N * K -> N

    # print(f'total_bias.shape: {total_bias.shape}')

    # TODO??: Normalize AND plot the size of normalization factor
    # Track the new sigma values that we calculate and use that for all calcs

    # TODO: variable sigma's dependent on the size of the eigenvalue. Larger eigenvalue = larger Gaussian
    # NEED that as it might potentially help the AE specifically

    return total_bias  # scalar

############
# PLOTTING #
############

def wolfeschlegel_potential(qx, qy, qn):
    V = 10 * (qx**4 + qy**4 - 2 * qx**2 - 4 * qy**2 + qx * qy + 0.2 * qx + 0.1 * qy + jnp.sum(qn**2))
    return V

def main_plot(potential, n, trajectory, qs, encoder_params_list, scale_factors, gradient_directions, encoded_values_list, decoded_values_list, sigma_list, height, NstepsDeposite, T):
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
    
def findTSTime(trajectory):
    x_dimension = trajectory[:, 0, 0]
    y_dimension = trajectory[:, 0, 1]

    first_occurrence_index_1 = -1
    first_occurrence_index_2 = -1
    first_occurrence_index_3 = -1

    # Upper Right Quadrant
    indices_1 = np.where((x_dimension > 0.1) & (y_dimension > 0.1))[0]
    # Check if any such indices exist
    if indices_1.size > 0:
        # Get the first occurrence
        first_occurrence_index_1 = indices_1[0]
        print(f"The first time step where the first dimension is greater than 0 is: {first_occurrence_index_1}")
    else:
        print("There are no time steps where the first dimension is greater than 0.")

    # Lower Left Quadrant
    indices_2 = np.where((x_dimension < -0.1) & (y_dimension < -0.1))[0]
    # Check if any such indices exist
    if indices_2.size > 0:
        # Get the first occurrence
        first_occurrence_index_2 = indices_2[0]
        print(f"The first time step where the second dimension is less than 0 is: {first_occurrence_index_2}")
    else:
        print("There are no time steps where the second dimension is less than 0.")

    # Lower Right Quadrant
    indices_3 = np.where((x_dimension > 0.1) & (y_dimension < -0.1))[0]
    # Check if any such indices exist
    if indices_3.size > 0:
        # Get the first occurrence
        first_occurrence_index_3 = indices_3[0]
        print(f"The first time step where the first dimension is greater than zero and the second dimension is less than 0 is: {first_occurrence_index_3}")
    else:
        print("There are no time steps where the first dimension is greater than zero and the second dimension is less than 0.")

    return first_occurrence_index_1, first_occurrence_index_2, first_occurrence_index_3


def load_data():
    def convert_to_float(obj):
        if isinstance(obj, dict):
            return {k: convert_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_float(item) for item in obj]
        elif isinstance(obj, str):
            # Check if the string represents a list or nested list of floats
            if re.match(r'^\[.*\]$', obj):
                try:
                    # Safely evaluate the string as a nested list of floats
                    return json.loads(obj, parse_float=float)
                except (ValueError, TypeError):
                    return obj
            else:
                try:
                    return float(obj)
                except (ValueError, TypeError):
                    return obj
        else:
            return obj

    with h5py.File('../results/run_n3_10.h5', 'r') as h5file:

        # The encoder parameters are saved via json
        json_strings = h5file['encoder_params_list'][:]
        encoder_params_list = [convert_to_float(json.loads(s)) for s in json_strings]

        elem = encoder_params_list[0]['linear']['b'][1]
        print(f'encoder params[0][linear][b][1]: {elem}')
        # print(f'encoder params[0][linear][b][1].dtype: {elem.dtype}')

        # Data from the simulation
        trajectory = h5file['trajectory'][:]
        qs = h5file['qs'][:]
        scale_factors = h5file['scale_factors'][:]
        gradient_directions = h5file['gradient_directions'][:]
        encoded_values_list = h5file['encoded_values_list'][:]
        decoded_values_list = h5file['decoded_values_list'][:]
        sigma_list = h5file['sigma_list'][:]

        # Simulation parameters
        parameters = {key: h5file.attrs[key] for key in h5file.attrs}

        print(parameters)
    
    pot_fn = None
    if parameters['potential'] == 'wolfeschlegel':
        pot_fn = wolfeschlegel_potential
    
    Tdeposite = parameters['Tdeposite']
    dt = parameters['dt']
    NstepsDeposite = int(Tdeposite / dt)

    findTSTime(trajectory)

    main_plot(pot_fn, parameters['n'], trajectory, qs, encoder_params_list, scale_factors, gradient_directions, encoded_values_list, decoded_values_list, sigma_list, parameters['height'], NstepsDeposite, parameters['T'])

if __name__ == '__main__':
    load_data()

