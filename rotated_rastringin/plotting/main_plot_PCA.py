import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import math
import jax
from jax import vmap
import jax.numpy as jnp
import haiku as hk
import re
from collections import defaultdict



#####################
# SUMMING GAUSSIANS #
#####################

# This function works but sums too early to be useful for plotting
def SumGaussiansPCA(q, qs, eigenvectors, choose_eigenvalue, height, sigma):
    choose_eigenvalue_ = jnp.expand_dims(choose_eigenvalue, axis=1)
    qs_jnp = jnp.array(qs)

    # V = np.empty((q.shape[0], 1))
    V = 0.0
    for i in range(q.shape[0]): # loop over N
        x_minus_centers = q[i:i + 1] - qs_jnp  # N * M
        print(f'x_minus_centers: {x_minus_centers}')
        x_minus_centers = jnp.expand_dims(x_minus_centers, axis=1)  # N * 1 * M
        x_projected = jnp.matmul(x_minus_centers, eigenvectors)
        x_projected_ = x_projected * choose_eigenvalue_
        x_projected_sq_sum = jnp.sum((x_projected_) ** 2, axis=(-2, -1))  # N

        V += jnp.sum(height * jnp.exp(-jnp.expand_dims(x_projected_sq_sum, axis=1) / 2 / sigma ** 2), axis=0)
        # V += height * jnp.exp(-jnp.expand_dims(x_projected_sq_sum, axis=1) / 2 / sigma ** 2)

    return V

# This function and SumGaussiansPCA do the same thing (maybe, there have been changes)
def JSumGaussiansPCA(q, qs, eigenvectors, choose_eigenvalue, height, sigma):
    choose_eigenvalue_ = jnp.expand_dims(choose_eigenvalue, axis=1)
    qs_jnp = jnp.array(qs)

    def calc_exp(q_one):
        x_minus_centers = q_one - qs_jnp  # N * M
        x_minus_centers = jnp.expand_dims(x_minus_centers, axis=1)  # N * 1 * M
        x_projected = jnp.matmul(x_minus_centers, eigenvectors)
        x_projected_ = x_projected * choose_eigenvalue_
        x_projected_sq_sum = jnp.sum((x_projected_) ** 2, axis=(-2, -1))  # N
        return jnp.sum(height * jnp.exp(-jnp.expand_dims(x_projected_sq_sum, axis=1) / 2 / sigma ** 2), axis=0)

    vmap_calc_exp = vmap(calc_exp, in_axes=(0))
    result = vmap_calc_exp(q)
    result = jnp.sum(result, axis=-1)

    return result

# This function and GaussiansPCA do the same thing
# OLD: does not correctly handle multiple eigenvectors chosen together. 
# def JGaussiansPCA(q, qs, eigenvectors, choose_eigenvalue, height, sigma_list):
#     choose_eigenvalue_ = jnp.expand_dims(choose_eigenvalue, axis=1)
#     qs_jnp = jnp.array(qs)

#     def calc_exp(q_one):
#         x_minus_centers = q_one - qs_jnp  # N * M
#         x_minus_centers = jnp.expand_dims(x_minus_centers, axis=1)  # N * 1 * M
#         x_projected = jnp.matmul(x_minus_centers, eigenvectors)
#         x_projected_ = x_projected * choose_eigenvalue_
#         x_projected_sq = jnp.sum((x_projected_) ** 2, axis=(1))
#         another_exponent = - x_projected_sq / 2 / sigma_list ** 2
#         return jnp.sum(height * jnp.exp(another_exponent) * choose_eigenvalue, axis=0)

#         # x_projected_sq_sum = jnp.sum((x_projected_) ** 2, axis=(-2, -1))  # N
#         # return jnp.sum(height * jnp.exp(-jnp.expand_dims(x_projected_sq_sum, axis=1) / 2 / sigma ** 2), axis=0)

#     vmap_calc_exp = vmap(calc_exp, in_axes=(0))
#     result = vmap_calc_exp(q)

#     return result

# This function and GaussiansPCA do NOT do the same thing -- this one not updated for multiple sigmas
def JGaussiansPCA(q, qs, eigenvectors, choose_eigenvalue, height, sigma_list):
    # print(sigma_list)
    # print(choose_eigenvalue)
    qs_jnp = jnp.array(qs)

    def calc_exp(q_one):
        x_minus_centers = q_one - qs_jnp  # N * M
        x_minus_centers = jnp.expand_dims(x_minus_centers, axis=1)  # N * 1 * M
        x_projected = jnp.matmul(x_minus_centers, eigenvectors)
        x_projected_ = x_projected * choose_eigenvalue
        x_projected_sq = jnp.sum((x_projected_) ** 2, axis=(1))
        another_exponent = - x_projected_sq / 2 / (sigma_list ** 2)
        return height * jnp.exp(jnp.sum(another_exponent, axis=-1))

    vmap_calc_exp = vmap(calc_exp, in_axes=(0))
    result = vmap_calc_exp(q)

    return result

def GaussiansPCA(q, qs, eigenvectors, choose_eigenvalue, height, sigma):
    choose_eigenvalue_ = np.expand_dims(choose_eigenvalue, axis=1)

    V = np.empty((q.shape[0], 1))
    for i in range(q.shape[0]):
        x_minus_centers = q[i:i + 1] - qs  # N * M
        x_minus_centers = np.expand_dims(x_minus_centers, axis=1)  # N * 1 * M
        x_projected = np.matmul(x_minus_centers, eigenvectors)
        x_projected_ = x_projected * choose_eigenvalue_
        x_projected_sq = jnp.sum((x_projected_) ** 2, axis=(1))
        another_exponent = - x_projected_sq / 2 / sigma_list ** 2
        V[i] = jnp.sum(height * jnp.exp(another_exponent) * choose_eigenvalue, axis=0)

    return V

jax_PCASumGaussian = jax.grad(SumGaussiansPCA)
jax_PCASumGaussian_jit = jax.jit(jax_PCASumGaussian)

def PCAGradGaussians(q, qs, eigenvectors, choose_eigenvalue, height, sigma):
    print(f'jax_PCASumGaussian_jit._cache_size: {jax_PCASumGaussian_jit._cache_size()}')
    # scale_factors_jnp = jnp.array(scale_factors)        # N * D
    # sigma_list_jnp = jnp.array(sigma_list)

    q_jnp = jnp.array(q)
    qs_jnp = jnp.array(qs)
    eigenvectors_jnp = jnp.array(eigenvectors)
    choose_eigenvalue_jnp = jnp.array(choose_eigenvalue)

    grad = jax.grad(lambda x: jnp.sum(SumGaussiansPCA(x, qs_jnp, eigenvectors_jnp, choose_eigenvalue_jnp, height, sigma)))(q_jnp)
    # return jnp.zeros(grad.shape)
    # print(f'gradients.shape: {(jnp.sum(grad, axis=0)).shape}')
    return jnp.sum(grad, axis=0)


############
# PLOTTING #
############

def wolfeschlegel_potential(qx, qy, qn):
    V = 10 * (qx**4 + qy**4 - 2 * qx**2 - 4 * qy**2 + qx * qy + 0.2 * qx + 0.1 * qy + jnp.sum(qn**2))
    return V

def rastringin_potential(qx, qy, qn):
    # TODO: fix qn
    x = np.array([qx, qy])
    A = 10
    B = 0.5
    d = x.shape[0]
    xsq = B*jnp.power(x,2)
    wave = jnp.cos(2*np.pi*x)
    return A*d + jnp.sum(xsq - A*wave,axis=0)

def full_rotation_matrix(angle, dim):
    def givens_rotation_matrix(i, j, theta, dim):
        rotation = np.eye(dim)
        rotation[i, i] = np.cos(theta)
        rotation[j, j] = np.cos(theta)
        rotation[i, j] = -np.sin(theta)
        rotation[j, i] = np.sin(theta)
        return rotation
    
    rotation = np.eye(dim)
    for i in range(0, dim - 1, 2):
        j = i + 1
        rotation_step = givens_rotation_matrix(i, j, angle, dim)
        rotation = rotation @ rotation_step
    return rotation

def rotated_rastringin_potential(qx, qy, qn, angle=(np.pi / 4)):
    X_flat = qx.flatten()
    Y_flat = qy.flatten()
    q = np.vstack((X_flat, Y_flat))
    dim = len(qn)   # extra dimensions to account for later
    rotation = full_rotation_matrix(angle, 2)   # only thinking about the x and y dimensions
    q_rotated = rotation @ q
    A = 10
    B = 0.5
    d = q_rotated.shape[0]
    xsq = B * jnp.power(q_rotated, 2)
    wave = jnp.cos(2 * np.pi * q_rotated) + jnp.cos(0) * (dim - 2)
    result = A * d + jnp.sum(xsq - A * wave, axis=0)
    print(f'result.shape: {result.shape}')
    print(f'result resshape.shape: {result.reshape(qx.shape).shape}')
    return result.reshape(qx.shape)

def rotated_rastringin_potential_q(q, angle=(np.pi / 4)):
    need_swap = False
    if q.ndim > 1:
        need_swap = True
        # Move the last dimension to the first position
        q = np.moveaxis(q, -1, 0)
    dim = len(q)
    rotation = full_rotation_matrix(angle, dim)
    # print(f'rotation: {rotation}')
    q_rotated = rotation @ q
    A = 10
    B = 0.5
    d = q_rotated.shape[0]
    xsq = B * jnp.power(q_rotated, 2)
    wave = jnp.cos(2 * np.pi * q_rotated)
    result = A * d + jnp.sum(xsq - A * wave, axis=0)
    if need_swap:
        result = np.moveaxis(result, 0, -1)
    return result



    # q = np.array([qx, qy])
    # print(f'q: {q}')
    # dim = len(qn) + 2
    # print(f'dim: {dim}')
    # rotation = full_rotation_matrix(angle, 2)
    # q_rotated_x = rotation @ qx
    # q_rotated_y = rotation @ qy
    # print(f'q_rotated: {q_rotated}')
    # A = 10
    # B = 0.5
    # # d = q_rotated.shape[0]
    # d = dim
    # xsq = B * jnp.power(q_rotated_x, 2) + B * jnp.power(q_rotated_y, 2)
    # # all the qn's are set to be 0
    # wave = jnp.cos(2 * np.pi * q_rotated_x) + jnp.cos(2 * np.pi * q_rotated_y) + jnp.cos(0) * (dim - 2)
    # return A * d + jnp.sum(xsq - A * wave, axis=0)

def main_plot(potential, potential_name, n, trajectory, qs, eigenvectors_list, choose_eigenvalue_list, gradient_directions, sigma, sigma_list, decay_sigma, height, NstepsDeposite, T, threshold, well_indices):
    filename = None

    xmin = -6
    xmax = 6
    ymin = -6
    ymax = 6

    cmap = plt.get_cmap('plasma')

    xx = np.linspace(xmin, xmax, 200)
    yy = np.linspace(ymin, ymax, 200)

    [X, Y] = np.meshgrid(xx, yy)  # 100*100
    W = potential(X, Y, np.zeros(n))
    W1 = W.copy()
    W1 = W1.at[W > 300].set(float('nan'))  # Use JAX .at[] method

    # fig = plt.figure(figsize=(14,6))
    fig = plt.figure(figsize=(10.5,4.5))
    ax1 = fig.add_subplot(1, 3, 1)
    contourf_ = ax1.contourf(X, Y, W1, levels=29)
    plt.colorbar(contourf_)

    indices = np.arange(trajectory.shape[0])
    ax1.scatter(trajectory[:, 0, 0], trajectory[:, 0, 1], c=indices, cmap=cmap)
    ax1.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    plt.title('Trajectory')

    num_points = X.shape[0] * X.shape[1]

    # Initialize an empty list to store the results
    results = []

    # Gs = JSumGaussian(jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1), jnp.zeros((num_points, n))], axis=1), qs, encoder_params_list, scale_factors, h=height, sigma=sigma)
    points = jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1), jnp.zeros((num_points, n))], axis=1)

    # print("TEST0")
    # result1 = jnp.sum(JSumGaussiansPCA(points[0], qs, eigenvectors_list, choose_eigenvalue_list, height=height, sigma=sigma))
    # print(f'result1: {result1}')
    # result2 = SumGaussiansPCA(jnp.array(points[0]), qs, eigenvectors_list, choose_eigenvalue_list, height=height, sigma=sigma)
    # print(f'result2: {result2}')
    # result3 = jnp.sum(JGaussiansPCA(points[0], qs, eigenvectors_list, choose_eigenvalue_list, height=height, sigma=sigma), axis=0)
    # print(f'result3: {result3}')
    # result4 = jnp.sum(GaussiansPCA(points[0], qs, eigenvectors_list, choose_eigenvalue_list, height=height, sigma=sigma), axis=0)
    # print(f'result4: {result4}')

    # print("TEST")
    # result1 = JSumGaussiansPCA(points[0:7], [qs[0]], eigenvectors_list[0], choose_eigenvalue_list[0], height=height, sigma=sigma)
    # print(f'result1: {result1}')
    # result2 = SumGaussiansPCA(points[0:7], [qs[0]], eigenvectors_list[0], choose_eigenvalue_list[0], height=height, sigma=sigma)
    # print(f'result2: {result2}')
    # result3 = jnp.sum(JGaussiansPCA(points[0:7], [qs[0]], eigenvectors_list[0], choose_eigenvalue_list[0], height=height, sigma=sigma), axis=0)
    # print(f'result3: {result3}')

    # print("TEST2")
    # result1 = JSumGaussiansPCA(points[0:7], qs, eigenvectors_list, choose_eigenvalue_list, height=height, sigma=sigma)
    # print(f'result1: {result1}')
    # result2 = SumGaussiansPCA(points[0:7], qs, eigenvectors_list, choose_eigenvalue_list, height=height, sigma=sigma)
    # print(f'result2: {result2}')
    # result3 = jnp.sum(JGaussiansPCA(points[0:7], qs, eigenvectors_list, choose_eigenvalue_list, height=height, sigma=sigma), axis=0)
    # print(f'result3: {result3}')

    # Gaussian values not summed over time steps:
    for i, (center, eigenvectors, choose_eigenvalue, sigmas) in enumerate(zip(qs, eigenvectors_list, choose_eigenvalue_list, sigma_list)):
        result = JGaussiansPCA(points, jnp.array([center]), eigenvectors, choose_eigenvalue, height=height, sigma_list=np.array([sigmas]))
        results.append(result)
    results = jnp.array(results)
    # print(f'results.shape: {results.shape}')
    results = jnp.sum(results, axis=-1)         # TODO: check if this is correct
    Gs = jnp.sum(results, axis=0)
    Gs = Gs.reshape(X.shape)
    Sum = Gs + W1

    ax2 = fig.add_subplot(1, 3, 2)
    cnf2 = ax2.contourf(X, Y, Gs.reshape(X.shape[0], X.shape[1]), levels=29)
    plt.colorbar(cnf2)
    indices = np.arange(qs.shape[0])
    ax2.scatter(qs[:, 0], qs[:, 1], c=indices, cmap=cmap)
    ax2.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    plt.title('Gaussian Bias')
    # ax2.quiver(qs[:, 0], qs[:, 1])
    # ax2.quiver(qs[:, 0], qs[:, 1], gradient_directions[:, 0], gradient_directions[:, 1])
    # ax2.axis('equal')
    indices = np.arange(trajectory.shape[0])
    # ax2.scatter(trajectory[:, 0, 0], trajectory[:, 0, 1], c=indices, cmap=cmap, alpha=0.1)

    # ax2.scatter(trajectory[:, 0], trajectory[:, 1], c=indices, cmap=cmap)
    ax3 = fig.add_subplot(1, 3, 3)
    cnf3 = ax3.contourf(X, Y, Sum, levels=29)

    ax3.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

    plt.subplots_adjust(wspace=0.25)

    # fig.colorbar(contourf_)
    plt.title('Biased Potential')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

    ###################
    # PLOT TRAJECTORY #
    ###################

    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 1, 1)
    # contourf_ = ax1.contourf(X, Y, W1, levels=29)
    # plt.colorbar(contourf_)

    # indices = np.arange(trajectory.shape[0])
    # ax1.scatter(trajectory[:, 0, 0], trajectory[:, 0, 1], c=indices, cmap=cmap)

    # num_points = X.shape[0] * X.shape[1]

    # # Initialize an empty list to store the results
    # # results = []

    # # Gs = JSumGaussian(jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1), jnp.zeros((num_points, n))], axis=1), qs, encoder_params_list, scale_factors, h=height, sigma=sigma)
    # points = jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1), jnp.zeros((num_points, n))], axis=1)

    # # Gaussian values not summed over time steps:
    # # print(f'centers: {qs}')
    # # print(f'eigenvectors: {eigenvectors_list}')
    # # print(f'choose_eigenvalue_list: {choose_eigenvalue_list}')
    # # for i, (center, eigenvectors, choose_eigenvalue) in enumerate(zip(qs, eigenvectors_list, choose_eigenvalue_list)):
    # #     result = JGaussiansPCA(points, jnp.array([center]), eigenvectors, choose_eigenvalue, height=height, sigma=sigma)
    # #     results.append(result)
    # # results = jnp.array(results)
    # # # print(f'results.shape: {results.shape}')
    # # results = jnp.sum(results, axis=-1)         # TODO: check if this is correct
    # # Gs = jnp.sum(results, axis=0)
    # # Gs = Gs.reshape(X.shape)
    # # Sum = Gs + W1

    # ax1.set(xlim=(-3, 3), ylim=(-3, 3))
    # plt.show()


    ################
    # PLOT CENTERS #
    ################

    # fig = plt.figure()
    # ax2 = fig.add_subplot(1, 1, 1)
    # cnf2 = ax2.contourf(X, Y, Gs.reshape(X.shape[0], X.shape[1]), levels=29)
    # plt.colorbar(cnf2)
    # indices = np.arange(qs.shape[0])
    # ax2.scatter(qs[:, 0], qs[:, 1], c=indices, cmap=cmap)
    # # ax2.quiver(qs[:, 0], qs[:, 1])
    # # ax2.quiver(qs[:, 0], qs[:, 1], gradient_directions[:, 0], gradient_directions[:, 1])
    # # ax2.axis('equal')
    # indices = np.arange(trajectory.shape[0])
    # # ax2.scatter(trajectory[:, 0, 0], trajectory[:, 0, 1], c=indices, cmap=cmap, alpha=0.1)

    # ax2.set(xlim=(-3, 3), ylim=(-3, 3))
    # plt.show()

    ######################
    # PLOT FINAL SURFACE #
    ######################

    # fig = plt.figure()
    # # ax2.scatter(trajectory[:, 0], trajectory[:, 1], c=indices, cmap=cmap)
    # ax3 = fig.add_subplot(1, 1, 1)
    # cnf3 = ax3.contourf(X, Y, Sum, levels=29)

    # # fig.colorbar(contourf_)
    # plt.title('Local PCA dynamics')
    # if filename is None:
    #     plt.show()
    # else:
    #     plt.savefig(filename)

    # ax3.set(xlim=(-3, 3), ylim=(-3, 3))
    # plt.show()

    ##################
    # PLOT POTENTIAL #
    ##################
    fig_pot, ax_pot = plt.subplots(figsize=(12,5))
    print(f'trajectory.shape: {trajectory.shape}')
    pot_trajectory = jnp.sum(trajectory, axis=1)
    np_pot_list = rotated_rastringin_potential_q(pot_trajectory)
    N = len(trajectory)
    indices = np.arange(N)

    ax_pot.plot(indices, np_pot_list)
        # ax_sigma.scatter(indices, np_sigma_list[:, k], label=f'PCA Component {k+1}')
    for i, well_index in enumerate(well_indices):
        if i == 0:
            ax_pot.axvline(x=well_index, color='r', linestyle='--', label='New Well Discovered')
        else:
            ax_pot.axvline(x=well_index, color='r', linestyle='--')
    # ax_sigma.axvline(x=timings[2]/100, color='b', linestyle='--', label=f'Lower Right Well at Iter {math.ceil(timings[2]/100)}')
    # ax_sigma.axvline(x=timings[1]/100, color='g', linestyle='--', label=f'Lower Left Well at Iter {math.ceil(timings[1]/100)}')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Potential value', fontsize=12)
    # plt.title('Gaussian sigma values', fontsize=14)
    # plt.legend()
    # ax_sigma.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.tight_layout()  # Adjusts the plot to make room for the legend
    plt.show()


    ###############
    # PLOT SIGMAS #
    ###############
    fig_sigma, ax_sigma = plt.subplots(figsize=(12,5))
    np_sigma_list = np.array(sigma_list)
    N, K = np_sigma_list.shape
    indices = np.arange(N)
    for k in range(K):
        valid_indices = np.where(choose_eigenvalue_list[:, k] != 0)[0]
        valid_sigma = np_sigma_list[valid_indices, k]
        if len(valid_indices) > 0:
            ax_sigma.scatter(valid_indices, valid_sigma, label=f'PCA Component {k+1}')
        # ax_sigma.scatter(indices, np_sigma_list[:, k], label=f'PCA Component {k+1}')
    for i, well_index in enumerate(well_indices):
        if i == 0:
            ax_sigma.axvline(x=well_index/100, color='r', linestyle='--', label='New Well Discovered')
        else:
            ax_sigma.axvline(x=well_index/100, color='r', linestyle='--')
    # ax_sigma.axvline(x=timings[2]/100, color='b', linestyle='--', label=f'Lower Right Well at Iter {math.ceil(timings[2]/100)}')
    # ax_sigma.axvline(x=timings[1]/100, color='g', linestyle='--', label=f'Lower Left Well at Iter {math.ceil(timings[1]/100)}')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Sigma value', fontsize=12)
    plt.title('Gaussian sigma values', fontsize=14)
    # plt.legend()
    ax_sigma.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()  # Adjusts the plot to make room for the legend
    plt.show()


    ##################
    # PLOT GAUSSIANS #
    ##################
    Ncenter = int(NstepsDeposite / 2)
    reshaped_trajectory = trajectory[:-1].reshape(T, NstepsDeposite, 1, 2 + n)  # Assuming 10 iterations and calculating steps
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
        ax_gaussian.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        plt.show()


# def plot_performance(X, Y, W1, trajectory, qs, encoder_params_list, scale_factors, gradient_directions, encoded_values_list, decoded_values_list, sigma_list):


    
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
        print(f"The first time step in the UPPER RIGHT well: {first_occurrence_index_1}")
    else:
        print("There are no time steps in the UPPER RIGHT well.")

    # Lower Left Quadrant
    indices_2 = np.where((x_dimension < -0.1) & (y_dimension < -0.1))[0]
    # Check if any such indices exist
    if indices_2.size > 0:
        # Get the first occurrence
        first_occurrence_index_2 = indices_2[0]
        print(f"The first time step in the LOWER LEFT well: {first_occurrence_index_2}")
    else:
        print("There are no time steps in the LOWER LEFT well.")

    # Lower Right Quadrant
    indices_3 = np.where((x_dimension > 0.1) & (y_dimension < -0.1))[0]
    # Check if any such indices exist
    if indices_3.size > 0:
        # Get the first occurrence
        first_occurrence_index_3 = indices_3[0]
        print(f"The first time step in the LOWER RIGHT well: {first_occurrence_index_3}")
    else:
        print("There are no time steps in the LOWER RIGHT well.")

    return first_occurrence_index_1, first_occurrence_index_2, first_occurrence_index_3


def count_wells(trajectory, leeway=0.3, min_samples=10):
    # TODO: why is this line here? is it to get rid of an extra axis?
    trajectory = np.sum(trajectory, axis=1)
    # REVERSE or previous angle
    angle = -np.pi / 4
    need_swap = False
    if trajectory.ndim > 1:
        need_swap = True
        # Move the last dimension to the first position
        trajectory = np.moveaxis(trajectory, -1, 0)
    dim = len(trajectory)
    rotation = full_rotation_matrix(angle, dim)
    # print(f'rotation: {rotation}')
    trajectory = rotation @ trajectory
    if need_swap:
        trajectory = np.moveaxis(trajectory, 0, -1)

    # Define the well centers based on the given criteria
    well_centers = np.arange(-15.5, 16, 1)

    
    # Initialize a dictionary to store the count of samples in each well
    well_counts = defaultdict(int)
    # Initialize a dictionary to store the first index of each well
    first_indices = {}

    # Iterate through the trajectory data
    for idx, point in enumerate(trajectory):
        # Find the nearest well center for each dimension
        well_indices = np.round((point)).astype(int)
        # Calculate the distance from the point to the nearest well center
        distances = np.abs(point - (well_indices))
        if all(distances <= leeway):
            well_indices_tuple = tuple(well_indices)
            well_counts[well_indices_tuple] += 1
            if well_indices_tuple not in first_indices:
                first_indices[well_indices_tuple] = idx

    # Count the number of wells with more than min_samples samples
    counted_wells = sum(1 for count in well_counts.values() if count > min_samples)
    # Get the list of indices of the first point in each well that meets the sample criterion
    well_indices_list = [index for well, index in first_indices.items() if well_counts[well] > min_samples]

    return counted_wells, well_indices_list


def count_components_chosen(choose_eigenvalue):
    print(choose_eigenvalue.shape)
    dim = choose_eigenvalue.shape[-1]
    summed = jnp.sum(choose_eigenvalue, axis=-1)
    averaged = jnp.average(summed)
    print(f'dim: {dim}')
    print(f'Average Components chosen: {averaged}')

def load_data(filename):
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

    with h5py.File(filename, 'r') as h5file:

        # Data from the simulation
        trajectory = h5file['trajectory'][:]
        qs = h5file['qs'][:]
        eigenvectors = h5file['eigenvectors'][:]
        eigenvalues = h5file['eigenvalues'][:]
        choose_eigenvalue = h5file['choose_eigenvalue'][:]
        gradient_directions = h5file['gradient_directions'][:]
        sigma_list = h5file['sigma_list'][:]

        # Simulation parameters
        parameters = {key: h5file.attrs[key] for key in h5file.attrs}

        print(parameters)
    
    pot_fn = None
    if parameters['potential'] == 'wolfeschlegel':
        pot_fn = wolfeschlegel_potential
    if parameters['potential'] == 'rosenbrock':
        pot_fn = rosenbrock_potential
    if parameters['potential'] == 'rastringin':
        pot_fn = rastringin_potential
    if parameters['potential'] == 'rotated_rastringin':
        pot_fn = rotated_rastringin_potential
    
    Tdeposite = parameters['Tdeposite']
    dt = parameters['dt']
    NstepsDeposite = int(Tdeposite / dt)

    num_wells, well_indices = count_wells(trajectory)

    count_components_chosen(choose_eigenvalue)

    main_plot(pot_fn, parameters['potential'], parameters['n'], trajectory, qs, eigenvectors, choose_eigenvalue, gradient_directions, parameters['sigma'], sigma_list, parameters['decay_sigma'], parameters['height'], NstepsDeposite, parameters['T'], parameters['threshold'], well_indices)

    return num_wells

if __name__ == '__main__':

    # filename = f'../results/run_n1steepdecay_1.h5'
    # load_data(filename)

    results = []
    for i in range(1):
        # filename = f'../PCA_results/run_rotras_n8_{i+11}.h5'
        # filename = f'../PCA_results/run_rotras_n28_{i+2}.h5'
        filename = f'../PCA_results/run_rotras_n48_{i+41}.h5'
        results.append(load_data(filename))
    for result in results:
        print(f'{result}')

