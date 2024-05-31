import time
import random
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import vmap
import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tqdm import tqdm
from matplotlib import colormaps

jax.config.update("jax_debug_nans", True)

def Psi(dist):

    sigma = 1
    epsilon = 1

    return 4*epsilon*(sigma**12/dist**12 - sigma**6/dist**6)

def PsiSq(distSq):

    sigma = 1
    epsilon = 1

    return 4*epsilon*(sigma**12/distSq**6 - sigma**6/distSq**3)

def GradPsi(atom1, atom2):
    sigma = 1
    epsilon = 1

    distSq = np.sum((atom1-atom2)**2)
    DPsi_DdistSq = 4*epsilon*(-6*sigma**12/distSq**7 + 3*sigma**6/distSq**4)
    return 2*(atom1-atom2)*DPsi_DdistSq

def LJpotential(r): ## r size 1*M M is a multiple of 3
    M = r.shape[1]
    Natoms = M // 3
    V = 0
    for i in range(Natoms-1):
        for j in range(i+1, Natoms):
            atom1 = r[0, i*3:i*3+3]
            atom2 = r[0, j*3:j*3+3]
            V += PsiSq(np.sum((atom1-atom2)**2))
    return V

def GradLJpotential(r): ## r size 1*M M is a multiple of 3
    M = r.shape[1]
    grad = np.zeros((1,M))
    Natoms = M // 3

    for i in range(Natoms):
        lst = np.arange(Natoms).tolist()
        lst.remove(i)
        # print(lst)
        for j in lst:
            atom1 = r[0, i * 3:i * 3 + 3]
            atom2 = r[0, j * 3:j * 3 + 3]
            grad[0,i * 3:i * 3 + 3] += GradPsi(atom1, atom2)
    return grad

def disconnects_graph(adjacency_list, edge):
    # Perform depth-first search to check connectivity after edge removal
    visited = set()
    stack = [0]
    visited.add(0)
    while stack:
        node = stack.pop()
        for neighbor in adjacency_list[node]:
            if (node, neighbor) != edge and (neighbor, node) != edge and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)

    # If any node is not visited, the graph is disconnected
    return len(visited) != len(adjacency_list)

def get_pairwise_distances(x):
    Natoms = int(x.shape[-1] / 3)
    x = np.reshape(x, (Natoms, 3))
    all_diffs = np.expand_dims(x, axis=1) - np.expand_dims(x, axis=0) # N * N * M
    # diffs = np.array([all_diffs[i][j] for [i,j] in pairs])
    pairwise_distances = np.sqrt(np.sum(all_diffs**2, axis=-1)) # N * N

    pairwise_distances = pairwise_distances[np.triu_indices(Natoms, 1)]

    return pairwise_distances

def Jget_pairwise_distances(x):
    Natoms = int(x.shape[-1] / 3)
    x = jnp.reshape(x, (Natoms, 3))
    all_diffs = jnp.expand_dims(x, axis=1) - jnp.expand_dims(x, axis=0) # N * N * M
    sq_diffs = jnp.power(all_diffs, 2.)
    sum_sq_diffs = jnp.sum(sq_diffs, axis=-1)
    pairwise_distances = jnp.sqrt(sum_sq_diffs) # N * N
    pairwise_distances = pairwise_distances[jnp.triu_indices(Natoms, 1)]

    return pairwise_distances

def PCA(data):  # datasize: N * dim
    # Step 4.0: Convert to distances
    data_pw = np.array([np.array(get_pairwise_distances(data[i])).flatten() for i in range(len(data))])

    # Step 4.1: Compute the mean of the data
    data_z = data_pw  # bs*3

    mean_vector = np.mean(data, axis=0, keepdims=True)
    std_vector = np.std(data, axis=0, keepdims=True)

    # Step 4.2: Center the data by subtracting the mean
    centered_data = (data - mean_vector)

    # Step 4.3: Compute the covariance matrix of the centered data
    covariance_matrix = np.cov(data_z, rowvar=False)

    # Step 4.4: Perform eigendecomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 4.5: Sort the eigenvectors based on eigenvalues (descending order)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 4.6: Choose the number of components (optional)
    k = 3  # Set the desired number of components

    # Step 4.7: Retain the top k components
    selected_eigenvectors = eigenvectors[:, 0:k]

    return mean_vector, selected_eigenvectors
    # return mean_vector, std_vector, selected_eigenvectors, eigenvalues

def AE(data):
    data_pw = np.array([get_pairwise_distances(data[i]) for i in range(len(data))])

    mean_vector = np.mean(data, axis=0, keepdims=True)
    std_vector = np.std(data, axis=0, keepdims=True)
    
    input_dim = data_pw.shape[1]
    encoding_dim = 3 # Set the desired encoding dimension
    intermediate_dim = 64 # Set the width of the intermediate layer

    # Define the Autoencoder architecture
    input_layer = Input(shape=(input_dim,))
    encoder = Sequential([Dense(intermediate_dim, activation='relu'),
                        Dense(intermediate_dim, activation='relu'),
                        Dense(encoding_dim)])
    encoded = encoder(input_layer)
    decoder = Sequential([Dense(intermediate_dim, activation='relu'),
                        Dense(intermediate_dim, activation='relu'),
                        Dense(input_dim)])
    # NOTE: the absolute value is an important aspect of the decoder
    decoded = tf.math.abs(decoder(encoded))

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train the Autoencoder
    autoencoder.fit(data_pw, data_pw, epochs=300, batch_size=32, shuffle=True, validation_split=0.2)
    
    # Get out base vectors to plot
    base_vectors = np.identity(input_dim)
    encoded_base_vectors = encoder.predict(base_vectors)

    ae_comps = encoded_base_vectors.T[:,0:input_dim]

    ae_comp_norms = la.norm(ae_comps.T, axis=0)

    ae_comps_normalized = (ae_comps / ae_comp_norms[:, np.newaxis]).T

    centered_data = (data - mean_vector)
    covariance_matrix = np.cov(data_pw, rowvar=False)

    # variance of vector v is v dot (Sigma v) where Sigma is the covariance matrix
    # source: https://math.stackexchange.com/questions/4806778/variance-in-the-direction-of-a-unit-vector-intuition-behind-formula#:~:text=Say%20we%20have%20a%20set,Σ%20is%20the%20covariance%20matrix.
    # TODO: find better source for this
    ae_variance = np.array([np.dot(ae_comps_normalized[:, i], np.dot(covariance_matrix, ae_comps_normalized[:, i]))for i in range(len(ae_comps_normalized[0]))])


    sorted_indices = np.argsort(ae_variance)[::-1]
    ae_variance = ae_variance[sorted_indices]
    ae_comps_normalized = ae_comps_normalized[:, sorted_indices]

    # Graham-Schmidt
    # loop through each vector in ae_comps_normalized
    #   project out each previous vector, if there are any
    GS_eigenvectors = np.zeros((len(ae_comps_normalized), len(ae_comps_normalized[0])))
    for i in range(len(ae_comps_normalized)):
        cur_vec = ae_comps_normalized[i]
        for j in range(i-1, -1, -1):
            prev_vec = GS_eigenvectors[j]
            # subtract the projection of cur_vec onto prev_vec
            cur_vec = cur_vec - prev_vec * np.dot(cur_vec, prev_vec) / np.dot(prev_vec, prev_vec)
        GS_eigenvectors[i] = cur_vec
    # calcuate the variances (eigenvalues) of the new vectors
    GS_eigenvalues = np.array([np.dot(GS_eigenvectors[:, i], np.dot(covariance_matrix, GS_eigenvectors[:, i]))for i in range(len(GS_eigenvectors[0]))])
    
    # Sort the eigenvectors and eigenvalues based on the GS variance/eigenvalues (descending order)
    sorted_indices = np.argsort(GS_eigenvalues)[::-1]
    GS_eigenvalues = GS_eigenvalues[sorted_indices]
    GS_eigenvectors = GS_eigenvectors[:, sorted_indices]
    ae_variance = ae_variance[sorted_indices]
    ae_comps_normalized = ae_comps_normalized[:, sorted_indices]

    # the variances are the equivalent of the eigenvalues from PCA
    return mean_vector, std_vector, ae_comps_normalized, ae_variance, GS_eigenvectors, GS_eigenvalues

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

@jax.jit
def JSumGaussianPW(pw_x, pw_centers, eigenvectors, orth_eigenvectors, h, sigma):
    # x: 1 * M
    # centers: N * M
    # eigenvectors: N * M * k
    # orth_eigenvectors: N * M * k
    env_sigma = sigma

    # Vectorize the single computation over the batch dimension N
    vmap_sum_gaussian_pw = vmap(SumGaussianPW_single, in_axes=(None, 0, 0, 0, None, None, None))

    total_bias = vmap_sum_gaussian_pw(pw_x, pw_centers, eigenvectors, orth_eigenvectors, h, sigma, env_sigma)  # N

    # TODO??: Normalize AND plot the size of normalization factor
    # Track the new sigma values that we calculate and use that for all calcs

    # TODO: variable sigma's dependent on the size of the eigenvalue. Larger eigenvalue = larger Gaussian
    # NEED that as it might potentially help the AE specifically

    return jnp.sum(total_bias)  # scalar

def JSumGaussianPWUnsummed(pw_x, pw_centers, eigenvectors, orth_eigenvectors, h, sigma):
    # x: 1 * M
    # centers: N * M
    # eigenvectors: N * M * k
    # orth_eigenvectors: N * M * k
    env_sigma = sigma

    # Vectorize the single computation over the batch dimension N
    vmap_sum_gaussian_pw = vmap(SumGaussianPW_single, in_axes=(None, 0, 0, 0, None, None, None))

    total_bias = vmap_sum_gaussian_pw(pw_x, pw_centers, eigenvectors, orth_eigenvectors, h, sigma, env_sigma)  # N

    # TODO??: Normalize AND plot the size of normalization factor
    # Track the new sigma values that we calculate and use that for all calcs

    # TODO: variable sigma's dependent on the size of the eigenvalue. Larger eigenvalue = larger Gaussian
    # NEED that as it might potentially help the AE specifically

    return total_bias

def SumGaussianPW(pw_x, pw_centers, eigenvectors, orth_eigenvectors, h, sigma, unsummed=False):
    env_sigma = sigma

    pw_x_minus_centers_original = pw_x - pw_centers                             # N * D
    pw_x_minus_centers = np.expand_dims(pw_x_minus_centers_original, axis=1)    # N * 1 * D
    pw_x_projected = np.matmul(pw_x_minus_centers, eigenvectors)        # N * 1 * k
    pw_x_projected_sq_sum = np.sum(pw_x_projected**2, axis=(-2, -1))    # N
    exps = h*np.exp(-np.expand_dims(pw_x_projected_sq_sum, axis=1)/2/sigma**2)        # N * 1

    # Gaussian in the directions orthogonal to the CVs
    # eigenvectors_orth = np.array([scipy.linalg.qr(eigenvectors[i], mode='economic')[0] for i in range(len(eigenvectors))]) #  N * D * k
    eigenvectors_orth = orth_eigenvectors
    pairdists_projected_orth = np.matmul(pw_x_minus_centers, eigenvectors_orth) * eigenvectors_orth    # N * D * k
    pairdists_projected_sum = np.sum(pairdists_projected_orth, axis=-1)             # N * D
    pairdists_remainder = pw_x_minus_centers_original - pairdists_projected_sum     # N * D

    pairdists_envelope_sq_sum = np.sum(pairdists_remainder**2, axis=-1)   # N

    envelope_exps = h*np.exp(-np.expand_dims(pairdists_envelope_sq_sum, axis=1)/2/env_sigma**2) # N * 1

    # Combine the Gaussians
    exps = exps * envelope_exps # N * 1

    # total_bias = np.sum(exps, axis=0, keepdims=True) # N * 1? NOPE not at all true
    total_bias = np.sum(exps, axis=1) # 1

    # TODO??: Normalize AND plot the size of normalization factor
    # Track the new sigma values that we calculate and use that for all calcs

    # TODO: variable sigma's dependent on the size of the eigenvalue. Larger eigenvalue = larger Gaussian
    # NEED that as it might potentially help the AE specifically

    if unsummed:
        return total_bias
    else:
        return np.sum(total_bias)

def SumGaussian(x, centers_pw, eigenvectors, orth_eigenvectors, h, sigma, unsummed=False):
    # M = total dimensions, 3 * Natoms
    # N = number of iterations
    # k = number of CV dimensions
    # D = reduced dimensions, M - 6
    # x: 1 * M
    # centers: N * M
    # eigenvectors: N * M * k

    pw_x = np.array([get_pairwise_distances(x)])     # 1 * D
    pw_centers = centers_pw                                 # N * D

    return SumGaussianPW(pw_x, pw_centers, eigenvectors, orth_eigenvectors, h, sigma, unsummed)

# This function is never used, I think?
def JSumGaussian(x, centers_pw, eigenvectors, orth_eigenvectors, h, sigma, unsummed=False):
    # x: 1 * M
    # centers: N * M
    # eigenvectors: N * M * k

    pw_x = jnp.array([Jget_pairwise_distances(x)])   # 1 * D
    pw_centers = jnp.array(centers_pw)                      # N * D

    if unsummed:
        return JSumGaussianPWUnsummed(pw_x, pw_centers, eigenvectors, orth_eigenvectors, h, sigma)

    return JSumGaussianPW(pw_x, pw_centers, eigenvectors, orth_eigenvectors, h, sigma)

jax_SumGaussianPW = jax.grad(JSumGaussianPW)
jax_SumGaussianPW_jit = jax.jit(jax_SumGaussianPW)

def GradGaussian(x, centers_pw, eigenvectors, orth_eigenvectors, h, sigma):
    # print(f'jax_SumGaussianPW_jit._cache_size: {jax_VSumGaussianPW_jit._cache_size()}')
    pw_x_jnp = jnp.array([Jget_pairwise_distances(x)])   # 1 * D
    pw_centers_jnp = centers_pw                                 # N * D
    eigenvectors_jnp = jnp.array(eigenvectors)
    orth_eigenvectors_jnp = jnp.array(orth_eigenvectors)

    pw_grad = jax_SumGaussianPW_jit(pw_x_jnp, pw_centers_jnp, eigenvectors_jnp, orth_eigenvectors_jnp, h, sigma)

    # TODO: test that the below code is correct
    N = pw_grad.shape[0]
    M = x.shape[-1]
    Natoms = int(M / 3)
    D = int(Natoms * (Natoms - 1) / 2)

    x_grad = np.zeros((Natoms, 3))
    x_vals = np.reshape(x, (Natoms, 3))

    # TODO: sanity check the signs

    iu = np.triu_indices(Natoms, 1)
    for k in range(N):
        for l in range(D):
            # for j in range(i):
            i = iu[0][l]
            j = iu[1][l]
            rij = pw_x_jnp[0][l] if (pw_x_jnp[0][l] != 0) else 0.000001
            update_vec = pw_grad[k][l] * (x_vals[j] - x_vals[i]) / rij
            x_grad[i] -= update_vec
            x_grad[j] += update_vec

    x_grad = x_grad.flatten()   # M

    return x_grad

def DistSumGaussian(x, centers_pw, eigenvectors, orth_eigenvectors, h, sigma):
    # x: 1 * M
    # centers: N * M
    # eigenvectors: N * M * k
    env_sigma = sigma

    pw_x = np.array([get_pairwise_distances(x)])     # 1 * D
    pw_centers = centers_pw                                 # N * D

    pw_x_minus_centers_original = pw_x - pw_centers                             # N * D
    pw_x_minus_centers_dists = np.linalg.norm(pw_x_minus_centers_original, axis=1) # N
    pw_x_minus_centers = np.expand_dims(pw_x_minus_centers_original, axis=1)    # N * 1 * D
    pw_x_projected = np.matmul(pw_x_minus_centers, eigenvectors)        # N * 1 * k
    pw_x_projected_sq_sum = np.sum(pw_x_projected**2, axis=(-2, -1))    # N
    exps = h*np.exp(-np.expand_dims(pw_x_projected_sq_sum, axis=1)/2/sigma**2)        # N * 1

    # Gaussian in the directions orthogonal to the CVs
    # eigenvectors_orth = np.array([scipy.linalg.qr(eigenvectors[i], mode='economic')[0] for i in range(len(eigenvectors))]) #  N * D * k
    eigenvectors_orth = orth_eigenvectors
    pairdists_projected_orth = np.matmul(pw_x_minus_centers, eigenvectors_orth) * eigenvectors_orth    # N * D * k
    pairdists_projected_sum = np.sum(pairdists_projected_orth, axis=-1)             # N * D
    pairdists_remainder = pw_x_minus_centers_original - pairdists_projected_sum     # N * D

    pairdists_envelope_sq_sum = np.sum(pairdists_remainder**2, axis=-1)   # N

    envelope_exps = h*np.exp(-np.expand_dims(pairdists_envelope_sq_sum, axis=1)/2/env_sigma**2) # N * 1

    # Combine the Gaussians
    exps = exps * envelope_exps # N * 1

    result = [pw_x_minus_centers_dists, exps[:,0]]
    return result

def LD_MetaDyn(M, T, Tdeposite, dt, h, sigma, kbT, ic_method='PCA'):
    # M: dim

    parameters = {
        'M': M,
        'T': T,
        'Tdeposite': Tdeposite,
        'dt': dt,
        'h': h,
        'sigma': sigma,
        'kbT': kbT,
        'ic_method': ic_method
    }

    # Initialization
    r = np.random.randn(1, M)*1
    if M == 30:
        r = np.reshape(ten_atom_init, (1,30))

    if M == 12:
        r = np.reshape(four_atom_init, (1,12))

    if M == 9:
        r = np.reshape(three_atom_init, (1,9))

    Natoms = int(M / 3)

    Nsteps = round(T / dt)
    NstepsDeposite = round(Tdeposite / dt)
    print(NstepsDeposite)
    trajectories4PCA = np.zeros((NstepsDeposite, 1, M))

    rcenters = None
    rcenters_pw = None
    eigenvectors = None
    orth_eigenvectors = None

    LJ_values = []
    LJGrad_values = []
    Gauss_values = []
    GaussGrad_values = []
    LJ_Gauss_values = []
    LJGrad_GaussGrad_values = []
    Gauss_v_dist_values = []
    r_values = []

    LJ_center_values = []
    Gauss_center_values = []

    for i in tqdm(range(Nsteps)):
        print(LJpotential(r))

        LJpot = LJpotential(r)
        LJGrad = np.sum(GradLJpotential(r))
        if rcenters is None:
            Gauss = 0
            GaussGrad = 0
        else:
            Gauss_v_dist = DistSumGaussian(r, rcenters_pw, eigenvectors, orth_eigenvectors, h, sigma)
            Gauss = np.sum(JSumGaussian(r, rcenters_pw, eigenvectors, orth_eigenvectors, h, sigma, unsummed=True))
            GaussGrad = np.sum(GradGaussian(r, rcenters_pw, eigenvectors, orth_eigenvectors, h, sigma))     # NOTE: this is the timing bottleneck to to jit recompiling
            Gauss_v_dist_values.append(Gauss_v_dist)

        LJ_values.append(LJpot)
        LJGrad_values.append(LJGrad)
        Gauss_values.append(Gauss)
        GaussGrad_values.append(GaussGrad)
        LJ_Gauss_values.append(LJpot + Gauss)
        LJGrad_GaussGrad_values.append(LJGrad + GaussGrad)
        r_values.append(r)

        r = next_step(r, rcenters_pw, eigenvectors, orth_eigenvectors, h, sigma, kbT, dt)
        trajectories4PCA[i % NstepsDeposite, :] = r
        if (i + 1) % NstepsDeposite == 0:
            if rcenters is None:
                ### conducting PCA ###
                data = trajectories4PCA

                data = np.squeeze(data, axis=1)
                if ic_method == 'PCA':
                    mean_vector, selected_eigenvectors = PCA(data)
                else:
                    mean_vector, std_vector, selected_eigenvectors, eigenvalues, GS_eigenvectors, GS_eigenvalues = AE(data)
                rcenters = mean_vector
                rcenters_pw = np.array([get_pairwise_distances(rcenters)])
                # TODO: should this be using the GS eigenvectors??
                eigenvectors = np.expand_dims(selected_eigenvectors, axis=0)            # N=1 * D * k
                orth_eigenvectors = jscipy.linalg.qr(eigenvectors, mode='economic')[0]  # N=1 * D * (M-k)?
                # orth_eigenvectors = np.array([jscipy.linalg.qr(eigenvectors[i], mode='economic')[0] for i in range(len(eigenvectors))]) #  N * D * k

                ### reset the PCA dataset
                trajectories4PCA = np.zeros((NstepsDeposite, 1, M))

                LJ_center_values.append(LJpotential(mean_vector))
                Gauss_center_values.append(JSumGaussian(mean_vector, rcenters_pw, eigenvectors, orth_eigenvectors, h, sigma, unsummed=True))
            else:
                ### conducting PCA ###
                data = trajectories4PCA

                data = np.squeeze(data, axis=1)
                if ic_method == 'PCA':
                    mean_vector, selected_eigenvectors = PCA(data)
                else:
                    mean_vector, std_vector, selected_eigenvectors, eigenvalues, GS_eigenvectors, GS_eigenvalues = AE(data)

                rcenters = np.concatenate([rcenters, mean_vector], axis=0)
                rcenters_pw = np.concatenate([rcenters_pw, np.array([get_pairwise_distances(mean_vector)])], axis=0)
                eigenvectors = np.concatenate([eigenvectors, np.expand_dims(selected_eigenvectors, axis=0)], axis=0)
                orth_eigenvectors = np.concatenate([orth_eigenvectors, np.expand_dims(scipy.linalg.qr(eigenvectors[-1], mode='economic')[0], axis=0)], axis=0)
                # orth_eigenvectors = np.array([scipy.linalg.qr(eigenvectors[i], mode='economic')[0] for i in range(len(eigenvectors))])  # TODO: don't recalculate past iterations

                ### reset the PCA dataset
                trajectories4PCA = np.zeros((NstepsDeposite, 1, M))

                LJ_center_values.append(LJpotential(mean_vector))
                Gauss_center_values.append(JSumGaussian(mean_vector, rcenters_pw, eigenvectors, orth_eigenvectors, h, sigma))

    save_data(r_values, rcenters, eigenvectors, LJ_values, Gauss_values, Gauss_v_dist_values, LJGrad_values, GaussGrad_values, Gauss_center_values, LJ_center_values, parameters)

    # analyze_means(rcenters)
    # analyze_dist_gauss(Gauss_v_dist_values)
    # analyze_iter_gauss(Gauss_v_dist_values)
    # analyze_LJ_potential(LJ_values, LJGrad_values, Gauss_values, GaussGrad_values, LJ_Gauss_values, LJGrad_GaussGrad_values)
    # if M == 9:
    #     # show_trajectory_plot(np.array(r_values).reshape((len(r_values), 9)), LJ_values, Gauss_values)
    #     show_trajectory_plot(rcenters, np.array(LJ_center_values), np.reshape(np.array(Gauss_center_values), (len(Gauss_center_values))))

    return None

def save_data(r_values, rcenters, eigenvectors, LJ_values, Gauss_values, Gauss_v_dist_values, LJGrad_values, GaussGrad_values, Gauss_center_values, LJ_center_values, parameters):
    # Directory to save the data to
    # Get the run number and update it
    run_number_file = 'run_number.txt'
    with open(run_number_file, 'r') as f:
        run_number = int(f.read().strip())
    run_number += 1
    with open(run_number_file, 'w') as f:
        f.write(str(run_number))
    # Create a new directory for the current run
    run_dir = os.path.join('simulation_runs', f'run_{run_number}')
    os.makedirs(run_dir)

    np.savez(os.path.join(run_dir, 'trajectory.npz'), r_values=r_values)
    np.savez(os.path.join(run_dir, 'gaussians.npz'), rcenters=rcenters, eigenvectors=eigenvectors)
    np.savez(os.path.join(run_dir, 'traj_energies.npz'), LJ_values=LJ_values, Gauss_values=Gauss_values, Gauss_v_dist_values=np.array(Gauss_v_dist_values, dtype=object))
    np.savez(os.path.join(run_dir, 'traj_gradients.npz'), LJGrad_values=LJGrad_values, GaussGrad_values=GaussGrad_values)
    np.savez(os.path.join(run_dir, 'center_energies.npz'), Gauss_center_values=np.array(Gauss_center_values, dtype=object), LJ_center_values=LJ_center_values)
    np.savez(os.path.join(run_dir, 'parameters.npz'), parameters=np.array(parameters, dtype=object))

def next_LD(r, dt, kbT):

    rnew = r - (GradLJpotential(r)) * dt + np.sqrt(2 * dt *kbT) * np.random.randn(*r.shape)

    return rnew

def next_LD_Gaussian(r, dt, rcenters_pw, eigenvectors, orth_eigenvectors, h, sigma, kbT):
    rnew = r - (GradLJpotential(r) + GradGaussian(r, rcenters_pw, eigenvectors, orth_eigenvectors, h, sigma)) * dt + np.sqrt(2 * dt * kbT) * np.random.randn(*r.shape)

    return rnew

def next_step(r, rcenters_pw, eigenvectors, orth_eigenvectors, h, sigma, kbT, dt):

    if rcenters_pw is None:
        r = next_LD(r, dt, kbT)
    else:
        r = next_LD_Gaussian(r, dt, rcenters_pw, eigenvectors, orth_eigenvectors, h, sigma, kbT)
    return r

def analyze_means(means):
    # TODO: use RELATIVE means, not absolute
    # origin = np.zeros(M)
    dists_to_start = [np.linalg.norm(elem - means[0]) for elem in means]
    plt.plot(dists_to_start)
    plt.xlabel('Iteration')
    plt.ylabel('\'Distance\' From Start')
    plt.show()
    return

def analyze_dist_gauss(Gauss_v_dist_values):
    num_iters = len(Gauss_v_dist_values)
    cmap = colormaps.get_cmap('plasma')
    for i in range(num_iters):
        color_scale_factor = float(i) / num_iters
        if i == 0:
            plt.scatter(Gauss_v_dist_values[i][0], Gauss_v_dist_values[i][1], c=cmap(color_scale_factor), label='Iteration 1')
        elif i == num_iters - 1:
            plt.scatter(Gauss_v_dist_values[i][0], Gauss_v_dist_values[i][1], c=cmap(color_scale_factor), label=f'Iteration {num_iters}')
        else:
            plt.scatter(Gauss_v_dist_values[i][0], Gauss_v_dist_values[i][1], c=cmap(color_scale_factor))
    plt.xlabel('Distance from Gaussian Center')
    plt.ylabel('Magnitude of Felt Gaussian')
    plt.title('Distance From Gaussian Center vs. Felt Gaussian, Shown for Many Iterations')
    plt.legend()
    plt.show()

def analyze_iter_gauss(Gauss_v_dist_values):
    num_iters = len(Gauss_v_dist_values)
    cmap = colormaps.get_cmap('hsv')
    num_gaussians = len(Gauss_v_dist_values[-1][0])
    data = []
    for i in range(num_gaussians):
        this_gaussian_data = [[], []]
        for j in range(i, num_iters):
            if len(Gauss_v_dist_values[j][1]) > i:
                this_gaussian_data[0].append(j)
                this_gaussian_data[1].append(Gauss_v_dist_values[j][1][i])
        data.append(this_gaussian_data)
    
    for i in range(num_gaussians):
        i = num_gaussians - i - 1
        color_scale_factor = float(i) / num_gaussians
        if i == 0:
            plt.plot(data[i][0], data[i][1], c=cmap(color_scale_factor), label='Gaussian 1')
        elif i == num_gaussians - 1:
            plt.plot(data[i][0], data[i][1], c=cmap(color_scale_factor), label=f'Gaussian {num_gaussians}')
        else:
            plt.plot(data[i][0], data[i][1], c=cmap(color_scale_factor))
    plt.xlabel('Iteration')
    plt.ylabel('Magnitude of Felt Gaussian')
    plt.title('Iteration vs. Felt Gaussian, For Each Gaussian')
    plt.legend()
    plt.show()

def analyze_LJ_potential(LJ_values, LJGrad_values, Gauss_values, GaussGrad_values, LJ_Gauss_values, LJGrad_GaussGrad_values):
    plt.plot(LJ_values, label='LJ potential')
    # plt.legend()
    # plt.show()
    # plt.plot(LJGrad_values, label='LJ grad potential')
    # plt.legend()
    # plt.show()
    plt.plot(LJ_Gauss_values, label='potential + bias')
    plt.legend()
    plt.show()
    plt.plot(Gauss_values, label='bias')
    plt.legend()
    plt.show()
    # plt.plot(GaussGrad_values, label='grad bias')
    # plt.legend()
    # plt.show()
    # plt.plot(LJGrad_GaussGrad_values, label='grad potential + grad bias')
    # plt.legend()
    # plt.show()

def torsion(xyzs, i, j, k):
    # compute the torsion angle for atoms i,j,k
    ibeg = (i - 1) * 3
    iend = i * 3
    jbeg = (j - 1) * 3
    jend = j * 3
    kbeg = (k - 1) * 3
    kend = k * 3

    rij = xyzs[ibeg:iend] - xyzs[jbeg:jend]
    rkj = xyzs[kbeg:kend] - xyzs[jbeg:jend]
    cost = np.sum(rij * rkj)
    sint = np.linalg.norm(np.cross(rij, rkj))
    angle = np.arctan2(sint, cost)
    return angle

def cart2zmat(X):
    X = X.T
    nrows, ncols = X.shape
    na = nrows // 3
    Z = []

    for j in range(ncols):
        rlist = []  # list of bond lengths
        alist = []  # list of bond angles (radian)
        dlist = []  # list of dihedral angles (radian)
        # calculate the distance matrix between atoms
        distmat = np.zeros((na, na))
        for jb in range(na):
            jbeg = jb * 3
            jend = (jb + 1) * 3
            xyzb = X[jbeg:jend, j]
            for ia in range(jb + 1, na):
                ibeg = ia * 3
                iend = (ia + 1) * 3
                xyza = X[ibeg:iend, j]
                distmat[ia, jb] = np.linalg.norm(xyza - xyzb)
        distmat = distmat + np.transpose(distmat)

        if na > 1:
            rlist.append(distmat[0, 1])

        if na > 2:
            rlist.append(distmat[0, 2])
            alist.append(torsion(X[:, j], 3, 1, 2))

        Z.append(rlist + alist + dlist)

    Z = np.array(Z)
    return Z.T

def show_trajectory_plot(r_values, LJ_values, Gauss_values):
    # all_potential_values = [LJpotential(np.array([r_values[i]])) for i in range(len(r_values))]

    # Gauss_values starts as a triangular shaped 2-d array (sort of), but what we want is to be plotting the sum of the bias
    # at each center location. 
    Gauss_values = np.array([np.sum(Gauss_values[i]) for i in range(len(Gauss_values))])

    all_z_values = cart2zmat(r_values)

    num_iters = len(r_values)
    
    iter_fractions = [float(i) / num_iters for i in range(num_iters)]
    
    min_potential = min(LJ_values)
    max_potential = max(LJ_values)
    energy_fractions = [(LJ_values[i] - min_potential) / (max_potential - min_potential) for i in range(num_iters)]

    min_bias = min(Gauss_values)
    max_bias = max(Gauss_values)
    bias_fractions = [(Gauss_values[i] - min_bias) / (max_bias - min_bias) for i in range(num_iters)]

    # Create a 3D plot
    cmap = plt.get_cmap('plasma')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(z_trajectory[:, 0], z_trajectory[:, 1], z_trajectory[:, 2], c=indices, cmap=cmap)
    traj_plot = ax.scatter(all_z_values[0, :], all_z_values[1, :], all_z_values[2, :], c=energy_fractions, cmap=cmap)
    traj_colorbar = plt.colorbar(traj_plot)
    traj_colorbar.set_label(f'Potential with max {max_potential:.3f} and min {min_potential:.3f}')
    plt.show()

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(z_trajectory[:, 0], z_trajectory[:, 1], z_trajectory[:, 2], c=indices, cmap=cmap)
    traj_plot = ax.scatter(all_z_values[0, :], all_z_values[1, :], all_z_values[2, :], c=bias_fractions, cmap=cmap)
    traj_colorbar = plt.colorbar(traj_plot)
    traj_colorbar.set_label(f'Bias with max {max_bias:.3f} and min {min_bias:.3f}')
    plt.show()



    # # the below would only work for a 2d potential...
    # xx = np.linspace(-2, 2, 100)
    # yy = np.linspace(-1, 2, 100)
    # [X, Y] = np.meshgrid(xx, yy)  # 100*100
    # # W = LJpotential(X, Y)
    # W = np.zeros((len(xx), len(yy)))
    # for i in range(len(xx)):
    #     for j in range(len(yy)):
    #         W[i][j] = LJpotential(xx[i], yy[j])

# M = 30  # M = 30 for 10 atoms, each with 3 dimensions
M = 12
# M = 9
# T = 20
T = 1
Tdeposite = 0.05    # time until place gaussian
dt = 0.001
# h = 1
# sigma = 1

# [CHANGED now] This seemed to work well for 3 atoms with the COM/MOI symmetry reduction strategy
T = 1
Tdeposite = 0.05    # time until place gaussian
dt = 0.001
h = 0.01         # height
sigma = 0.001     # stdev
kbT = 0.0001    # 0.0001 gives a y range of -5.99997 to -5.99898

ten_atom_init = [   [-0.112678561569957,   1.154350068036701,  -0.194840194577019],
                    [0.455529927957375,  -0.141698933988423,   1.074987039970359],
                    [-1.076845089999501,  -0.472850203002737,  -0.000759561321676],
                    [0.772283646029137,   0.650565133523509,   0.318426126854726],
                    [-0.156611774846419,  -0.917108951862921,   0.505406908904027],
                    [-0.704034080497556,   0.331528029107990,  -0.717967548406147],
                    [0.749396644213383,  -0.438279561655135,   0.048302291736783],
                    [-0.145155893839899,  -0.631595300103604,  -0.579684879295956],
                    [0.417694325594403,   0.318313250559209,  -0.692983073948749],
                    [-0.199579143040966,   0.146776469285409,   0.239112890083651]]

four_atom_init = [  [-0.3616353090,        0.0439914505,        0.5828840628],
                    [0.2505889242,        0.6193583398,       -0.1614607010],
                    [-0.4082757926,       -0.2212115329,       -0.5067996704],
                    [0.5193221773,       -0.4421382574,        0.0853763087]]

three_atom_init = [ [0.4391356726,        0.1106588251,       -0.4635601962],
                    [-0.5185079933,        0.3850176090,        0.0537084789],
                    [0.0793723207,       -0.4956764341,        0.4098517173]]

time1 = time.time()
LD_MetaDyn(M, T, Tdeposite, dt, h, sigma, kbT, ic_method='PCA')
time2 = time.time()
print(f'Total time: {time2-time1}')
# LD_MetaDyn(M, T, Tdeposite, dt, h, sigma, kbT, ic_method='PCA')

# N = 1
# M = 2
# k = 1
# h = 0.1
# sigma = 0.2

# h = 10
# sigma = 10

# N = 5
# M = 7
# k = 2
# h = 0.1
# sigma = 0.4

# x = np.random.rand(1, M)
# centers = np.random.rand(N, M)
# eigenvectors = np.random.rand(N, M, k)
# x = np.array([[1.,1.]])
# centers = np.array([[0.,0.]])
# eigenvectors = np.array([[[0.],[1.]]])


# xx = np.linspace(-.5, .5, 100)
# yy = np.linspace(-.5, .5, 100)
# [X, Y] = np.meshgrid(xx, yy)  # 100*100
# W = np.array([[JSumGaussian([X[i][j], Y[i][j]], centers, eigenvectors, orth_eigenvectors, h, sigma)[0][0] for j in range(100)] for i in range(100)])
# print(W.shape)
# fig = plt.figure(figsize=(10,6))
# ax1 = fig.add_subplot(1, 1, 1)
# contourf_ = ax1.contourf(X, Y, W, levels=29)
# plt.colorbar(contourf_)
# plt.show()

# print('Full')
# print(GradGaussian(x, centers, eigenvectors, h, sigma))
# for i in range(M):
#     shift = 0.0001
#     e=np.zeros((1,M))
#     e[0, i]= shift
#     print((JSumGaussian(x+e, centers, eigenvectors, h, sigma)-JSumGaussian(x, centers, eigenvectors, orth_eigenvectors, h, sigma))/shift)

# LD(24, 0.001, 200)
# #
# # Set up basinhopping optimization
# minimizer_kwargs = {"method": "L-BFGS-B"}
# result = basinhopping(LJpotential1, x0=[0.0, 0.0, 0.0, -0.1, 0.1], minimizer_kwargs=minimizer_kwargs, niter=10000)
# print(result)
# print(result.fun)
#
# r = np.zeros((1,10))
# print(LJpotential(np.array([[0.1, -0.2, 0.1, -0.1, 0.1]])))