import time
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tqdm import tqdm

from matplotlib import colormaps

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

def center_of_mass(r):
    total_mass = len(r)
    com = np.sum(r, axis=0) / total_mass
    return com

def moments_of_inertia(r, com):
    inertia_tensor = np.zeros((3, 3))
    for particle in r:
        rel_position = particle - com
        # print(f'relative position: {rel_position}')
        inertia_tensor += np.outer(rel_position, rel_position)
        # print(f'inertia tensor: {inertia_tensor}')
    return inertia_tensor

def principal_axes(inertia_tensor):
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
    # Sort eigenvectors based on eigenvalues
    order = np.argsort(eigenvalues)[::-1]
    # eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    return eigenvectors

# Shift positions so center of mass is at the origin, and so that the moments of inertia
# are aligned with the z,y,x axes; z is the direction of the largest moment of inertia
def rotate_r_to_principal_axes(r):
    Natoms = int(r.shape[-1] / 3)
    r = np.reshape(r, (Natoms, 3))
    com = center_of_mass(r)
    r_shifted = r - com
    inertia_tensor = moments_of_inertia(r_shifted, com)
    eigenvectors = principal_axes(inertia_tensor)
    r_rotated = np.dot(r_shifted, eigenvectors)
    r_rotated = np.reshape(r_rotated, (1, 3*Natoms))
    return r_rotated

def rotate_all_to_principal_axes(r, rcenters, ic_eigenvectors):
    Natoms = int(r.shape[-1] / 3)
    print(f'Natoms: {Natoms}')
    r = np.reshape(r, (Natoms, 3))
    com = center_of_mass(r)
    r_shifted = r - com
    inertia_tensor = moments_of_inertia(r_shifted, com)
    eigenvectors = principal_axes(inertia_tensor)

    r_rotated = np.dot(r_shifted, eigenvectors)
    r_rotated = np.reshape(r_rotated, (1, 3*Natoms))

    print(f'rcenters.shape: {rcenters.shape}')
    Natoms = int(rcenters.shape[-1] / 3)
    print(f'Natoms: {Natoms}')
    rcenters = np.reshape(r, (Natoms, 3))   # from 1 * 3Natoms to 3 * Natoms
    print(f'rcenters.shape: {rcenters.shape}')
    rcenters_shifted = rcenters - com
    rcenters_rotated = np.dot(rcenters_shifted, eigenvectors)
    rcenters_rotated = np.reshape(rcenters_rotated, (1, 3*Natoms))
    print(f'rcenters_rotated.shape: {rcenters_rotated.shape}')

    print(f'ic_eigenvectors.shape: {ic_eigenvectors.shape}')
    k = ic_eigenvectors.shape[-1]
    Natoms = int(ic_eigenvectors.shape[-2] / 3)
    Ngauss = int(ic_eigenvectors.shape[0])
    print(f'Natoms: {Natoms}')
    ic_eigenvectors = np.reshape(ic_eigenvectors, (Ngauss, Natoms, 3, k))
    print(f'ic_eigenvectors.shape: {ic_eigenvectors.shape}')
    ic_eigenvectors_rotated = np.tensordot(ic_eigenvectors, eigenvectors, axes=([2], [1]))
    ic_eigenvectors_rotated = np.reshape(ic_eigenvectors_rotated, (Ngauss, 3*Natoms, k))
    print(f'ic_eigenvectors_rotated.shape: {ic_eigenvectors_rotated.shape}')

    return r_rotated, rcenters_rotated, ic_eigenvectors_rotated

def PCA(data):  # datasize: N * dim
    # Step 4.1: Compute the mean of the data
    data_z = data  # bs*3

    mean_vector = np.mean(data_z, axis=0, keepdims=True)
    std_vector = np.std(data_z, axis=0, keepdims=True)

    # Step 4.2: Center the data by subtracting the mean
    centered_data = (data_z - mean_vector)

    # Step 4.3: Compute the covariance matrix of the centered data
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Step 4.4: Perform eigendecomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 4.5: Sort the eigenvectors based on eigenvalues (descending order)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 4.6: Choose the number of components (optional)
    k = 4  # Set the desired number of components

    # Step 4.7: Retain the top k components
    selected_eigenvectors = eigenvectors[:, 0:k]

    return mean_vector, selected_eigenvectors
    # return mean_vector, std_vector, selected_eigenvectors, eigenvalues

def AE(data):
    mean_vector = np.mean(data, axis=0, keepdims=True)
    std_vector = np.std(data, axis=0, keepdims=True)
    
    input_dim = data.shape[1]
    print(f'input dim: {input_dim}')
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
    autoencoder.fit(data, data, epochs=300, batch_size=32, shuffle=True, validation_split=0.2)
    
    # Get out base vectors to plot
    base_vectors = np.identity(input_dim)
    encoded_base_vectors = encoder.predict(base_vectors)

    ae_comps = encoded_base_vectors.T[:,0:input_dim]

    ae_comp_norms = la.norm(ae_comps.T, axis=0)

    ae_comps_normalized = (ae_comps / ae_comp_norms[:, np.newaxis]).T

    centered_data = (data - mean_vector)
    covariance_matrix = np.cov(centered_data, rowvar=False)

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
    # GS_eigenvalues = np.array([])
    # print(ae_comps_normalized)
    for i in range(len(ae_comps_normalized)):
        cur_vec = ae_comps_normalized[i]
        for j in range(i-1, -1, -1):
            prev_vec = GS_eigenvectors[j]
            # subtract the projection of cur_vec onto prev_vec
            cur_vec = cur_vec - prev_vec * np.dot(cur_vec, prev_vec) / np.dot(prev_vec, prev_vec)
        GS_eigenvectors[i] = cur_vec
        # print(i)
        # print(cur_vec)
        # print(GS_eigenvectors)
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

def GradGuassianTF(x, centers, eigenvectors, h, sigma):
    # TODO: make it so that this tape only gets set up once. Probably need to eliminate this function altogether
    x_value = tf.constant(x)
    with tf.GradientTape() as tape:
        tape.watch(x_value)
        y = tf.constant(SumGuassianTF(x_value, centers, eigenvectors, h, sigma))
    dy_dx = tape.gradient(y, x_value)
    return dy_dx

def SumGuassian(x, centers, eigenvectors, h, sigma):
    # x: 1 * M
    # centers: N * M
    # eigenvectors: N * M * k
    env_sigma = sigma

    # Gaussian in the direction of the CVs
    x_minus_centers_original = x - centers # N * M
    x_minus_centers = np.expand_dims(x_minus_centers_original, axis=1) # N * 1 * M
    x_projected = np.matmul(x_minus_centers, eigenvectors) # N * 1 * k
    x_projected_sq_sum = np.sum(x_projected**2, axis=(-2, -1)) # N
    exps = h*np.exp(-np.expand_dims(x_projected_sq_sum, axis=1)/2/sigma**2) # N * 1

    # Gaussian in the directions orthogonal to the CVs
    eigenvectors_orth = np.array([scipy.linalg.orth(eigenvectors[i]) for i in range(len(eigenvectors))]) # N * M * k
    x_projected_orth = np.matmul(x_minus_centers, eigenvectors_orth) * eigenvectors_orth    # N * M * k

    x_projected_sum = np.sum(x_projected_orth, axis=-1) # N * M
    x_remainder = x_minus_centers_original - x_projected_sum     # N * M

    x_envelope_sq_sum = np.sum(x_remainder**2, axis=-1)   # N * M -> N

    envelope_exps = h*np.exp(-np.expand_dims(x_envelope_sq_sum, axis=1)/2/env_sigma**2) # N * 1

    # Combine the Gaussians
    exps = exps * envelope_exps # N * 1

    # total_bias = np.sum(exps, axis=0, keepdims=True) # N * 1? NOPE not at all true
    total_bias = np.sum(exps, axis=1) # 1

    # TODO??: Normalize AND plot the size of normalization factor
    # Track the new sigma values that we calculate and use that for all calcs

    # TODO: variable sigma's dependent on the size of the eigenvalue. Larger eigenvalue = larger Gaussian
    # NEED that as it might potentially help the AE specifically

    # print(total_bias)
    # print(f'exps.shape: {exps.shape}')
    # print(f'x.shape: {x.shape}')
    # print(f'total_bias.shape: {total_bias.shape}')
    return total_bias

@jax.jit
def JSumGuassian(x, centers, eigenvectors, h, sigma):
    # x: 1 * M
    # centers: N * M
    # eigenvectors: N * M * k
    env_sigma = sigma

    # Gaussian in the direction of the CVs
    x_minus_centers_original = x - centers # N * M
    x_minus_centers = jnp.expand_dims(x_minus_centers_original, axis=1) # N * 1 * M
    x_projected = jnp.matmul(x_minus_centers, eigenvectors) # N * 1 * k
    x_projected_sq_sum = jnp.sum(x_projected**2, axis=(-2, -1)) # N
    exps = h*jnp.exp(-jnp.expand_dims(x_projected_sq_sum, axis=1)/2/sigma**2) # N * 1

    # Gaussian in the directions orthogonal to the CVs
    eigenvectors_orth = jnp.array([eigenvectors[i] / jnp.linalg.norm(eigenvectors[i]) for i in range(len(eigenvectors))]) # N * M * k
    x_projected_orth = jnp.matmul(x_minus_centers, eigenvectors_orth) * eigenvectors_orth    # N * M * k

    x_projected_sum = jnp.sum(x_projected_orth, axis=-1) # N * M
    x_remainder = x_minus_centers_original - x_projected_sum     # N * M

    x_envelope_sq_sum = jnp.sum(x_remainder**2, axis=-1)   # N * M -> N

    envelope_exps = h*jnp.exp(-jnp.expand_dims(x_envelope_sq_sum, axis=1)/2/env_sigma**2) # N * 1

    # Combine the Gaussians
    exps = exps * envelope_exps

    total_bias = jnp.sum(exps) # 1 * M

    # TODO??: Normalize AND plot the size of normalization factor
    # Track the new sigma values that we calculate and use that for all calcs

    # TODO: variable sigma's dependent on the size of the eigenvalue. Larger eigenvalue = larger Gaussian
    # NEED that as it might potentially help the AE specifically

    # TODO: would using vmap lead to a faster implementation?

    return total_bias

jax_SumGaussian = jax.value_and_grad(JSumGuassian)
jax_SumGaussian_jit = jax.jit(jax_SumGaussian)

def GradGuassian(x, centers, eigenvectors, h, sigma):
    x_jnp = jnp.array(x)
    centers_jnp = jnp.array(centers)
    eigenvectors_jnp = jnp.array(eigenvectors)

    val, grad = jax_SumGaussian_jit(x_jnp, centers_jnp, eigenvectors_jnp, h, sigma)

    return grad

def SumGuassianTF(x, centers, eigenvectors, h, sigma):
    # x: 1 * M
    # centers: N * M
    # eigenvectors: N * M * k

    env_sigma = sigma

    # Convert NumPy arrays to TensorFlow tensors
    x_tensor = tf.constant(x)
    centers_tensor = tf.constant(centers)
    eigenvectors_tensor = tf.constant(eigenvectors)

    # Gaussian in the direction of the CVs
    x_minus_centers = x_tensor - centers_tensor  # N * M
    x_minus_centers_expanded = tf.expand_dims(x_minus_centers, axis=1)  # N * 1 * M
    x_projected = tf.matmul(x_minus_centers_expanded, eigenvectors_tensor)  # N * 1 * k
    x_projected_sq_sum = tf.reduce_sum(x_projected ** 2, axis=(-2, -1))  # N
    exps = h * tf.exp(-tf.expand_dims(x_projected_sq_sum, axis=1) / (2 * sigma ** 2))  # N * 1

    # Gaussian in the directions orthogonal to the CVs
    eigenvectors_orth = tf.convert_to_tensor([scipy.linalg.orth(eigenvectors[i]) for i in range(len(eigenvectors))]) # N * M * k
    x_projected_orth = tf.matmul(x_minus_centers_expanded, eigenvectors_orth) * eigenvectors_orth  # N * M * k
    x_projected_sum = tf.reduce_sum(x_projected_orth, axis=-1) # N * M
    x_remainder = x_minus_centers - x_projected_sum     # N * M
    x_envelope_sq_sum = tf.reduce_sum(x_remainder**2, axis=-1)   # N * M -> N

    envelope_exps = h * tf.exp(-tf.expand_dims(x_envelope_sq_sum, axis=1) / (2 * env_sigma ** 2))  # N * 1
    total_bias = tf.reduce_sum(envelope_exps * exps, axis=0, keepdims=True)  # 1 * M

    return total_bias

def DistSumGuassian(x, centers, eigenvectors, h, sigma):
    # x: 1 * M
    # centers: N * M
    # eigenvectors: N * M * k
    env_sigma = sigma

    # Gaussian in the direction of the CVs
    x_minus_centers_original = x - centers # N * M
    x_minus_centers_dists = np.linalg.norm(x_minus_centers_original, axis=1) # N
    x_minus_centers = np.expand_dims(x_minus_centers_original, axis=1) # N * 1 * M
    x_projected = np.matmul(x_minus_centers, eigenvectors) # N * 1 * k
    x_projected_sq_sum = np.sum(x_projected**2, axis=(-2, -1)) # N
    exps = h*np.exp(-np.expand_dims(x_projected_sq_sum, axis=1)/2/sigma**2) # N * 1

    # Gaussian in the directions orthogonal to the CVs
    eigenvectors_orth = np.array([scipy.linalg.orth(eigenvectors[i]) for i in range(len(eigenvectors))]) # N * M * k
    x_projected_orth = np.matmul(x_minus_centers, eigenvectors_orth) * eigenvectors_orth    # N * M * k

    x_projected_sum = np.sum(x_projected_orth, axis=-1) # N * M
    x_remainder = x_minus_centers_original - x_projected_sum     # N * M

    x_envelope_sq_sum = np.sum(x_remainder**2, axis=-1)   # N * M -> N

    envelope_exps = h*np.exp(-np.expand_dims(x_envelope_sq_sum, axis=1)/2/env_sigma**2) # N * 1

    # Combine the Gaussians
    exps = exps * envelope_exps

    result = [x_minus_centers_dists, exps[:,0]]
    return result

def LD_MetaDyn(M, T, Tdeposite, dt, h, sigma, kbT, ic_method='PCA'):
    # M: dim
    r = np.random.randn(1, M)*1

    if M == 30:
        r = np.reshape(ten_atom_init, (1,30))

    if M == 9:
        r = np.reshape(three_atom_init, (1,9))

    Nsteps = round(T / dt)
    NstepsDeposite = round(Tdeposite / dt)
    print(NstepsDeposite)
    trajectories4PCA = np.zeros((NstepsDeposite, 1, M))

    rcenters = None
    eigenvectors = None
    # Smallest = float('inf')

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
            Gauss_v_dist = DistSumGuassian(r, rcenters, eigenvectors, h, sigma)
            Gauss = np.sum(SumGuassian(r, rcenters, eigenvectors, h, sigma))
            GaussGrad = np.sum(GradGuassian(r, rcenters, eigenvectors, h, sigma))
            Gauss_v_dist_values.append(Gauss_v_dist)

        LJ_values.append(LJpot)
        LJGrad_values.append(LJGrad)
        Gauss_values.append(Gauss)
        GaussGrad_values.append(GaussGrad)
        LJ_Gauss_values.append(LJpot + Gauss)
        LJGrad_GaussGrad_values.append(LJGrad + GaussGrad)
        r_values.append(r)

        r = next_step(r, rcenters, eigenvectors, h, sigma, kbT, dt)
        # print(trajectories4PCA.shape, r.shape)
        trajectories4PCA[i % NstepsDeposite, :] = r
        # print(trajectories4PCA)
        if (i + 1) % NstepsDeposite == 0:
            # print(r, LJpotential(r))
            # r = next_step(r, rcenters, eigenvectors, h, sigma, kbT, dt)
            if rcenters is None:
                ### conducting PCA ###
                data = trajectories4PCA

                data = np.squeeze(data, axis=1)
                if ic_method == 'PCA':
                    mean_vector, selected_eigenvectors = PCA(data)
                else:
                    mean_vector, std_vector, selected_eigenvectors, eigenvalues, GS_eigenvectors, GS_eigenvalues = AE(data)
                # print(selected_eigenvectors.shape, eigenvalues.shape)
                rcenters = mean_vector
                # TODO: should this be using the GS eigenvectors??
                eigenvectors = np.expand_dims(selected_eigenvectors, axis=0)

                ### reset the PCA dataset
                trajectories4PCA = np.zeros((NstepsDeposite, 1, M))

                LJ_center_values.append(LJpotential(mean_vector))
                Gauss_center_values.append(SumGuassian(mean_vector, rcenters, eigenvectors, h, sigma))
            else:
                ### conducting PCA ###
                data = trajectories4PCA

                data = np.squeeze(data, axis=1)
                if ic_method == 'PCA':
                    mean_vector, selected_eigenvectors = PCA(data)
                else:
                    mean_vector, std_vector, selected_eigenvectors, eigenvalues, GS_eigenvectors, GS_eigenvalues = AE(data)

                rcenters = np.concatenate([rcenters, mean_vector], axis=0)
                print(f'rcenters shape: {rcenters.shape}')
                eigenvectors = np.concatenate([eigenvectors, np.expand_dims(selected_eigenvectors, axis=0)], axis=0)
                # print(rcenters.shape, eigenvectors.shape)

                # if rcenters.shape[0]>20:
                #     rcenters = rcenters[-50:]
                #     eigenvectors = eigenvectors[-50:]
                # print(rcenters.shape, eigenvectors.shape)
                ### reset the PCA dataset
                trajectories4PCA = np.zeros((NstepsDeposite, 1, M))

                LJ_center_values.append(LJpotential(mean_vector))
                Gauss_center_values.append(SumGuassian(mean_vector, rcenters, eigenvectors, h, sigma))

    analyze_means(rcenters)
    # analyze_dist_gauss(Gauss_v_dist_values)
    analyze_iter_gauss(Gauss_v_dist_values)
    analyze_LJ_potential(LJ_values, LJGrad_values, Gauss_values, GaussGrad_values, LJ_Gauss_values, LJGrad_GaussGrad_values)
    if M == 9:
        # show_trajectory_plot(np.array(r_values).reshape((len(r_values), 9)), LJ_values, Gauss_values)
        show_trajectory_plot(rcenters, np.array(LJ_center_values), np.reshape(np.array(Gauss_center_values), (len(Gauss_center_values))))

    return None

def next_LD(r, dt, kbT):

    rnew = r - (GradLJpotential(r)) * dt + np.sqrt(2 * dt *kbT) * np.random.randn(*r.shape)

    return rnew

def next_LD_Gaussian(r, dt, rcenters, eigenvectors, h, sigma, kbT):

    rnew = r - (GradLJpotential(r) + GradGuassian(r, rcenters, eigenvectors, h, sigma)) * dt + np.sqrt(2 * dt * kbT) * np.random.randn(*r.shape)

    return rnew

def next_step(r, rcenters, eigenvectors, h, sigma, kbT, dt):

    if rcenters is None:
        r = next_LD(r, dt, kbT)
        r = rotate_r_to_principal_axes(r)
    else:
        r = next_LD_Gaussian(r, dt, rcenters, eigenvectors, h, sigma, kbT)
        # r, rcenters, eigenvectors = rotate_all_to_principal_axes(r, rcenters, eigenvectors)
        r = rotate_r_to_principal_axes(r)
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
                # print(f'i: {i}, j: {j}, len(Gauss_v_dist_values[j][1]: {len(Gauss_v_dist_values[j][1])}')
                # print(Gauss_v_dist_values[j])
                # print(Gauss_v_dist_values[j][1])
                # print(Gauss_v_dist_values[j][1][i])
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
    print(f'na: {na}')
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
M = 9
# T = 20
T = 1
Tdeposite = 0.05    # time until place gaussian
dt = 0.001
# h = 1
# sigma = 1

# This seemed to work well for 3 atoms with the COM/MOI symmetry reduction strategy
T = 1
Tdeposite = 0.05    # time until place gaussian
dt = 0.001
h = 0.5         # height
sigma = 0.5     # stdev
kbT = 0.1    # 0.0001 seems to work??

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

three_atom_init = [ [0.4391356726,        0.1106588251,       -0.4635601962],
                    [-0.5185079933,        0.3850176090,        0.0537084789],
                    [0.0793723207,       -0.4956764341,        0.4098517173]]

LD_MetaDyn(M, T, Tdeposite, dt, h, sigma, kbT, ic_method='PCA')
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
# W = np.array([[SumGuassian([X[i][j], Y[i][j]], centers, eigenvectors, h, sigma)[0][0] for j in range(100)] for i in range(100)])
# print(W.shape)
# fig = plt.figure(figsize=(10,6))
# ax1 = fig.add_subplot(1, 1, 1)
# contourf_ = ax1.contourf(X, Y, W, levels=29)
# plt.colorbar(contourf_)
# plt.show()

# print('Full')
# print(GradGuassian(x, centers, eigenvectors, h, sigma))
# for i in range(M):
#     shift = 0.0001
#     e=np.zeros((1,M))
#     e[0, i]= shift
#     print((SumGuassian(x+e, centers, eigenvectors, h, sigma)-SumGuassian(x, centers, eigenvectors, h, sigma))/shift)

# x_value = tf.constant(x)
# with tf.GradientTape() as tape:
#     tape.watch(x_value)
#     y = tf.constant(SumGuassianTF(x_value, centers, eigenvectors, h, sigma))
# dy_dx = tape.gradient(y, x_value)
# print(dy_dx)

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