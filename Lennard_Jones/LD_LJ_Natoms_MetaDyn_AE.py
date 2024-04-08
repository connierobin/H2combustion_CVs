from matplotlib import pyplot as plt
import numpy as np
import scipy
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

def PCA(data):  # datasize: N * dim
    # Step 4.1: Compute the mean of the data
    data_z = data  # bs*3

    mean_vector = np.mean(data_z, axis=0, keepdims=True)

    # Step 4.2: Center the data by subtracting the mean
    centered_data = (data_z - mean_vector)

    # Step 4.3: Compute the covariance matrix of the centered data
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Step 4.4: Perform eigendecomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 4.5: Sort the eigenvectors based on eigenvalues (descending order)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 4.6: Choose the number of components (optional)
    k = 4  # Set the desired number of components

    # Step 4.7: Retain the top k components
    selected_eigenvectors = eigenvectors[:, 0:k]

    return mean_vector, selected_eigenvectors

def GradGuassian(x, centers, eigenvectors, h, sigma):
    # M = number of dimensions: 3 * number of atoms
    # N = number of centers 
    # k = number of CVs
    # x: 1 * M
    # centers: N * M
    # eigenvectors: N * M * k
    env_sigma = sigma / 10.
    k = eigenvectors.shape[-1]

    x_minus_centers = x - centers # N * M

    # don't use this prob
    # x_distance = np.linalg.norm(x_minus_centers, axis=1)   # want N * 1 from the N * M

    # this is wrong
    # envelope_exps = (1/env_sigma**2)*np.exp(-np.expand_dims(x_distance, axis=1)/2/env_sigma**2)   # want N * 1
    # print(f'grad envelope_exps: {envelope_exps.shape}')

    x_minus_centers = np.expand_dims(x_minus_centers, axis=1) # N * 1 * M
    x_projected = np.matmul(x_minus_centers, eigenvectors) # N * 1 * k          # dot product with evecs. (N * 1 * M) * (N * M * k) = (N * 1 * k)
    x_projected_sq_sum = np.sum(x_projected**2, axis=(-2, -1)) # N

    # print(f'x_minus_centers.shape: {x_minus_centers.shape}')
    # print(f'eigenvectors.shape: {eigenvectors.shape}')  # N * M * k
    eigenvectors_orth = np.array([scipy.linalg.orth(eigenvectors[i]) for i in range(len(eigenvectors))]) # N * M * k
    # print(f'eigenvectors_orth.shape: {eigenvectors_orth.shape}')
    x_projected_orth_dists = np.matmul(x_minus_centers, eigenvectors_orth)  # N * 1 * k??
    # print(f'x_projected_orth_dists.shape: {x_projected_orth_dists.shape}')
    x_projected_orth = np.matmul(x_minus_centers, eigenvectors_orth) * eigenvectors_orth    # N * M * k
    # print(f'x_projected_orth.shape: {x_projected_orth.shape}')

    # x_projected_orth_sum = np.expand_dims(np.sum(x_projected_orth, axis=-1), axis=1)    # N * M
    # print(f'x_projected_orth_sum.shape: {x_projected_orth_sum.shape}')

    x_minus_centers_expanded = np.tile(np.squeeze(x_minus_centers, axis=1)[:, :, np.newaxis], (1, 1, k))
    # print(f'x_minus_centers_expanded.shape: {x_minus_centers_expanded.shape}')
    x_envelope = x_minus_centers_expanded - x_projected_orth      # N * M * k
    # print(f'x_envelope.shape: {x_envelope.shape}')
    x_envelope_sq_sum = np.sum(x_envelope**2, axis=(-2, -1))   # N
    # print(f'x_envelope_sq_sum.shape: {x_envelope_sq_sum.shape}')

    # TEST whether it is orthogonal
    # print(np.einsum('nmk,nmk->n', x_projected_orth, x_envelope))

    exps = -h*np.exp(-np.expand_dims(x_projected_sq_sum, axis=1)/2/sigma**2) # N * 1

    envelope_exps = -h*np.exp(-np.expand_dims(x_envelope_sq_sum, axis=1)/2/env_sigma**2) # N * 1

    # print(f'exps: {exps.shape}')
    exps = exps * envelope_exps # still N * 1 ideally
    # print(f'exps.shape: {exps.shape}')

    # TODO: properly calculate the derivative!

    PTPx = np.matmul(eigenvectors, np.transpose(x_projected, axes=(0,2,1))) # N * M * 1
    PTPx = np.squeeze(PTPx, axis=2) # N * M

    # print(f'x_projected.shape: {x_projected.shape}')
    # print(f'x_envelope_dists.shape: {x_projected_orth_dists.shape}')

    PTPx_envelope = np.matmul(eigenvectors, np.transpose(x_projected_orth_dists, axes=(0,2,1))) # N * M * 1
    # print(f'PTPx_envelope.shape: {PTPx_envelope.shape}')
    PTPx_envelope = np.squeeze(PTPx_envelope, axis=2) # N * M
    # print(f'PTPx_envelope.shape: {PTPx_envelope.shape}')

    grad = np.sum(exps * (PTPx / sigma**2 + PTPx_envelope / env_sigma**2), axis=0, keepdims=True) # 1 * M

    return grad

def GradEnvGuassian(x, centers, eigenvectors, h, sigma):
    # M = number of dimensions: 3 * number of atoms
    # N = number of centers 
    # k = number of CVs
    # x: 1 * M
    # centers: N * M
    # eigenvectors: N * M * k
    env_sigma = sigma / 10.
    k = eigenvectors.shape[-1]

    x_minus_centers = x - centers # N * M

    # don't use this prob
    # x_distance = np.linalg.norm(x_minus_centers, axis=1)   # want N * 1 from the N * M

    # this is wrong
    # envelope_exps = (1/env_sigma**2)*np.exp(-np.expand_dims(x_distance, axis=1)/2/env_sigma**2)   # want N * 1
    # print(f'grad envelope_exps: {envelope_exps.shape}')

    x_minus_centers = np.expand_dims(x_minus_centers, axis=1) # N * 1 * M
    x_projected = np.matmul(x_minus_centers, eigenvectors) # N * 1 * k          # dot product with evecs. (N * 1 * M) * (N * M * k) = (N * 1 * k)
    x_projected_sq_sum = np.sum(x_projected**2, axis=(-2, -1)) # N

    # print(f'x_minus_centers.shape: {x_minus_centers.shape}')
    # print(f'eigenvectors.shape: {eigenvectors.shape}')  # N * M * k
    eigenvectors_orth = np.array([scipy.linalg.orth(eigenvectors[i]) for i in range(len(eigenvectors))]) # N * M * k
    # print(f'eigenvectors_orth.shape: {eigenvectors_orth.shape}')
    x_projected_orth_dists = np.matmul(x_minus_centers, eigenvectors_orth)  # N * 1 * k??
    # print(f'x_projected_orth_dists.shape: {x_projected_orth_dists.shape}')
    x_projected_orth = np.matmul(x_minus_centers, eigenvectors_orth) * eigenvectors_orth    # N * M * k
    # print(f'x_projected_orth.shape: {x_projected_orth.shape}')

    # x_projected_orth_sum = np.expand_dims(np.sum(x_projected_orth, axis=-1), axis=1)    # N * M
    # print(f'x_projected_orth_sum.shape: {x_projected_orth_sum.shape}')

    x_minus_centers_expanded = np.tile(np.squeeze(x_minus_centers, axis=1)[:, :, np.newaxis], (1, 1, k))
    # print(f'x_minus_centers_expanded.shape: {x_minus_centers_expanded.shape}')
    x_envelope = x_minus_centers_expanded - x_projected_orth      # N * M * k
    # print(f'x_envelope.shape: {x_envelope.shape}')
    x_envelope_sq_sum = np.sum(x_envelope**2, axis=(-2, -1))   # N
    # print(f'x_envelope_sq_sum.shape: {x_envelope_sq_sum.shape}')

    # TEST whether it is orthogonal
    # print(np.einsum('nmk,nmk->n', x_projected_orth, x_envelope))

    envelope_exps = -h*np.exp(-np.expand_dims(x_envelope_sq_sum, axis=1)/2/env_sigma**2) # N * 1


    # print(f'x_projected.shape: {x_projected.shape}')
    # print(f'x_envelope_dists.shape: {x_projected_orth_dists.shape}')

    PTPx_envelope = np.matmul(eigenvectors, np.transpose(x_projected_orth_dists, axes=(0,2,1))) # N * M * 1
    # print(f'PTPx_envelope.shape: {PTPx_envelope.shape}')
    PTPx_envelope = np.squeeze(PTPx_envelope, axis=2) # N * M
    # print(f'PTPx_envelope.shape: {PTPx_envelope.shape}')

    grad = np.sum(envelope_exps * PTPx_envelope / env_sigma**2, axis=0, keepdims=True) # 1 * M

    return grad

def SumGuassian(x, centers, eigenvectors, h, sigma):
    # x: 1 * M
    # centers: N * M
    # eigenvectors: N * M * k
    env_sigma = sigma / 10.
    k = eigenvectors.shape[-1]

    # Gaussian in the direction of the CVs
    x_minus_centers = x - centers # N * M
    x_minus_centers = np.expand_dims(x_minus_centers, axis=1) # N * 1 * M
    x_projected = np.matmul(x_minus_centers, eigenvectors) # N * 1 * k
    x_projected_sq_sum = np.sum(x_projected**2, axis=(-2, -1)) # N
    exps = h*np.exp(-np.expand_dims(x_projected_sq_sum, axis=1)/2/sigma**2) # N * 1

    # Gaussian in the directions orthogonal to the CVs
    eigenvectors_orth = np.array([scipy.linalg.orth(eigenvectors[i]) for i in range(len(eigenvectors))]) # N * M * k
    x_projected_orth = np.matmul(x_minus_centers, eigenvectors_orth) * eigenvectors_orth    # N * M * k
    x_minus_centers_expanded = np.tile(np.squeeze(x_minus_centers, axis=1)[:, :, np.newaxis], (1, 1, k)) # N * M * k
    x_envelope = x_minus_centers_expanded - x_projected_orth      # N * M * k
    x_envelope_sq_sum = np.sum(x_envelope**2, axis=(-2, -1))   # N
    envelope_exps = np.exp(-np.expand_dims(x_envelope_sq_sum, axis=1)/2/env_sigma**2) # N * 1

    # Combine the Gaussians
    exps = exps * envelope_exps

    total_bias = np.sum(exps, axis=0, keepdims=True) # 1 * M

    # TODO??: Normalize AND plot the size of normalization factor
    # Track the new sigma values that we calculate and use that for all calcs

    # TODO: variable sigma's dependent on the size of the eigenvalue. Larger eigenvalue = larger Gaussian
    # NEED that as it might potentially help the AE specifically

    return total_bias

def SumEnvGuassian(x, centers, eigenvectors, h, sigma):
    # x: 1 * M
    # centers: N * M
    # eigenvectors: N * M * k
    env_sigma = sigma / 10.
    k = eigenvectors.shape[-1]

    # Gaussian in the direction of the CVs
    x_minus_centers = x - centers # N * M
    x_minus_centers = np.expand_dims(x_minus_centers, axis=1) # N * 1 * M
    x_projected = np.matmul(x_minus_centers, eigenvectors) # N * 1 * k
    x_projected_sq_sum = np.sum(x_projected**2, axis=(-2, -1)) # N
    exps = h*np.exp(-np.expand_dims(x_projected_sq_sum, axis=1)/2/sigma**2) # N * 1

    # Gaussian in the directions orthogonal to the CVs
    eigenvectors_orth = np.array([scipy.linalg.orth(eigenvectors[i]) for i in range(len(eigenvectors))]) # N * M * k
    x_projected_orth = np.matmul(x_minus_centers, eigenvectors_orth) * eigenvectors_orth    # N * M * k
    x_minus_centers_expanded = np.tile(np.squeeze(x_minus_centers, axis=1)[:, :, np.newaxis], (1, 1, k)) # N * M * k
    x_envelope = x_minus_centers_expanded - x_projected_orth      # N * M * k
    x_envelope_sq_sum = np.sum(x_envelope**2, axis=(-2, -1))   # N
    envelope_exps = np.exp(-np.expand_dims(x_envelope_sq_sum, axis=1)/2/env_sigma**2) # N * 1

    total_bias = np.sum(envelope_exps, axis=0, keepdims=True) # 1 * M

    # TODO??: Normalize AND plot the size of normalization factor
    # Track the new sigma values that we calculate and use that for all calcs

    # TODO: variable sigma's dependent on the size of the eigenvalue. Larger eigenvalue = larger Gaussian
    # NEED that as it might potentially help the AE specifically

    return total_bias

def DistSumGuassian(x, centers, eigenvectors, h, sigma):
    # x: 1 * M
    # centers: N * M
    # eigenvectors: N * M * k

    env_sigma = sigma / 10.
    k = eigenvectors.shape[-1]

    x_minus_centers = x - centers # N * M
    x_minus_centers_dists = np.linalg.norm(x_minus_centers, axis=1) # N

    # Gaussian in the direction of the CVs
    x_minus_centers = np.expand_dims(x_minus_centers, axis=1) # N * 1 * M
    x_projected = np.matmul(x_minus_centers, eigenvectors) # N * 1 * k
    x_projected_sq_sum = np.sum(x_projected**2, axis=(-2, -1)) # N
    exps = h*np.exp(-np.expand_dims(x_projected_sq_sum, axis=1)/2/sigma**2) # N * 1

    # Gaussian in the directions orthogonal to the CVs
    eigenvectors_orth = np.array([scipy.linalg.orth(eigenvectors[i]) for i in range(len(eigenvectors))]) # N * M * k
    x_projected_orth = np.matmul(x_minus_centers, eigenvectors_orth) * eigenvectors_orth    # N * M * k
    x_minus_centers_expanded = np.tile(np.squeeze(x_minus_centers, axis=1)[:, :, np.newaxis], (1, 1, k)) # N * M * k
    x_envelope = x_minus_centers_expanded - x_projected_orth      # N * M * k
    x_envelope_sq_sum = np.sum(x_envelope**2, axis=(-2, -1))   # N
    envelope_exps = np.exp(-np.expand_dims(x_envelope_sq_sum, axis=1)/2/env_sigma**2) # N * 1

    # Combine the Gaussians
    exps = exps * envelope_exps

    result = [x_minus_centers_dists, exps[:,0]]
    return result

def LD_MetaDyn(M, T, Tdeposite, dt, h, sigma, kbT):
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
                mean_vector, selected_eigenvectors = PCA(data)
                # print(selected_eigenvectors.shape, eigenvalues.shape)
                rcenters = mean_vector
                eigenvectors = np.expand_dims(selected_eigenvectors, axis=0)

                ### reset the PCA dataset
                trajectories4PCA = np.zeros((NstepsDeposite, 1, M))

                LJ_center_values.append(LJpotential(mean_vector))
                Gauss_center_values.append(SumGuassian(mean_vector, rcenters, eigenvectors, h, sigma))
            else:
                ### conducting PCA ###
                data = trajectories4PCA

                data = np.squeeze(data, axis=1)
                mean_vector, selected_eigenvectors = PCA(data)

                rcenters = np.concatenate([rcenters, mean_vector], axis=0)
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
    else:
        r = next_LD_Gaussian(r, dt, rcenters, eigenvectors, h, sigma, kbT)
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

    print(f'r_values.shape: {r_values.shape}')
    print(f'LJ_values.shape: {LJ_values.shape}')
    print(f'Gauss_values.shape: {Gauss_values.shape}')

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
    traj_colorbar.set_label(f'Potential with max {max_bias:.3f} and min {min_bias:.3f}')
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

M = 30  # M = 30 for 10 atoms, each with 3 dimensions
M = 9
T = 20
# T = 1
Tdeposite = 0.05    # time until place gaussian
dt = 0.001
h = 1
sigma = 1
# h = 0.1         # height
# sigma = 0.1     # stdev
kbT = 0.01

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

# LD_MetaDyn(M, T, Tdeposite, dt, h, sigma, kbT)

N = 5
M = 7
k = 2
h = 0.1
sigma = 0.2

h = 10
sigma = 10

x = np.random.rand(1, M)
centers = np.random.rand(N, M)
eigenvectors = np.random.rand(N, M, k)
print(GradGuassian(x, centers, eigenvectors, h, sigma))
for i in range(M):
    shift = 0.0001
    e=np.zeros((1,M))
    e[0, i]= shift
    print((SumGuassian(x+e, centers, eigenvectors, h, sigma)-SumGuassian(x, centers, eigenvectors, h, sigma))/shift)
print(GradEnvGuassian(x, centers, eigenvectors, h, sigma))
for i in range(M):
    shift = 0.0001
    e=np.zeros((1,M))
    e[0, i]= shift
    print((SumEnvGuassian(x+e, centers, eigenvectors, h, sigma)-SumEnvGuassian(x, centers, eigenvectors, h, sigma))/shift)
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