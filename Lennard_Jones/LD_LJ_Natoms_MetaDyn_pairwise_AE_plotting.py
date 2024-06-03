import os
from matplotlib import pyplot as plt
from matplotlib import colormaps
import numpy as np

def load_simulation_data(run_number):
    run_dir = os.path.join('simulation_runs', f'run_{run_number}')
    
    data = {}

    # Load all data
    data_file = os.path.join(run_dir, 'data.npz')
    if os.path.exists(data_file):
        with np.load(data_file, allow_pickle=True) as all_data:
            data['r_values'] = all_data['r_values']
            data['rcenters'] = all_data['rcenters']
            data['eigenvectors'] = all_data['eigenvectors']
            data['LJ_values'] = all_data['LJ_values']
            data['Gauss_values'] = all_data['Gauss_values']
            data['Gauss_v_dist_values'] = all_data['Gauss_v_dist_values']
            data['LJGrad_values'] = all_data['LJGrad_values']
            data['GaussGrad_values'] = all_data['GaussGrad_values']
            data['Gauss_center_values'] = all_data['Gauss_center_values']
            data['LJ_center_values'] = all_data['LJ_center_values']
            data['parameters'] = all_data['parameters'].item()
    
    return data

def get_pairwise_distances(x):
    Natoms = int(x.shape[-1] / 3)
    x = np.reshape(x, (Natoms, 3))
    all_diffs = np.expand_dims(x, axis=1) - np.expand_dims(x, axis=0) # N * N * M
    # diffs = np.array([all_diffs[i][j] for [i,j] in pairs])
    pairwise_distances = np.sqrt(np.sum(all_diffs**2, axis=-1)) # N * N

    pairwise_distances = pairwise_distances[np.triu_indices(Natoms, 1)]

    return pairwise_distances

def PsiSq(distSq):

    sigma = 1
    epsilon = 1

    return 4*epsilon*(sigma**12/distSq**6 - sigma**6/distSq**3)

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

# Non-pairwise PCA
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
    k = 2  # Set the desired number of components

    # Step 4.7: Retain the top k components
    selected_eigenvectors = eigenvectors[:, 0:k]

    return mean_vector, selected_eigenvectors
    # return mean_vector, std_vector, selected_eigenvectors, eigenvalues

def pairwise_PCA(data):  # datasize: N * dim
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
    k = 2  # Set the desired number of components

    # Step 4.7: Retain the top k components
    selected_eigenvectors = eigenvectors[:, 0:k]

    return mean_vector, selected_eigenvectors
    # return mean_vector, std_vector, selected_eigenvectors, eigenvalues

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

def show_2D_trajectory(r_values):
    # r_values: N * 1 * M
    N = r_values.shape[0]
    M = r_values.shape[-1]
    data = np.reshape(r_values, (N, M))
    # data: N * M

    # Calculate main PCA vectors of the entire trajectory
    mean_vector, selected_eigenvectors = PCA(data)
    # eigenvectors: M * k  =>  M * 2

    # Project trajectory onto the top 2 evecs
    projected_data = np.dot(data, selected_eigenvectors)
    # projected_data: N * k  =>  N * 2

    # Plot the data
    # Create a color map based on the order of the data points
    colors = np.linspace(0, 1, N)

    # Plot the data points with color representing their order
    plt.scatter(data[:, 0], data[:, 1], c=colors, cmap='viridis', edgecolor='k')

    # Create a mesh grid for plotting the background energy surface
    buffer = 0.0
    x_min, x_max = data[:, 0].min() - buffer, data[:, 0].max() + buffer
    y_min, y_max = data[:, 1].min() - buffer, data[:, 1].max() + buffer
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Transform the mesh grid points using the eigenvectors
    concatenated_points = np.c_[xx.ravel(), yy.ravel()]
    print(concatenated_points.shape)
    print(selected_eigenvectors.T.shape)
    grid_points = np.dot(np.c_[xx.ravel(), yy.ravel()], selected_eigenvectors.T)
    print(grid_points.shape)
    print(grid_points[0].shape)

    # Calculate the energy potential at each grid point
    zz = np.array([LJpotential(np.expand_dims(point,0)) for point in grid_points]).reshape(xx.shape)

    # Plot the energy potential surface
    plt.contourf(xx, yy, zz, levels=50, cmap='coolwarm', alpha=0.6)

    # Add colorbar for the energy surface
    plt.colorbar(label='Energy Potential')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D Plot of Data Points with Energy Potential Background')

    # Show the plot
    plt.show()

def show_2D_pairwise_trajectory(r_values):
    # Calculate main PCA vectors of the entire trajectory
    mean_vector, selected_eigenvectors = PCA(r_values)
    # Project trajectory onto the top 2 evecs -- be careful!




# Set to the desired run number
run_number = 1
simulation_data = load_simulation_data(run_number)

# Print loaded data keys to verify
print(simulation_data.keys())

r_values = simulation_data.get('r_values')
rcenters = simulation_data.get('rcenters')
eigenvectors = simulation_data.get('eigenvectors')
LJ_values = simulation_data.get('LJ_values')
Gauss_values = simulation_data.get('Gauss_values')
Gauss_v_dist_values = simulation_data.get('Gauss_v_dist_values')
LJGrad_values = simulation_data.get('LJGrad_values')
GaussGrad_values = simulation_data.get('GaussGrad_values')
Gauss_center_values = simulation_data.get('Gauss_center_values')
LJ_center_values = simulation_data.get('LJ_center_values')
parameters = simulation_data.get('parameters')

LJ_Gauss_values = LJ_values + Gauss_values
LJGrad_GaussGrad_values = LJGrad_values + GaussGrad_values


show_2D_trajectory(r_values)
# analyze_means(rcenters)
# analyze_dist_gauss(Gauss_v_dist_values)
# analyze_iter_gauss(Gauss_v_dist_values)
# analyze_LJ_potential(LJ_values, LJGrad_values, Gauss_values, GaussGrad_values, LJ_Gauss_values, LJGrad_GaussGrad_values)
# if parameters['M'] == 9:
#     # show_trajectory_plot(np.array(r_values).reshape((len(r_values), 9)), LJ_values, Gauss_values)
#     show_trajectory_plot(rcenters, np.array(LJ_center_values), np.reshape(np.array(Gauss_center_values), (len(Gauss_center_values))))

