import os
from matplotlib import pyplot as plt
from matplotlib import colormaps
import numpy as np

def load_simulation_data(run_number):
    run_dir = os.path.join('simulation_runs', f'run_{run_number}')
    
    data = {}
    
    # Load trajectory data
    trajectory_file = os.path.join(run_dir, 'trajectory.npz')
    if os.path.exists(trajectory_file):
        with np.load(trajectory_file) as traj_data:
            data['r_values'] = traj_data['r_values']
    
    # Load gaussians data
    gaussians_file = os.path.join(run_dir, 'gaussians.npz')
    if os.path.exists(gaussians_file):
        with np.load(gaussians_file) as gauss_data:
            data['rcenters'] = gauss_data['rcenters']
            data['eigenvectors'] = gauss_data['eigenvectors']
    
    # Load trajectory energies
    traj_energies_file = os.path.join(run_dir, 'traj_energies.npz')
    if os.path.exists(traj_energies_file):
        with np.load(traj_energies_file, allow_pickle=True) as traj_energy_data:
            data['LJ_values'] = traj_energy_data['LJ_values']
            data['Gauss_values'] = traj_energy_data['Gauss_values']
            data['Gauss_v_dist_values'] = traj_energy_data['Gauss_v_dist_values']
    
    # Load trajectory gradients
    traj_gradients_file = os.path.join(run_dir, 'traj_gradients.npz')
    if os.path.exists(traj_gradients_file):
        with np.load(traj_gradients_file) as traj_grad_data:
            data['LJGrad_values'] = traj_grad_data['LJGrad_values']
            data['GaussGrad_values'] = traj_grad_data['GaussGrad_values']
    
    # Load center energies
    center_energies_file = os.path.join(run_dir, 'center_energies.npz')
    if os.path.exists(center_energies_file):
        with np.load(center_energies_file, allow_pickle=True) as center_energy_data:
            # TODO: this one throws an error
            data['Gauss_center_values'] = center_energy_data['Gauss_center_values']
            data['LJ_center_values'] = center_energy_data['LJ_center_values']
    
    # Load parameters
    parameters_file = os.path.join(run_dir, 'parameters.npz')
    if os.path.exists(parameters_file):
        with np.load(parameters_file, allow_pickle=True) as param_data:
            # TODO: this one throws an error
            data['parameters'] = param_data['parameters'].item()
    
    return data

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




# Set to the desired run number
run_number = 6
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

analyze_means(rcenters)
# analyze_dist_gauss(Gauss_v_dist_values)
analyze_iter_gauss(Gauss_v_dist_values)
analyze_LJ_potential(LJ_values, LJGrad_values, Gauss_values, GaussGrad_values, LJ_Gauss_values, LJGrad_GaussGrad_values)
if parameters['M'] == 9:
    # show_trajectory_plot(np.array(r_values).reshape((len(r_values), 9)), LJ_values, Gauss_values)
    show_trajectory_plot(rcenters, np.array(LJ_center_values), np.reshape(np.array(Gauss_center_values), (len(Gauss_center_values))))

