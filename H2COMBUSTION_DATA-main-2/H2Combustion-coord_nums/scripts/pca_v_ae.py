import sys

from matplotlib import pyplot as plt
import numpy as np
import ase
from sklearn.decomposition import PCA

from combust.utils.utility import check_data_consistency
from combust.utils.utility import get_data_remove_appended_zero


def load_data(path_npz):
    """Return the data from path_npz.

    Parameters
    ----------
    path_npz : str
        Path to *.npz file containing reaction data.
 
    """

    # load data
    data = np.load(path_npz)

    # check consistency of data arrays shape
    check_data_consistency(data)
    print(f"{path_npz} number of data points: {data['Z'].shape[0]}")

    # # get data corresponding to reaction
    # data = get_data_subset(data, rxn_num[3:])

    # # remove appended zero to data arrays
    data = get_data_remove_appended_zero(data)

    # get unique reaction numbers included in data
    rxn_nums = np.unique(data['RXN'])
    print(f"{path_npz} reactions: {rxn_nums}")

    # only use the first reaction if there are multiple
    rxn_num = f"rxn{rxn_nums[0]}"

    print(data.keys())

    # from parse():
        #     Returns
        # -------
        # dict: A dictionary with following keys:
        #     'R': positions, array of (n_snapshots, n_atoms, 3)
        #     'Z': atomic numbers, array of (n_snapshots, n_atoms)
        #     'E': energy, array of (n_snapshots, 1)
        #     'F': forces, array of (n_snapshots, n_atoms, 3)


    # # coords: ndarray, shape=(M, N, 3)
    # #     The 3D array of atomic coordinates for M data points and N atoms.
    # R_data = data['R']
    # print(f"R shape: {R_data.shape}")
    # print(R_data[0])

    # #     nums : ndarray, shape=(M, N)
    # #     The 2D array of atomic numbers for M data points and N atoms
    # Z_data = data['Z']
    # print(f"Z shape: {Z_data.shape}")
    # print(Z_data[0])
    
    # # nmber of atoms??
    # N_data = data['N']
    # print(f"N shape: {N_data.shape}")
    # print(N_data[0])

    # # energy
    # E_data = data['E']
    # print(f"E shape: {E_data.shape}")
    # print(E_data[0])

    # # force
    # F_data = data['F']
    # print(f"F shape: {F_data.shape}")
    # print(F_data[0])

    # # reaction numbers??
    # RXN_data = data['RXN']
    # print(f"RXN shape: {RXN_data.shape}")
    # print(RXN_data[0])


    # NOTE: it appears that the "coordination numbers" used as an axis here are the typical
    # chemical meaning of coordination number. The coordination number is larger where the
    # pairwise distance is smaller, following a Fermi-Dirac distribution. The `dist` variable
    # tells it which atoms to look at. 
    # # get reaction data
    # cn1s = rxn_dict[rxn_num]['cn1']
    # cn2s = rxn_dict[rxn_num]['cn2']
    # mu = rxn_dict[rxn_num]['mu']

    # # make namedtuple for plotting
    # cn1 = get_cn_arrays(data['Z'], data['R'], cn1s, mu=mu[0], sigma=3.0)
    # cn2 = get_cn_arrays(data['Z'], data['R'], cn2s, mu=mu[1], sigma=3.0)
    # data_namedtuple = get_rxn_namedtuple(cn1, cn2, data['E'].flatten())

    # return data_namedtuple, rxn_num

    return data

def paired_atoms(atom_positions):
    pairs_A = []
    pairs_B = []
    for i in range(len(atom_positions)):
        for j in range(i+1, len(atom_positions)):
            pairs_A.append(atom_positions[i])
            pairs_B.append(atom_positions[j])
    return pairs_A, pairs_B

def trio_atoms(atom_positions):
    trio_A = []
    trio_B = []
    trio_C = []
    for i in range(len(atom_positions)):
        for j in range(i+1, len(atom_positions)):
            for k in range(j+1, len(atom_positions)):
                trio_A.append(atom_positions[i])
                trio_B.append(atom_positions[j])
                trio_C.append(atom_positions[k])
    return trio_A, trio_B, trio_C

def convert_to_invariant_dists_and_angles(data):
    # TODO: try without the angles;
    # TODO: try with only the angles (based on neighbor list??)
    # TODO: try with both distances and angles

    inv_data = []
    for step_data in data['R']:
        pos_vectors, dist_matrix = ase.geometry.get_distances(step_data)
        print(pos_vectors)
        # TODO: only upper triangle
        print(dist_matrix)
        flat_dist = dist_matrix.flatten()
        print(flat_dist)

        # TODO: what is this actually producing??
        # should I be using get_dihedrals
        angle_matrix = ase.geometry.get_angles(step_data, step_data)
        flat_angle = angle_matrix.flatten()
        print(flat_angle)

        combined_vars = np.append(flat_dist, angle_matrix)
        inv_data.append(combined_vars)
        break
    # TODO: use more than just [0]
    print(inv_data)

    print('\nangles example\n')
    example_data = np.array([[2,1,0], [1,1,0], [1,2,0]])
    # pair_atoms = [[[example_data[i],example_data[j]] for j in range(i+1,len(example_data))] for i in range(len(example_data))]
    pairs_A, pairs_B = paired_atoms(example_data)
    print('paired atoms')
    print([[pairs_A[i], pairs_B[i]] for i in range(len(pairs_A))])
    print('angles')
    print(ase.geometry.get_angles(pairs_A, pairs_B))

    print('\nvectors')
    print(ase.geometry.get_distances(example_data))

    # WRONG
    # print('\ntrio atoms')
    # trio_A, trio_B, trio_C = trio_atoms(example_data)
    # print([[trio_A[i], trio_B[i], trio_C[i]] for i in range(len(trio_A))])
    # print('dihedrals')
    # print(ase.geometry.get_dihedrals(trio_A, trio_B, trio_C))

    return np.array(inv_data)

def convert_to_invariant_dists(data):
    inv_data = []
    for step_data in data['R']:
        pos_vectors, dist_matrix = ase.geometry.get_distances(step_data)
        # Only use upper triangle to avoid repeated information
        flat_dist = dist_matrix[np.triu_indices(len(dist_matrix), k = 1)]
        inv_data.append(flat_dist)
    # print(inv_data)

    return np.array(inv_data)

# Function to plot arrows
def plot_arrows(ax, mean, vectors, color, label):
    u = []
    v = []
    w = []

    for vec in vectors:
        u.append(vec[0])
        v.append(vec[1])
        w.append(vec[2])
    
    num_arrows = len(vectors)
    x, y, z = np.meshgrid(mean[0],
                          mean[1],
                          mean[2])
    ax.quiver(x, y, z, u, v, w, color=color, label=label)

if __name__ == "__main__":
    # args = sys.argv[1:]
    # task = args.pop(0)

    data = load_data('/Users/chemchair/Documents/GitHub/H2combustion_CVs/H2COMBUSTION_DATA-main-2/01_aimd.npz')
    inv_data = convert_to_invariant_dists(data)

    print(inv_data.shape)
    
    # example_data = {'R': np.array([[[2,1,0], [1,1,0], [1,2,0]]])}
    # example = convert_to_invariant_dists(example_data)

    # split the array into (possibly not exactly equal) portions
    partitioned_data = np.array_split(inv_data, 100)

    # choose one set of 100 data points to test on that isn't just the first one
    test_data = partitioned_data[3]

    # set up PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(test_data)
    # means=pca.mean_[:1]
    means=pca.mean_
    # comps=pca.components_[:,:1]
    comps=pca.components_
    print(means)
    print(pca.mean_)
    print(comps)
    print(pca.components_)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = test_data[:,0]
    ys = test_data[:,1]
    zs = test_data[:,2]
    ax.scatter(xs, ys, zs)

    # Plot PCA arrows
    plot_arrows(ax, means, comps, 'red', 'PCA Components')

    plt.show()