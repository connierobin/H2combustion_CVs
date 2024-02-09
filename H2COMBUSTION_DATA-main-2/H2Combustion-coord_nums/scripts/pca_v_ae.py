import sys

from matplotlib import pyplot as plt
import numpy as np

import ase
from ase import Atoms
from ase.visualize import view
from ase.visualize.plot import plot_atoms

import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

from combust.utils.utility import check_data_consistency
from combust.utils.utility import get_data_remove_appended_zero

atom_with_number = ['N/A', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F']

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

    #     nums : ndarray, shape=(M, N)
    #     The 2D array of atomic numbers for M data points and N atoms
    Z_data = data['Z']
    print(f"Z shape: {Z_data.shape}")
    print(Z_data[0])
    
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

def run_senwei_pca(data):
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
    k = 1  # Set the desired number of components
    
    # eigenvectors = eigenvectors/np.linalg.norm(eigenvectors,axis=0,keepdims=True)
    # Step 4.7: Retain the top k components
    selected_eigenvectors = eigenvectors[:, 0:1]
    # print(selected_eigenvectors.shape)
    # selected_eigenvectors = np.array([[1], [0]])
    # Step 4.8: Transform your data to the new lower-dimensional space
    transformed_data = np.dot(centered_data, selected_eigenvectors)

    # print(np.dot(centered_data, selected_eigenvectors)-np.matmul(centered_data, selected_eigenvectors))
    return mean_vector, selected_eigenvectors

def run_skl_pca(data):
    # set up PCA
    pca = PCA(n_components=2)
    pca.fit_transform(data)
    # means=pca.mean_[:1]
    means=pca.mean_
    # comps=pca.components_[:,:1]
    comps=pca.components_
    return (means, comps)

def AE(data):
    input_dim = data.shape[1]
    print(f'input dim: {input_dim}')
    encoding_dim = 2 # Set the desired encoding dimension
    intermediate_dim = 64 # Set the width of the intermediate layer

    # Define the Autoencoder architecture
    input_layer = Input(shape=(input_dim,))
    encoder = Sequential([Dense(intermediate_dim, activation='relu'),
                          Dense(encoding_dim)])
    encoded = encoder(input_layer)
    decoded = tf.math.abs(Sequential([Dense(intermediate_dim, activation='relu'),
                          Dense(input_dim)
                          ])(encoded))

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train the Autoencoder
    autoencoder.fit(data, data, epochs=500, batch_size=32, shuffle=True, validation_split=0.2)
    
    # Get out base vectors to plot
    base_vectors = np.identity(input_dim)
    encoded_base_vectors = encoder.predict(base_vectors)

    ae_comps = encoded_base_vectors.T[:,0:3]

    return ae_comps

def run_ae(data):
    # TODO: put in normalization?
    input_dim = data.shape[1]
    print(f'input dim: {input_dim}')
    encoding_dim = 2 # Set the desired encoding dimension
    intermediate_dim = 32 # Set the width of the intermediate layer

    # Define the Autoencoder architecture
    input_layer = Input(shape=(input_dim,))
    encoder = Sequential([Dense(intermediate_dim, activation='relu'),
                          Dense(encoding_dim)])
    encoded = encoder(input_layer)
    decoded = tf.math.abs(Sequential([Dense(intermediate_dim, activation='relu'),
                          Dense(input_dim)
                          ])(encoded))

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train the Autoencoder
    duplicate_data = False
    if (duplicate_data):
        autoencoder.fit(np.tile(data, (10,1)), np.tile(data, (10,1)), epochs=500, batch_size=32, shuffle=True, validation_split=0.2)
    else:
        autoencoder.fit(data, data, epochs=500, batch_size=32, shuffle=True, validation_split=0.2)
    
    # Get out base vectors to plot
    # TODO: if I'm normalizing all my data I should also normalize this
    base_vectors = np.identity(input_dim)
    # Define the encoder model
    encoded_base_vectors = encoder.predict(base_vectors)

    # print(encoded_base_vectors.mean(axis=1))
    ae_means = encoded_base_vectors.mean(axis=0)    # Don't want this???
    ae_comps = encoded_base_vectors.T[:,0:input_dim]

    return ae_means, ae_comps

def test_3_atom():
    data = load_data('../01_aimd.npz')
    inv_data = convert_to_invariant_dists(data)

    atom_names = [atom_with_number[int(z)] for z in data['Z'][0]]
    pair_names = [f'{a}-{b}' for idx, a in enumerate(atom_names) for b in atom_names[idx + 1:]]
    
    example_data = {'R': np.array([[[2,1,0], [1,1,-1], [1,2,1]]])}
    example = convert_to_invariant_dists(example_data)

    # split the array into (possibly not exactly equal) portions
    partitioned_data = np.array_split(inv_data, 100)

    # choose one set of 100 data points to test on that isn't just the first one
    test_data = partitioned_data[2]
    # test_data = example_data['R'][0]

    # NOTE: Connie confirmed that Senwei's PCA results in the same vector (including magnitude) as the scikit-learn PCA. 
    pca_means, pca_comps = run_skl_pca(test_data)

    ae_means, ae_comps = run_ae(test_data)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = test_data[:,0]
    ys = test_data[:,1]
    zs = test_data[:,2]
    ax.scatter(xs, ys, zs)
    ax.set_xlabel(pair_names[0])
    ax.set_ylabel(pair_names[1])
    ax.set_zlabel(pair_names[2])

    # Plot PCA arrows
    plot_arrows(ax, pca_means, pca_comps, 'red', 'PCA Components')

    # Plot AE arrows    
    plot_arrows(ax, pca_means, ae_comps, 'blue', 'Autoencoder Vectors')

    plt.legend()

    plt.show()

def test_n_atom():
    # FUNCTION UNDER CONSTRUCTION
    # Plot using t-sne?
    # Plot using atomic forces?
    # Make Atoms object?

    data = load_data('../01_aimd.npz')
    inv_data = convert_to_invariant_dists(data)

    atom_names = [atom_with_number[int(z)] for z in data['Z'][0]]
    combined_atom_names = "".join([str(i) for i in atom_names])
    pair_names = [f'{a}-{b}' for idx, a in enumerate(atom_names) for b in atom_names[idx + 1:]]

    # TODO: draw atoms in 3d space, then draw arrows based on the calculated forces
    mols = [Atoms(combined_atom_names, positions=data['R'][i]) for i in range(len(data['R']))]
    view(mols[650], viewer='vmd')
    fig, ax = plt.subplots()
    # plot_atoms(mols[650], ax, rotation=('90x,45y,56z'))
    plt.show()

    return

# Function to plot arrows
def plot_arrows(ax, mean, vectors, color, label):
    u = []
    v = []
    w = []
    x = []
    y = []
    z = []

    for vec in vectors:
        u.append(vec[0])
        v.append(vec[1])
        w.append(vec[2])
        x.append(mean[0] - vec[0]/2)
        y.append(mean[1] - vec[1]/2)
        z.append(mean[2] - vec[2]/2)
    
    ax.quiver(x, y, z, u, v, w, color=color, label=label)

if __name__ == "__main__":
    # args = sys.argv[1:]
    # task = args.pop(0)

    test_3_atom()

    # test_n_atom()