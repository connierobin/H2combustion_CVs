import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import RegularGridInterpolator
import numpy.linalg as la
from tqdm import tqdm

import time
from pyscf import gto, dft, scf, mp, cc

'''
(1) Load the potential energy surface on the grid points
(2) Define an interpolator to get the energy values on grid points 
'''

datadict_PES = scipy.io.loadmat('R09PES3D.mat')
max_PES = np.max(datadict_PES['V'])
box = np.zeros((3, 2))
for idx, item in enumerate(['X', 'Y', 'Z']):
    box[idx, 0] = np.min(datadict_PES[item])
    box[idx, 1] = np.max(datadict_PES[item])

# Create a 3D grid
X, Y, Z = np.meshgrid(datadict_PES['X'], datadict_PES['Y'], datadict_PES['Z'], indexing='ij')
XYZ_ = np.concatenate((np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1)), np.reshape(Z, (-1, 1))), axis=1)
V = datadict_PES['V']
V = V.reshape(-1)

# Create a RegularGridInterpolator
interpolator = RegularGridInterpolator((datadict_PES['X'][:, 0], datadict_PES['Y'][:, 0], datadict_PES['Z'][:, 0]),
                                       datadict_PES['V'])

# global cc_time = 0

# Outside the grid, we add the harmonic potential to avoid a configuration going out
def is_inside_box(x, box):
    if len(x) != len(box):
        print(x.shape, box.shape)
        raise ValueError("Vector dimensions must match the number of box dimensions")

    for i in range(len(x)):
        if not (box[i][0] < x[i] < box[i][1]):
            return False
    return True


def harmonic_potential_outside_box(x, box):
    if len(x) != len(box):
        raise ValueError("Vector dimensions must match the number of box dimensions")
    harmonic_potential = 0
    coef = 10000  ### huge potential outside the grid box
    harmonic_potential = np.sum(coef * (np.maximum(box[:, 0] - x, 0)) ** 2 + coef * (np.maximum(x - box[:, 1], 0)) ** 2)
    return harmonic_potential

def test_cc_time(x):
    global cc_time
    atoms = ['H', 'O', 'H']
    # print(x.shape)
    # NOTE: WHEN using O, O, H instead of H, O, H: the spin = 3 here is a guess, I don't actually know what the spin should be
    mol = gto.M(atom = f'{atoms[0]} {x[0]} {x[1]} {x[2]}; {atoms[1]} {x[3]} {x[4]} {x[5]}; {atoms[2]} {x[6]} {x[7]} {x[8]}', basis = 'ccpvdz')
    cc_start = time.time()
    ccsd_mol = mol.RHF().run()
    ccsd_mol = ccsd_mol.CCSD().run()
    cc_end = time.time()
    cc_time = cc_time + (cc_end - cc_start)
    return

def test_energy_calculators(x, ref_energy, ref_time):
    atoms = ['O', 'O', 'H']
    # print(x.shape)
    start_setup = time.time()
    # NOTE: the spin here is a guess, I don't actually know what the spin should be
    mol = gto.M(atom = f'{atoms[0]} {x[0]} {x[1]} {x[2]}; {atoms[1]} {x[3]} {x[4]} {x[5]}; {atoms[2]} {x[6]} {x[7]} {x[8]}', basis = 'ccpvdz', spin = 3)
    end_setup = time.time()

    # Hartree Fock
    print("Hartree Fock")
    start_hf = time.time()
    hf_mol = scf.RHF(mol)
    hf_mol.kernel()
    end_hf = time.time()

    # Default DFT
    print("DFT")
    start_default_dft = time.time()
    dft_mol = dft.RKS(mol)
    dft_mol.kernel()
    end_default_dft = time.time()

    # PBE DFT
    print("PBE DFT")
    start_pbe_dft = time.time()
    pbe_mol = dft.RKS(mol)
    pbe_mol.xc = 'pbe'
    # pbe_mol = pbe_mol.newton()  # what does this do?
    pbe_mol.kernel()
    end_pbe_dft = time.time()

    # CCSD
    print("CCSD")
    start_ccsd = time.time()
    ccsd_mol = mol.RHF().run()
    ccsd_mol = ccsd_mol.CCSD().run()
    end_ccsd = time.time()

    # MP2
    print("MP2")
    # TODO: why is this not running!!!!!
    start_mp2 = time.time()
    mp2_mol = mol.HF()
    # mp2_mol = mp2_mol.MP2().run()
    mp2_mol = mp.MP2(mp2_mol)
    mp2_mol = mp2_mol.kernel()
    end_mp2 = time.time()

    setup_time = end_setup - start_setup
    hf_time = end_hf - start_hf
    default_dft_time = end_default_dft - start_default_dft
    pbe_dft_time = end_pbe_dft - start_pbe_dft
    ccsd_time = end_ccsd - start_ccsd
    mp2_time = end_mp2 - start_mp2

    # print(f'Setup time: {setup_time}')
    # print(f'HF time: {hf_time}')
    # print(f'Default DFT time: {default_dft_time}')
    # print(f'PBE DFT time: {pbe_dft_time}')
    # print(f'CCSD time: {ccsd_time}')
    # print(f'MP2 time: {mp2_time}')

    methods = ['Reference', 'HF', 'Default DFT', 'PBE', 'CCSD', 'MP2']
    times = [ref_time, hf_time, default_dft_time, pbe_dft_time, ccsd_time, mp2_time]
    energies = [ref_energy, hf_mol.e_tot, dft_mol.e_tot, pbe_mol.e_tot, ccsd_mol.e_tot, mp2_mol.e_tot]

    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(len(methods)):
        print(times[i])
        print(energies[i])
        ax.scatter([times[i]], [energies[i]], label=methods[i])
    plt.legend()
    plt.xlabel('Calculation Time')
    plt.ylabel('Energy')
    plt.show()

def potential(x):
    z = cart2zmat(x)

    start_interpolator = time.time()
    if is_inside_box(z, box):
        result = interpolator(z.reshape(1, -1))
    else:
        distances = np.linalg.norm(z.reshape(1, -1) - XYZ_, axis=1)
        minidx = np.argmin(distances)
        result = V[minidx] + harmonic_potential_outside_box(z.reshape(-1), box)
    end_interpolator = time.time()
    interpolator_time = end_interpolator - start_interpolator

    # Comment out these lines to stop testing
    # test_energy_calculators(x[0], result, interpolator_time)
    # test_cc_time(x[-1])

    return result


'''
Transformation between Cartesian coordinates and internal coordinates
'''


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
    sint = la.norm(np.cross(rij, rkj))
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
                distmat[ia, jb] = la.norm(xyza - xyzb)
        distmat = distmat + np.transpose(distmat)

        if na > 1:
            rlist.append(distmat[0, 1])

        if na > 2:
            rlist.append(distmat[0, 2])
            alist.append(torsion(X[:, j], 3, 1, 2))

        Z.append(rlist + alist + dlist)

    Z = np.array(Z)

    return Z.T


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

    # # Step 4.8: Transform your data to the new lower-dimensional space
    # transformed_data = np.dot(centered_data, selected_eigenvectors)

    return mean_vector, std_vector, selected_eigenvectors, eigenvalues


'''
Finite difference to compute the gradient with respect to the Cartesian coordinates
'''


def gradV(x):  # x shape 1*9
    eps = 1e-3
    f = potential(x)
    grad = np.zeros_like(x)

    for i in range(x.shape[1]):
        x_plus_h = x.copy()
        x_plus_h[0, i] += eps
        f_plus_h = potential(x_plus_h)
        grad[0, i] = (f_plus_h - f) / eps

    return grad


def Gaussian(x, qsz, choose_eigenvalue, save_eigenvector_s, height, sigma):

    x_z = cart2zmat(x).T
    dist = np.sum(np.expand_dims((x_z - qsz), axis=2) * save_eigenvector_s, axis=1) ** 2
    exps = np.exp(-np.sum(dist * choose_eigenvalue, axis=1) / 2 / sigma ** 2)
    V = height * np.sum(exps)

    return V


def gradGaussians(x, qsz, choose_eigenvalue, save_eigenvector_s, height, sigma):
    eps = 1e-2
    f = Gaussian(x, qsz, choose_eigenvalue, save_eigenvector_s, height, sigma)
    grad = np.zeros_like(x)

    for i in range(x.shape[1]):
        x_plus_h = x.copy()
        x_plus_h[0, i] += eps
        f_plus_h = Gaussian(x_plus_h, qsz, choose_eigenvalue, save_eigenvector_s, height, sigma)
        grad[0, i] = (f_plus_h - f) / eps

    return grad


def MD(q0, T, Tdeposite, height, sigma, dt=1e-3, beta=1.0, coarse=1):
    '''
    :param q0: Initial configuration
    :param T:  Total time for simulation
    :return:
    '''
    Nsteps = round(T / dt)   ## Total steps
    NstepsDeposite = round(Tdeposite / dt)   ## Number of step to deposit Gaussian
    NstepsSave = round(T / (dt * coarse))   ## Number of step to save a configuration every time
    trajectories = np.zeros((NstepsSave + 1, 1, 9))  ## Trajectory to save
    trajectories_PCA = np.zeros((NstepsDeposite, 1, 3)) ## Configuration to perform PCA

    q = q0

    qsz = None ## Gaussian Centers
    save_eigenvector = None ## Eigenvectors
    save_eigenvalue = None ## Eigenvectors
    chosen_eigenvector = None ## each row represents the which eigenvector is chosen

    variance = 0.8 ## The variance threshold to keep some eigenvector out of all eigenvectors

    for i in tqdm(range(Nsteps)):

        q = next_step(q, qsz, chosen_eigenvector, save_eigenvector, height, sigma, dt, beta)  # 1*9
        trajectories_PCA[i % NstepsDeposite, :] = cart2zmat(q).T

        if i % coarse == 0:
            trajectories[i // coarse, :] = q

        if (i + 1) % NstepsDeposite == 0:
            if qsz is None:

                ### conducting PCA ###
                data = trajectories_PCA  # (N_steps, 1, 3)
                data = np.squeeze(data, axis=1)  # (N_steps, 3)
                mean_vector, std_vector, selected_eigenvectors, eigenvalues = PCA(data)
                qsz = mean_vector

                ### reset the PCA dataset
                trajectories_PCA = np.zeros((NstepsDeposite, 1, 3))

                save_eigenvector = np.expand_dims(selected_eigenvectors, axis=0)
                save_eigenvalue = np.expand_dims(eigenvalues, axis=0)

                chosen_eigenvector_tmp = np.zeros((1, 3))
                cumsum = np.cumsum(save_eigenvalue, axis=1)

                var_ratio = cumsum / np.sum(save_eigenvalue)
                idx = np.argmax(var_ratio > variance)

                for s in range(idx + 1):
                    chosen_eigenvector_tmp[0, s] = 1
                chosen_eigenvector = chosen_eigenvector_tmp

            else:
                ### conducting PCA ###
                data = trajectories_PCA  # (N_steps, 1, 3)
                data = np.squeeze(data, axis=1)  # (N_steps, 3)

                mean_vector, std_vector, selected_eigenvectors, eigenvalues = PCA(data)
                q_deposit = mean_vector

                qsz = np.concatenate([qsz, q_deposit], axis=0)

                save_eigenvector = np.concatenate([save_eigenvector, np.expand_dims(selected_eigenvectors, axis=0)], axis=0)
                save_eigenvalue = np.concatenate([save_eigenvalue, np.expand_dims(eigenvalues, axis=0)], axis=0)

                ### check which eigenvector will be used
                eigenvalues = np.expand_dims(eigenvalues, axis=0)
                chosen_eigenvector_tmp = np.zeros((1, 3))
                cumsum = np.cumsum(eigenvalues, axis=1)
                var_ratio = cumsum / np.sum(eigenvalues)
                idx = np.argmax(var_ratio > variance)

                for s in range(idx + 1):
                    chosen_eigenvector_tmp[0, s] = 1
                chosen_eigenvector = np.concatenate([chosen_eigenvector, chosen_eigenvector_tmp], axis=0)

                trajectories_PCA = np.zeros((NstepsDeposite, 1, 3))
    trajectories[NstepsSave, :] = q
    return trajectories, qsz, save_eigenvector, save_eigenvalue


def next_step(qnow, qsz, chosen_eigenvector, saved_eigenvectors, height, sigma, dt=1e-3, beta=1.0):
    '''
    :param qnow: current configuration
    :param qsz:  Gaussian center (internal coordinate)
    :param choose_eigenvector: Eigenvector chosen  [1,0,0] represents we only choose the first eigenvector out of three eigenvectors
    :param saved_eigenvectors: All eigenvectors
    :param height: Gaussian height
    :param sigma: Gaussian width
    :param dt: step size
    :param beta: inverse temperature
    :return: the configuration of next time step
    '''
    if qsz is None: ### if no Gaussian is deposited
        qnext = qnow + (- gradV(qnow)) * dt + np.sqrt(2 * dt / beta) * np.random.randn(*qnow.shape)
    else: ### if some Gaussian is deposited
        qnext = qnow + (- (gradV(qnow) + gradGaussians(qnow, qsz, chosen_eigenvector, saved_eigenvectors, height, sigma))) * dt + np.sqrt(2 * dt / beta) * np.random.randn(*qnow.shape)
    return qnext


if __name__ == '__main__':

    T = .01
    cmap = plt.get_cmap('plasma')
    ircdata = scipy.io.loadmat('irc09.mat')['irc09'][0][0][3]

    # choose initial configuration
    x0 = ircdata[:, 5:6].T

    # global cc_time
    # cc_time = 0.
    coarse = 100
    for i in range(1):
        trajectories, qs, save_eigenvector_s, save_eigenvalue_s = MD(x0, T=T, Tdeposite=1e-2, height=1, sigma=0.3,
                                                                     dt=2e-5, beta=1, coarse=coarse)
    # print(cc_time)

    z_trajectory = (cart2zmat(trajectories[:, 0])).T

    indices = np.arange(z_trajectory.shape[0])
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z_trajectory[:, 0], z_trajectory[:, 1], z_trajectory[:, 2], c=indices, cmap=cmap)
    plt.show()
