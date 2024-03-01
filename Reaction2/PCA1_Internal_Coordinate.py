import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import RegularGridInterpolator
import numpy.linalg as la
from tqdm import tqdm


data = np.loadtxt('../../H_reaction2/R02_CV_vs_energies.mpg')
X = np.reshape(data[:,0], (10,10,10))
Y = np.reshape(data[:,1], (10,10,10))
Z = np.reshape(data[:,2], (10,10,10))

box = np.zeros((3, 2))
for idx, item in enumerate([X, Y, Z]):
    box[idx, 0] = np.min(item)
    box[idx, 1] = np.max(item)

# Create a 3D grid
# X, Y, Z = np.meshgrid(np.linspace(), datadict_PES['Y'], datadict_PES['Z'], indexing='ij')
npts = 10
X_range = np.linspace(box[0, 0], box[0, 1], num=npts)
Y_range = np.linspace(box[1, 0], box[1, 1], num=npts)
Z_range = np.linspace(box[2, 0], box[2, 1], num=npts)
X_, Y_, Z_ = np.meshgrid(X_range, Y_range, Z_range)
E_ = np.zeros_like(X_)
for i in range(npts):
    for j in range(npts):
        for k in range(npts):
            x = X_[i,j,k]
            y = Y_[i,j,k]
            z = Z_[i,j,k]
            E_[i,j,k] = data[np.argmin((data[:,0]-x)**2 + (data[:,1]-y)**2 + (data[:,2]-z)**2),3]

eH=-0.5004966690
eO=-75.0637742413
be = E_ - (2*eH+eO)
be = be*627.51
E_ = be

XYZ_ = np.concatenate((np.reshape(X_,(-1,1)), np.reshape(Y_,(-1,1)), np.reshape(Z_,(-1,1))), axis=1)
V = E_.reshape(-1)

# Create a RegularGridInterpolator
interpolator = RegularGridInterpolator((X_range, Y_range, Z_range), E_)


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
    coef = 10000
    harmonic_potential = np.sum(coef * (np.maximum(box[:, 0] - x, 0)) ** 2 + coef * (np.maximum(x - box[:, 1], 0)) ** 2)

    return harmonic_potential


def potential(x):
    z = cart2zmat(x)

    if is_inside_box(z, box):
        return interpolator(z.reshape(1, -1))
    else:
        # print(x.reshape(1,-1).shape, XYZ_.shape)
        distances = np.linalg.norm(z.reshape(1, -1) - XYZ_, axis=1)
        minidx = np.argmin(distances)
        return V[minidx] + harmonic_potential_outside_box(z.reshape(-1), box)


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
    # print('PCA', data.shape, cart2zmat(data).shape)
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

    # Step 4.7: Retain the top k components
    selected_eigenvectors = eigenvectors[:, 0:k]
    # print(selected_eigenvectors.shape, 'eigen')
    # selected_eigenvectors = np.array([[1], [0]])
    # Step 4.8: Transform your data to the new lower-dimensional space
    transformed_data = np.dot(centered_data, selected_eigenvectors)
    # print('selected_eigenvector:', data.shape, data_z.shape, eigenvectors.shape, selected_eigenvectors.shape)
    # print(np.dot(centered_data, selected_eigenvectors)-np.matmul(centered_data, selected_eigenvectors))
    return mean_vector, std_vector, selected_eigenvectors, eigenvalues


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


def Gaussian(x, qx, qstd, qsz, method, save_eigenvector_s, height, sigma):
    # x the point we what calculate the value
    # gaussian deposit
    # method the way that we want deposit gaussian
    V = None
    if method == 'second_bond':
        # print(x.shape, qx.shape)
        q_z = cart2zmat(x).T
        qs_z = qsz  # cart2zmat(qx).T
        # print(q_z.shape, qs_z.shape)
        V = height * np.sum(np.exp(-(q_z[0, 1] - qs_z[:, 1]) ** 2 / 2 / sigma ** 2))
    elif method == 'first_three_eigen':
        # print('qx shape', qx.shape, save_eigenvector_s.shape)
        x_z = cart2zmat(x).T
        # print('eigen: ', x_z.shape, qsz.shape, save_eigenvector_s)
        dist = np.sum(np.expand_dims((x_z - qsz), axis=2) * save_eigenvector_s, axis=1) ** 2
        # local_bool = (np.sum((x_z - qsz) ** 2, axis=-1) < threshold ** 2)
        # print(dist.shape, (x-qx).shape)
        exps = np.exp(-np.sum(dist, axis=1) / 2 / sigma ** 2)
        # print(dist, np.exp(-np.sum(dist, axis=1)/2/sigma**2), exps, local_bool, np.sum((x - qx)**2, axis=-1))
        # exps = np.exp(np.sum(dist, axis=1)/2/sigma**2)
        # print(local_bool)
        # print(local_bool.shape, exps.shape, np.sum(local_bool))
        # print(height, sigma, exps)
        V = height * np.sum(exps)
        # print('hehe')
        # print(x.shape)
        # print(qx.shape)  #(1, 9)
        # print(save_eigenvector_s.shape) #(1, 9, 3)
    return V


def gradGaussians(x, qs, qstd, qsz, method, save_eigenvector_s, height, sigma):
    eps = 1e-2
    f = Gaussian(x, qs, qstd, qsz, method, save_eigenvector_s, height, sigma)
    grad = np.zeros_like(x)

    for i in range(x.shape[1]):
        x_plus_h = x.copy()
        x_plus_h[0, i] += eps
        f_plus_h = Gaussian(x_plus_h, qs, qstd, qsz, method, save_eigenvector_s, height, sigma)
        grad[0, i] = (f_plus_h - f) / eps
        # print(f_plus_h, f)
    return grad


def MD(q0, T, Tdeposite, height, sigma, dt=1e-3, beta=1.0, coarse=1):
    Nsteps = round(T / dt)
    NstepsDeposite = round(Tdeposite / dt)
    NstepsSave = round(T / (dt * coarse))
    trajectories = np.zeros((NstepsSave + 1, 1, 9))
    trajectories_PCA = np.zeros((NstepsDeposite, 1, 3))

    q = q0
    qs = None
    qsz = None
    qstd = None
    save_eigenvector_s = None
    save_eigenvalue_s = None
    method = 'first_three_eigen'
    for i in tqdm(range(Nsteps)):
        trajectories[i // coarse, :] = q
        q = next_step(q, qs, qstd, qsz, method, save_eigenvector_s, height, sigma, dt, beta)  # 1*9
        trajectories_PCA[i % NstepsDeposite, :] = cart2zmat(q).T
        if i % coarse == 0:
            trajectories[i // coarse, :] = q

        if (i + 1) % NstepsDeposite == 0:
            if qs is None:

                ### conducting PCA ###
                data = trajectories_PCA  # (N_steps, 1, 2)
                # print(i, trajectories_PCA)
                data = np.squeeze(data, axis=1)  # (100, 2)
                mean_vector, std_vector, selected_eigenvectors, eigenvalues = PCA(data)
                # print(selected_eigenvectors.shape, eigenvalues.shape)
                qs = mean_vector  # q
                # print(mean_vector.shape)
                qsz = mean_vector  # cart2zmat(qs).T
                qstd = std_vector

                ### reset the PCA dataset
                trajectories_PCA = np.zeros((NstepsDeposite, 1, 3))

                save_eigenvector_s = np.expand_dims(selected_eigenvectors, axis=0)
                save_eigenvalue_s = np.expand_dims(eigenvalues, axis=0)

                # print(save_eigenvector_s.shape, save_eigenvalue_s.shape)
            else:
                ### conducting PCA ###
                data = trajectories_PCA  # (N_steps, 1, 2)
                # print(i, trajectories_PCA)
                data = np.squeeze(data, axis=1)  # (100, 2)
                mean_vector, std_vector, selected_eigenvectors, eigenvalues = PCA(data)

                q_deposit = q
                q_deposit = mean_vector
                # print(mean_vector.shape, 'mean vector')
                qs = np.concatenate([qs, q_deposit], axis=0)
                # qsz = np.concatenate([qsz, cart2zmat(q_deposit).T], axis=0)
                qsz = np.concatenate([qsz, q_deposit], axis=0)
                qstd = np.concatenate([qstd, std_vector], axis=0)

                save_eigenvector_s = np.concatenate([save_eigenvector_s, np.expand_dims(selected_eigenvectors, axis=0)],
                                                    axis=0)
                save_eigenvalue_s = np.concatenate([save_eigenvalue_s, np.expand_dims(eigenvalues, axis=0)], axis=0)

                # print(save_eigenvector_s.shape, save_eigenvalue_s.shape)
                # print(i, qs.shape, qsz.shape)

                # print(i, trajectories_PCA.shape, trajectories_PCA)

                trajectories_PCA = np.zeros((NstepsDeposite, 1, 3))
    trajectories[NstepsSave, :] = q
    return trajectories, qs, save_eigenvector_s, save_eigenvalue_s


def next_step(qnow, qs, qstd, qsz, method, save_eigenvector_s, height, sigma, dt=1e-3, beta=1.0):
    if qs is None:
        qnext = qnow + (- gradV(qnow)) * dt + np.sqrt(2 * dt / beta) * np.random.randn(*qnow.shape)
        # print('grad size', gradV(qnow).shape)
    else:
        # print(gradGaussians(qnow, qs, qstd, qsz, method, save_eigenvector_s, height, sigma))
        # print(qsz)
        qnext = qnow + (- (gradV(qnow) + gradGaussians(qnow, qs, qstd, qsz, method, save_eigenvector_s, height,
                                                       sigma))) * dt + np.sqrt(2 * dt / beta) * np.random.randn(
            *qnow.shape)

    return qnext


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--trial', type=int, default=0, help='trial.')
    args = parser.parse_args()

    T = 10
    cmap = plt.get_cmap('plasma')
    ircdata = scipy.io.loadmat('/global/u2/s/senwei/codes/H2data/H2COMBUST/MAT/irc02.mat')['irc02'][0][0][3]
    x0 = ircdata[:, 0:1].T  # 1*9
    print(ircdata.shape)

    coarse = 100
    for i in range(1):
        trajectories, qs, save_eigenvector_s, save_eigenvalue_s = MD(x0, T=T, Tdeposite=1e-2, height=0.02, sigma=0.1,
                                                                     dt=2e-5, beta=1, coarse=coarse)

    print(trajectories.shape)

    z_trajectory = (cart2zmat(trajectories[:, 0])).T

    indices = np.arange(z_trajectory.shape[0])
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z_trajectory[:, 0], z_trajectory[:, 1], z_trajectory[:, 2], c=indices, cmap=cmap)
    plt.show()
    np.savez('results/T10_deposit1e-2_dt2e-5-PCA1InternalCoordinates_height0p02sigma0p1_trial{}'.format(args.trial), z=z_trajectory,
             x=trajectories, qs=qs, eignvectors=save_eigenvector_s, eigenvalues=save_eigenvalue_s)

