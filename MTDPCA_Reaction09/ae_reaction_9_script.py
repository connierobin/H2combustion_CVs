import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import RegularGridInterpolator
import numpy.linalg as la
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import normalize

datadict_PES = scipy.io.loadmat('./R09PES3D.mat')
max_PES = np.max(datadict_PES['V'])
box = np.zeros((3, 2))
for idx, item in enumerate(['X', 'Y', 'Z']):
    box[idx, 0] = np.min(datadict_PES[item])
    box[idx, 1] = np.max(datadict_PES[item])

threshold = np.sqrt(1e-1)

# Create a 3D grid
X, Y, Z = np.meshgrid(datadict_PES['X'], datadict_PES['Y'], datadict_PES['Z'], indexing='ij')
XYZ_ = np.concatenate((np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1)), np.reshape(Z, (-1, 1))), axis=1)
V = datadict_PES['V']
V = V.reshape(-1)
print(datadict_PES['V'].shape)
# Create a RegularGridInterpolator
interpolator = RegularGridInterpolator((datadict_PES['X'][:, 0], datadict_PES['Y'][:, 0], datadict_PES['Z'][:, 0]), datadict_PES['V'])

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

def plot_comparison(PCA_data, AE_data):
    PCA_data = np.array(PCA_data)
    AE_data = np.array(AE_data)
    # TODO: plot a line for y=x
    fig, ax = plt.subplots()
    ax.scatter(PCA_data[:, 0], PCA_data[:, 1], label='PCA')
    ax.scatter(AE_data[:, 0], AE_data[:, 1], label='AE')

    # lims = [
    #     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    #     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    # ]
    # ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    plt.xlabel('Variance of First CV')
    plt.ylabel('Variance of Second CV')

    # TODO: color code based on angle between vectors!!

    ax.axline((0, 0), slope=1)

    ax.legend()

    ax.set_aspect('equal')
    buffer = 0.1
    xmin = min([min(PCA_data[:, 0]), min(AE_data[:, 0])])
    xmax = max([max(PCA_data[:, 0]), max(AE_data[:, 0])])
    ymin = min([min(PCA_data[:, 1]), min(AE_data[:, 1])])
    ymax = max([max(PCA_data[:, 1]), max(AE_data[:, 1])])
    all_min = min([xmin, ymin])
    all_max = max([xmax, ymax])
    axis_length = all_max - all_min
    # all_min = all_min +  axis_length * buffer
    all_min = 0
    all_max = all_max + axis_length * buffer
    ax.set_xlim([all_min, all_max])
    ax.set_ylim([all_min, all_max])

    plt.show()

def PCA(data): # datasize: N * dim
    # Step 4.1: Compute the mean of the data
    
    data_z = data # bs*3
    
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
    k = 3  # Set the desired number of components

    # Step 4.7: Retain the top k components
    selected_eigenvectors = eigenvectors[:, 0:k]

    # Step 4.8: Transform your data to the new lower-dimensional space
    transformed_data = np.dot(centered_data, selected_eigenvectors)
    # print('selected_eigenvector:', data.shape, data_z.shape, eigenvectors.shape, selected_eigenvectors.shape)
    # print(np.dot(centered_data, selected_eigenvectors)-np.matmul(centered_data, selected_eigenvectors))
    
    # reproduces eigenvalues
    # print('eigenvalues')
    # print(eigenvalues)
    # print('one eigenvector')
    # print(np.dot(selected_eigenvectors[:, 0], np.dot(covariance_matrix, selected_eigenvectors[:, 0])))
    # print('all eigenvectors')
    # print([np.dot(selected_eigenvectors[:, i], np.dot(covariance_matrix, selected_eigenvectors[:, i])) for i in range(len(selected_eigenvectors[0]))])
    
    return mean_vector, std_vector, selected_eigenvectors, eigenvalues

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

def gradV(x): # x shape 1*9
    eps = 1e-3
    f = potential(x)
    grad = np.zeros_like(x)

    for i in range(x.shape[1]):
        x_plus_h = x.copy()
        x_plus_h[0,i] += eps
        f_plus_h = potential(x_plus_h)
        grad[0, i] = (f_plus_h - f) / eps

    return grad

def Gaussian(x, qx, qstd, qsz, method, choose_eigenvalue, save_eigenvector_s, height, sigma):
    # x the point we what calculate the value
    # gaussian deposit
    # method the way that we want deposit gaussian
    V = None
    if method == 'second_bond':
        # print(x.shape, qx.shape)
        q_z = cart2zmat(x).T
        qs_z = qsz # cart2zmat(qx).T
        # print(q_z.shape, qs_z.shape)
        V = height * np.sum(np.exp(-(q_z[0,1] - qs_z[:,1])**2/2/sigma**2))
    elif method == 'first_three_eigen':
        # print('qx shape', qx.shape, save_eigenvector_s.shape)
        x_z = cart2zmat(x).T
        # print('eigen: ', x_z.shape, qsz.shape, save_eigenvector_s.shape) # 1*3, gn*3, gn*3*k

        # each distance vector from x to q, multiplied against the eigenvectors
        # TODO: print the shapes to understand this better
        dist = np.sum(np.expand_dims((x_z - qsz), axis=2)*save_eigenvector_s, axis=1)**2
        # local_bool = (np.sum((x_z - qsz)**2, axis=-1) < threshold**2)
        # print(dist.shape) # gn*k
        # print(choose_eigenvalue.shape, dist.shape)
        exps = np.exp(-np.sum(dist*choose_eigenvalue, axis=1)/2/sigma**2)
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


def gradGaussians(x, qs, qstd, qsz, method, choose_eigenvalue, save_eigenvector_s, height, sigma):
    eps = 1e-2
    f = Gaussian(x, qs, qstd, qsz, method, choose_eigenvalue, save_eigenvector_s, height, sigma)
    grad = np.zeros_like(x)

    for i in range(x.shape[1]):

        x_plus_h = x.copy()
        x_plus_h[0, i] += eps
        f_plus_h = Gaussian(x_plus_h, qs, qstd, qsz, method, choose_eigenvalue, save_eigenvector_s, height, sigma)
        grad[0, i] = (f_plus_h - f) / eps
        # print(f_plus_h, f)
    return grad


def MD(q0, T, Tdeposite, height, sigma, dt=1e-3, beta=1.0, coarse=1, ic_method='PCA'):
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
    choose_eigenvalue = None
    method = 'first_three_eigen'
    variance = 0.8

    compare_variances_PCA = []
    compare_variances_AE = []

    for i in tqdm(range(Nsteps)):
        trajectories[i // coarse, :] = q
        q = next_step(q, qs, qstd, qsz, method, choose_eigenvalue, save_eigenvector_s, height, sigma, dt, beta) # 1*9
        trajectories_PCA[i % NstepsDeposite, :] = cart2zmat(q).T
        if i % coarse == 0:
                trajectories[i // coarse, :] = q

        if (i + 1) % NstepsDeposite == 0:
            if qs is None:

                ### conducting PCA ###
                data = trajectories_PCA  # (N_steps, 1, 2)
                # print(i, trajectories_PCA)
                data = np.squeeze(data, axis=1)  # (100, 2)

                if ic_method == 'PCA':
                    mean_vector, std_vector, selected_eigenvectors, eigenvalues = PCA(data)
                elif ic_method == 'compare':
                    mean_vector, std_vector, selected_eigenvectors, eigenvalues, GS_eigenvectors, GS_eigenvalues = AE(data)
                    compare_variances_AE.append(eigenvalues)
                    mean_vector, std_vector, selected_eigenvectors, eigenvalues = PCA(data)
                    compare_variances_PCA.append(eigenvalues)
                else:
                    mean_vector, std_vector, selected_eigenvectors, eigenvalues, GS_eigenvectors, GS_eigenvalues = AE(data)

                # print(selected_eigenvectors.shape, eigenvalues.shape)
                qs = mean_vector#q
                # print(mean_vector.shape)
                qsz = mean_vector#cart2zmat(qs).T
                qstd = std_vector
                
                ### reset the PCA dataset
                trajectories_PCA = np.zeros((NstepsDeposite, 1, 3))
                
                save_eigenvector_s = np.expand_dims(selected_eigenvectors, axis=0)
                if ic_method == 'PCA' or ic_method == 'compare':
                    save_eigenvalue_s = np.expand_dims(eigenvalues, axis=0)
                else:
                    save_eigenvalue_s = np.expand_dims(GS_eigenvalues, axis=0)
                
                choose_eigenvalue_tmp = np.zeros((1,3))
                cumsum = np.cumsum(save_eigenvalue_s, axis=1)
                # Calculate CDF = cumulative distribution function
                var_ratio = cumsum/np.sum(save_eigenvalue_s)
                idx = np.argmax(var_ratio>variance)
                # print(save_eigenvalue_s, cumsum/np.sum(save_eigenvalue_s), np.argmax(var_ratio>variance))
                for s in range(idx+1):
                    choose_eigenvalue_tmp[0,s]=1
                choose_eigenvalue = choose_eigenvalue_tmp
                # for k in range(3):
                #     cumsum = np.cumsum(save_eigenvalue_s, axis=1)
                    
                # print(save_eigenvector_s.shape, save_eigenvalue_s.shape)
            else:
                ### conducting PCA ###
                data = trajectories_PCA  # (N_steps, 1, 2)
                # print(i, trajectories_PCA)
                data = np.squeeze(data, axis=1)  # (100, 2)
                if ic_method == 'PCA':
                    mean_vector, std_vector, selected_eigenvectors, eigenvalues = PCA(data)
                elif ic_method == 'compare':
                    mean_vector, std_vector, selected_eigenvectors, eigenvalues, GS_eigenvectors, GS_eigenvalues = AE(data)
                    compare_variances_AE.append(GS_eigenvalues)
                    mean_vector, std_vector, selected_eigenvectors, eigenvalues = PCA(data)
                    compare_variances_PCA.append(eigenvalues)
                else:
                    mean_vector, std_vector, selected_eigenvectors, eigenvalues, GS_eigenvectors, GS_eigenvalues = AE(data)

                q_deposit = q
                q_deposit = mean_vector
                # print(mean_vector.shape, 'mean vector')
                qs = np.concatenate([qs, q_deposit], axis=0)
                # qsz = np.concatenate([qsz, cart2zmat(q_deposit).T], axis=0)
                qsz = np.concatenate([qsz, q_deposit], axis=0)
                qstd = np.concatenate([qstd, std_vector], axis=0)
                
                save_eigenvector_s = np.concatenate([save_eigenvector_s, np.expand_dims(selected_eigenvectors, axis=0)], axis=0)
                save_eigenvalue_s = np.concatenate([save_eigenvalue_s, np.expand_dims(eigenvalues, axis=0)], axis=0)
                
                if method == 'PCA':
                    eigenvalues = np.expand_dims(eigenvalues, axis=0)
                    choose_eigenvalue_tmp = np.zeros((1,3))
                    cumsum = np.cumsum(eigenvalues, axis=1)
                    var_ratio = cumsum/np.sum(eigenvalues)
                    idx = np.argmax(var_ratio>variance)
                    # print(eigenvalues, var_ratio, np.argmax(var_ratio>variance))
                    for s in range(idx+1):
                        choose_eigenvalue_tmp[0,s]=1
                    choose_eigenvalue = np.concatenate([choose_eigenvalue, choose_eigenvalue_tmp], axis=0)
                else:
                    # for the AE method, choose all the eigenvectors
                    choose_eigenvalue = np.concatenate([choose_eigenvalue, np.ones((1,len(eigenvalues)))], axis=0)
                # print(choose_eigenvalue)
                
                # print(save_eigenvector_s.shape, save_eigenvalue_s.shape)
                # print(i, qs.shape, qsz.shape)

                # print(i, trajectories_PCA.shape, trajectories_PCA)

                trajectories_PCA = np.zeros((NstepsDeposite, 1, 3))
    trajectories[NstepsSave, :] = q

    if ic_method == 'compare':
        print('comparing')
        plot_comparison(compare_variances_PCA, compare_variances_AE)

    return trajectories, qs, save_eigenvector_s, save_eigenvalue_s


def next_step(qnow, qs, qstd, qsz, method, choose_eigenvalue, save_eigenvector_s, height, sigma, dt=1e-3, beta=1.0):

    if qs is None:
        qnext = qnow + (- gradV(qnow)) * dt + np.sqrt(2 * dt / beta) * np.random.randn(*qnow.shape)
        # print('grad size', gradV(qnow).shape)
    else:
        # print(gradGaussians(qnow, qs, qstd, qsz, method, save_eigenvector_s, height, sigma))
        # print(qsz)
        qnext = qnow + (- (gradV(qnow) + gradGaussians(qnow, qs, qstd, qsz, method, choose_eigenvalue, save_eigenvector_s, height, sigma))) * dt + np.sqrt(2 * dt / beta) * np.random.randn(*qnow.shape)

    return qnext


if __name__ == '__main__':

    # fig = plt.figure(figsize=(10, 6))
    # ax1 = fig.add_subplot(1, 2, 1)
    # contourf_ = ax1.contourf(X, Y, W, levels=29)

    ic_method = 'compare'
    T = 0.04

    cmap = plt.get_cmap('plasma')
    ircdata = scipy.io.loadmat('./irc09.mat')['irc09'][0][0][3]
    # x0 = ircdata[:, -2:-1].T  # 1*9
    x0 = ircdata[:, 5:6].T
    print(ircdata.shape)
    coarse = 100
    for i in range(1):
        trajectories, qs, save_eigenvector_s, save_eigenvalue_s = MD(x0, T = T, Tdeposite=1e-2, height=1, sigma=0.3, dt=2e-5, beta=1, coarse=coarse, ic_method=ic_method)

    print(trajectories.shape)

    z_trajectory = (cart2zmat(trajectories[:,0])).T

    indices = np.arange(z_trajectory.shape[0])
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z_trajectory[:, 0], z_trajectory[:, 1], z_trajectory[:, 2], c=indices, cmap=cmap)
    plt.savefig('rxn9_test.png')
    plt.show()
    np.savez(f'T10_deposit5e-3_dt2e-5-{ic_method}s-InternalCoordinates_height1_T10_debug-nonlocal', z = z_trajectory, x = trajectories, qs=qs, eignvectors=save_eigenvector_s, eigenvalues=save_eigenvalue_s)

    #     # print(trajectory.shape)
    #     indices = np.arange(trajectory.shape[0])
    #     ax1.scatter(trajectory[:, 0, 0], trajectory[:, 0, 1], c=indices, cmap=cmap)
    #
    # print(qs.shape)
    # Gs = GaussiansPCA(np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1), qs, eigenvectors, height=0.1,
    #                   sigma=0.1)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.contourf(X, Y, Gs.reshape(100, 100), levels=29)
    # indices = np.arange(qs.shape[0])
    # cmap = plt.get_cmap('plasma')
    # # ax2.scatter(qs[::-1, 0], qs[::-1, 1], c=indices, cmap=cmap)
    # # ax2.quiver(qs[:, 0], qs[:, 1], eigenvectors[0,:], eigenvectors[1,:])
    # # fig.colorbar(contourf_)
    # plt.title('Local PCA dynamics')
    # plt.show()


# if __name__ == '__main__':

#     cmap = plt.get_cmap('plasma')
#     ircdata = scipy.io.loadmat('irc09.mat')['irc09'][0][0][3]

#     x0 = (ircdata[:,5]) # 1*9
#     T = 2
#     for i in range(1):
#         trajectory = MD(x0, T, dt=1e-3, beta=1.0) #(steps, bs, dim)
#         print(trajectory.shape)
#     indices = np.arange(trajectory.shape[0])
#     ax1.scatter(trajectory[:,0, 0], trajectory[:,0, 1], c=indices, cmap=cmap)
# #     np.savez(foldername + '/Meta' + str(args.trial), trajectory = trajectory, centers = qs)

