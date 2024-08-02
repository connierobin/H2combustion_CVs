import numpy as np
import numpy.linalg as la
import scipy
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import pickle


class Rxn9Simulation:

    def __init__(self, T, Tdeposite, height, sigma, dt=1e-3, beta=1.0, course=100, ic_method='PCA', checkpoint_name='rxn9_test.checkpoint'):
        self.T = T
        self.Tdeposite = Tdeposite
        self.height = height
        self.sigma = sigma
        self.dt = dt
        self.beta = beta
        self.course = course
        self.ic_method = ic_method
        self.checkpoint_name = checkpoint_name
        # calculated initial values
        self.Nsteps = int(self.T/self.dt)
        self.NstepsDeposite = int(self.Tdeposite/self.dt)
        # zero initial values
        self.q0 = np.array([[0, -0.5]])
        self.q = self.q0
        self.qs = None
        self.qsz = None
        self.qstd = None
        self.save_eigenvector_s = None
        self.save_eigenvalue_s = None
        self.choose_eigenvalue = None
        self.iter = 0
        self.trajectories = np.zeros((self.Nsteps+1, self.q0.shape[0], 2))
        # constant setup
        self.datadict_PES = scipy.io.loadmat('./R09PES3D.mat')
        self.max_PES = np.max(self.datadict_PES['V'])
        self.box = np.zeros((3, 2))
        for idx, item in enumerate(['X', 'Y', 'Z']):
            self.box[idx, 0] = np.min(self.datadict_PES[item])
            self.box[idx, 1] = np.max(self.datadict_PES[item])
        self.threshold = np.sqrt(1e-1)
        # Create a 3D grid
        X, Y, Z = np.meshgrid(self.datadict_PES['X'], self.datadict_PES['Y'], self.datadict_PES['Z'], indexing='ij')
        self.XYZ_ = np.concatenate((np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1)), np.reshape(Z, (-1, 1))), axis=1)
        self.V = self.datadict_PES['V']
        self.V = self.V.reshape(-1)
        print(self.datadict_PES['V'].shape)
        # Create a RegularGridInterpolator
        self.interpolator = RegularGridInterpolator((self.datadict_PES['X'][:, 0], self.datadict_PES['Y'][:, 0], self.datadict_PES['Z'][:, 0]), self.datadict_PES['V'])

    def run(self, need_restart):
        # Last time the function crashed. Need to restore the state.
        if need_restart:
            with open(self.checkpoint_name, 'rb') as f:
                self = pickle.load(f)

        cmap = plt.get_cmap('plasma')
        ircdata = scipy.io.loadmat('./irc09.mat')['irc09'][0][0][3]
        # x0 = ircdata[:, -2:-1].T  # 1*9
        x0 = ircdata[:, 5:6].T
        print(ircdata.shape)
        coarse = 100
        for i in range(1):
            self.trajectories, self.qs, self.save_eigenvector_s, self.save_eigenvalue_s = self.MD(x0, T = self.T, Tdeposite=self.Tdeposite, height=self.height, sigma=self.sigma, dt=self.dt, beta=self.beta, coarse=coarse)

        print(self.trajectories.shape)

        z_trajectory = (self.cart2zmat(self.trajectories[:,0])).T

        indices = np.arange(z_trajectory.shape[0])
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(z_trajectory[:, 0], z_trajectory[:, 1], z_trajectory[:, 2], c=indices, cmap=cmap)
        plt.show()
        np.savez('T10_deposit5e-3_dt2e-5-PCAs-InternalCoordinates_height1_T10_debug-nonlocal', z = z_trajectory, x = self.trajectories, qs=self.qs, eignvectors=self.save_eigenvector_s, eigenvalues=self.save_eigenvalue_s)



##### I've worked on stuff above this line #######

    def plot(self):
        xx = np.linspace(-2, 2, 100)
        yy = np.linspace(-1, 2, 100)
        [X, Y] = np.meshgrid(xx, yy)  # 100*100
        W = self.potential(X, Y)

        fig = plt.figure(figsize=(10,6))
        ax1 = fig.add_subplot(1, 3, 1)
        contourf_ = ax1.contourf(X, Y, W, levels=29)
        plt.colorbar(contourf_)

        # T = 10
        # height = 0.1  # set smaller, to 0.05
        # sigma = 0.2

        cmap = plt.get_cmap('plasma')
        indices = np.arange(self.trajectories.shape[0])
        ax1.scatter(self.trajectories[:,0, 0], self.trajectories[:,0, 1], c=indices, cmap=cmap)

        Gs = self.GaussiansPCA(np.concatenate([X.reshape(-1,1), Y.reshape(-1,1)], axis=1), self.qs, self.eigenvectors, height=self.height, sigma=self.sigma)
        ax2 = fig.add_subplot(1, 3, 2)
        contourf_2 = ax2.contourf(X, Y, Gs.reshape(100,100)+W, levels=29)
        plt.colorbar(contourf_2)
        indices = np.arange(self.qs.shape[0])
        cmap = plt.get_cmap('plasma')
        # ax2.scatter(qs[::-1, 0], qs[::-1, 1], c=indices, cmap=cmap)
        ax2.quiver(self.qs[:, 0], self.qs[:, 1], self.eigenvectors[0,:], self.eigenvectors[1,:])

        ax3 = fig.add_subplot(1, 3, 3)
        contourf_3 = ax3.contourf(X, Y, Gs.reshape(100,100), levels=29)
        plt.colorbar(contourf_3)

        # fig.colorbar(contourf_)
        plt.title(f'Local AE dynamics')
        plt.show()

    def PCA(self, data): # datasize: N * dim
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

    def AE(self, data):
        mean_vector = np.mean(data, axis=0, keepdims=True)
        input_dim = data.shape[1]
        print(f'input dim: {input_dim}')
        encoding_dim = 1 # Set the desired encoding dimension
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

        return mean_vector, ae_comps.T


##### I've worked on stuff below this line #######

    def is_inside_box(self, x, box):
        if len(x) != len(box):
            print(x.shape, box.shape)
            raise ValueError("Vector dimensions must match the number of box dimensions")

        for i in range(len(x)):
            if not (box[i][0] < x[i] < box[i][1]):
                return False
        return True
    
    def harmonic_potential_outside_box(self, x, box):
        if len(x) != len(box):
            raise ValueError("Vector dimensions must match the number of box dimensions")
        harmonic_potential = 0
        coef = 10000
        harmonic_potential = np.sum(coef * (np.maximum(box[:, 0] - x, 0)) ** 2 + coef * (np.maximum(x - box[:, 1], 0)) ** 2)

        return harmonic_potential
    
    def potential(self, x):
        z = self.cart2zmat(x)

        if self.is_inside_box(z, self.box):
            return self.interpolator(z.reshape(1, -1))
        else:
            # print(x.reshape(1,-1).shape, XYZ_.shape)
            distances = np.linalg.norm(z.reshape(1, -1) - self.XYZ_, axis=1)
            minidx = np.argmin(distances)
            return self.V[minidx] + self.harmonic_potential_outside_box(z.reshape(-1), self.box)

    def torsion(self, xyzs, i, j, k):
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

    def cart2zmat(self, X):
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
                alist.append(self.torsion(X[:, j], 3, 1, 2))

            Z.append(rlist + alist + dlist)

        Z = np.array(Z)
        return Z.T

    def gradV(self, x): # x shape 1*9
        eps = 1e-3
        f = self.potential(x)
        grad = np.zeros_like(x)

        for i in range(x.shape[1]):
            x_plus_h = x.copy()
            x_plus_h[0,i] += eps
            f_plus_h = self.potential(x_plus_h)
            grad[0, i] = (f_plus_h - f) / eps

        return grad

    def Gaussian(self, x, qx, qstd, qsz, method, choose_eigenvalue, save_eigenvector_s, height, sigma):
        # x the point we what calculate the value
        # gaussian deposit
        # method the way that we want deposit gaussian
        V = None
        if method == 'second_bond':
            # print(x.shape, qx.shape)
            q_z = self.cart2zmat(x).T
            qs_z = qsz # cart2zmat(qx).T
            # print(q_z.shape, qs_z.shape)
            V = height * np.sum(np.exp(-(q_z[0,1] - qs_z[:,1])**2/2/sigma**2))
        elif method == 'first_three_eigen':
            # print('qx shape', qx.shape, save_eigenvector_s.shape)
            x_z = self.cart2zmat(x).T
            # print('eigen: ', x_z.shape, qsz.shape, save_eigenvector_s.shape) # 1*3, gn*3, gn*3*k
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

    def gradGaussians(self, x, qs, qstd, qsz, method, choose_eigenvalue, save_eigenvector_s, height, sigma):
        eps = 1e-2
        f = self.Gaussian(x, qs, qstd, qsz, method, choose_eigenvalue, save_eigenvector_s, height, sigma)
        grad = np.zeros_like(x)

        for i in range(x.shape[1]):

            x_plus_h = x.copy()
            x_plus_h[0, i] += eps
            f_plus_h = self.Gaussian(x_plus_h, qs, qstd, qsz, method, choose_eigenvalue, save_eigenvector_s, height, sigma)
            grad[0, i] = (f_plus_h - f) / eps
            # print(f_plus_h, f)
        return grad
    
    def MD(self, q0, T, Tdeposite, height, sigma, dt=1e-3, beta=1.0, coarse=1):
        Nsteps = round(T / dt)
        NstepsDeposite = round(Tdeposite / dt)
        NstepsSave = round(T / (dt * coarse))
        trajectories = np.zeros((NstepsSave + 1, 1, 9))
        trajectories_PCA = np.zeros((NstepsDeposite, 1, 3))

        # q = q0
        # qs = None
        # qsz = None
        # qstd = None
        # save_eigenvector_s = None
        # save_eigenvalue_s = None
        # choose_eigenvalue = None
        method = 'first_three_eigen'
        variance = 0.8
        for i in tqdm(range(Nsteps)):
            trajectories[i // coarse, :] = q
            self.q = self.next_step(self.q, self.qs, self.qstd, self.qsz, self.method, self.choose_eigenvalue, self.save_eigenvector_s, self.height, self.sigma, self.dt, self.beta) # 1*9
            self.trajectories_PCA[i % NstepsDeposite, :] = self.cart2zmat(q).T
            if i % coarse == 0:
                    trajectories[i // coarse, :] = q

            if (i + 1) % NstepsDeposite == 0:
                if qs is None:
                    ### conducting PCA ###
                    data = trajectories_PCA  # (N_steps, 1, 2)
                    # print(i, trajectories_PCA)
                    data = np.squeeze(data, axis=1)  # (100, 2)
                    if self.ic_method == 'PCA':
                        mean_vector, std_vector, selected_eigenvectors, eigenvalues = self.PCA(data)
                    else:
                        mean_vector, std_vector, selected_eigenvectors, eigenvalues = self.AE(data)
                    # print(selected_eigenvectors.shape, eigenvalues.shape)
                    qs = mean_vector#q
                    # print(mean_vector.shape)
                    qsz = mean_vector#cart2zmat(qs).T
                    qstd = std_vector
                    
                    ### reset the PCA dataset
                    trajectories_PCA = np.zeros((NstepsDeposite, 1, 3))
                    
                    save_eigenvector_s = np.expand_dims(selected_eigenvectors, axis=0)
                    save_eigenvalue_s = np.expand_dims(eigenvalues, axis=0)
                    
                    choose_eigenvalue_tmp = np.zeros((1,3))
                    cumsum = np.cumsum(save_eigenvalue_s, axis=1)
                    var_ratio = cumsum/np.sum(save_eigenvalue_s)
                    idx = np.argmax(var_ratio>variance)
                    # print(save_eigenvalue_s, cumsum/np.sum(save_eigenvalue_s), np.argmax(var_ratio>variance))
                    for s in range(idx+1):
                        choose_eigenvalue_tmp[0,s]=1
                    choose_eigenvalue = choose_eigenvalue_tmp
                    # print(choose_eigenvalue)
                    # for k in range(3):
                    #     cumsum = np.cumsum(save_eigenvalue_s, axis=1)
                        
                    # print(save_eigenvector_s.shape, save_eigenvalue_s.shape)
                else:
                    ### conducting PCA ###
                    data = trajectories_PCA  # (N_steps, 1, 2)
                    # print(i, trajectories_PCA)
                    data = np.squeeze(data, axis=1)  # (100, 2)
                    if self.ic_method == 'PCA':
                        mean_vector, std_vector, selected_eigenvectors, eigenvalues = self.PCA(data)
                    else:
                        mean_vector, std_vector, selected_eigenvectors, eigenvalues = self.AE(data)

                    q_deposit = q
                    q_deposit = mean_vector
                    # print(mean_vector.shape, 'mean vector')
                    qs = np.concatenate([qs, q_deposit], axis=0)
                    # qsz = np.concatenate([qsz, cart2zmat(q_deposit).T], axis=0)
                    qsz = np.concatenate([qsz, q_deposit], axis=0)
                    qstd = np.concatenate([qstd, std_vector], axis=0)
                    
                    save_eigenvector_s = np.concatenate([save_eigenvector_s, np.expand_dims(selected_eigenvectors, axis=0)], axis=0)
                    save_eigenvalue_s = np.concatenate([save_eigenvalue_s, np.expand_dims(eigenvalues, axis=0)], axis=0)
                    
                    eigenvalues = np.expand_dims(eigenvalues, axis=0)
                    choose_eigenvalue_tmp = np.zeros((1,3))
                    cumsum = np.cumsum(eigenvalues, axis=1)
                    var_ratio = cumsum/np.sum(eigenvalues)
                    idx = np.argmax(var_ratio>variance)
                    # print(eigenvalues, var_ratio, np.argmax(var_ratio>variance))
                    for s in range(idx+1):
                        choose_eigenvalue_tmp[0,s]=1
                    choose_eigenvalue = np.concatenate([choose_eigenvalue, choose_eigenvalue_tmp], axis=0)
                    # print(choose_eigenvalue)
                    
                    # print(save_eigenvector_s.shape, save_eigenvalue_s.shape)
                    # print(i, qs.shape, qsz.shape)

                    # print(i, trajectories_PCA.shape, trajectories_PCA)

                    trajectories_PCA = np.zeros((NstepsDeposite, 1, 3))
        trajectories[NstepsSave, :] = q
        return trajectories, qs, save_eigenvector_s, save_eigenvalue_s

    def next_step(self, qnow, qs, qstd, qsz, method, choose_eigenvalue, save_eigenvector_s, height, sigma, dt=1e-3, beta=1.0):

        if qs is None:
            qnext = qnow + (- self.gradV(qnow)) * dt + np.sqrt(2 * dt / beta) * np.random.randn(*qnow.shape)
        else:
            qnext = qnow + (- (self.gradV(qnow) + self.gradGaussians(qnow, qs, qstd, qsz, method, choose_eigenvalue, save_eigenvector_s, height, sigma))) * dt + np.sqrt(2 * dt / beta) * np.random.randn(*qnow.shape)

        return qnext



if __name__ == "__main__":
    # self.MD(x0, T = self.T, Tdeposite=1e-2, height=1, sigma=0.3, dt=2e-5, beta=1, coarse=coarse)
    t = Rxn9Simulation(T=10, Tdeposite=1e-2, height=1, sigma=0.3, dt=2e-5, beta=1.0, coarse=100, method='PCA', checkpoint_name='rxn9_test.checkpoint')
    t.run(need_restart=True)
    t.plot()

    #####################
    # TESTING FUNCTIONS #
    #####################

    # test torsion function
    # choose atoms 1, 2, and 3
    # atoms = np.array([1,2,3])
    # # atom positions are (0, 0, 0), (0, 0, 1), (1, 0, 0)
    # # are the positions supposed to have gone through cart2zmat??
    # atoms_xyzs = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0])
    # print(t.torsion(atoms_xyzs, atoms[0], atoms[1], atoms[2]))