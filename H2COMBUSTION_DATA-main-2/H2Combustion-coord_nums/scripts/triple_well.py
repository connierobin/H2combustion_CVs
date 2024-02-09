import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# np.random.seed(0)
threshold = 0.5

def potential(qx, qy):
    # qx = q[:,:1]
    # qy = q[:,1:]
    V = 3*np.exp(-qx**2-(qy-1/3)**2)-3*np.exp(-qx**2-(qy-5/3)**2)-5*np.exp(-(qx-1)**2-qy**2)-5*np.exp(-(qx+1)**2-qy**2)+0.2*qx**4+0.2*(qy-0.2)**4
    return V

def gradV(q):
    
    qx = q[:, 0:1]
    qy = q[:, 1:2]
    
    Vx = (-2*qx)*3*np.exp(-qx**2-(qy-1/3)**2)\
         -(-2*qx)*3*np.exp(-qx**2-(qy-5/3)**2)\
         -(-2*(qx-1))*5*np.exp(-(qx-1)**2-qy**2)\
         -(-2*(qx+1))*5*np.exp(-(qx+1)**2-qy**2)\
         +4*0.2*qx**3

    Vy = (-2*(qy-1/3))*3*np.exp(-qx**2-(qy-1/3)**2)\
         -(-2*(qy-5/3))*3*np.exp(-qx**2-(qy-5/3)**2)\
         -(-2*qy)*5*np.exp(-(qx-1)**2-qy**2)\
         -(-2*qy)*5*np.exp(-(qx+1)**2-qy**2)\
         +4*0.2*(qy-0.2)**3

    return np.concatenate((Vx, Vy), axis=1)

def PCA(data): # datasize: N * dim
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

def AE(data):
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

    return ae_comps.T

def GaussiansPCA(q, qs, eigenvectors, height, sigma):
    V = np.empty((q.shape[0],1))
    for i in range(q.shape[0]):
        # dist = np.sqrt(np.sum((q[i:i+1] - qs) ** 2, axis=1, keepdims=True))
        V[i] = height * np.sum(np.exp(-np.sum((q[i:i+1] - qs)*eigenvectors.T, axis=1, keepdims=True)**2/2/sigma**2), axis=0, keepdims=True)
    return V

def gradGaussians(q, qs, eigenvectors, height, sigma):
    # print(qs.shape, eigenvectors.shape)
    # dist = np.sqrt(np.sum((q - qs)**2, axis=1, keepdims=True))

    Gs_x = height * np.sum(eigenvectors.T[:,0:1]*(-np.sum((q - qs)*eigenvectors.T, axis=1, keepdims=True))/sigma**2* (np.exp(-np.sum((q - qs)*eigenvectors.T, axis=1, keepdims=True)**2/2/sigma**2)), axis=0, keepdims=True)
    Gs_y = height * np.sum(eigenvectors.T[:,1:2]*(-np.sum((q - qs)*eigenvectors.T, axis=1, keepdims=True))/sigma**2* (np.exp(-np.sum((q - qs)*eigenvectors.T, axis=1, keepdims=True)**2/2/sigma**2)), axis=0, keepdims=True)

    return np.concatenate((Gs_x, Gs_y), axis=1)

def MD_PCA(q0, T, Tdeposite, height, sigma, dt=1e-3, beta=1.0):
    Nsteps = int(T/dt)
    NstepsDeposite = int(Tdeposite/dt)
    trajectories = np.zeros((Nsteps+1, q0.shape[0], 2))
    print(trajectories.shape)
    q = q0
    qs = None
    eigenvectors = None
    for i in range(Nsteps):
        trajectories[i, :] = q
        q = next_step(q, qs, eigenvectors, height, sigma, dt, beta)
        if (i+1)%NstepsDeposite==0:
            if qs is None:

                data = trajectories[:NstepsDeposite] # (N_steps, 1, 2)
                data = np.squeeze(data, axis=1) # (100, 2)
                mean_vector, selected_eigenvectors = PCA(data)
                qs = mean_vector
                eigenvectors = selected_eigenvectors
                # print(np.mean(trajectories[:NstepsDeposite], axis=0))
            else:
                data = trajectories[i-NstepsDeposite+1:i+1]
                data = np.squeeze(data, axis=1)  # (100, 2)
                mean_vector, selected_eigenvectors = PCA(data)
                qs = np.concatenate([mean_vector, qs], axis=0)
                eigenvectors = np.concatenate([selected_eigenvectors, eigenvectors], axis=1)
    trajectories[Nsteps, :] = q
    return trajectories, qs, eigenvectors

def MD_AE(q0, T, Tdeposite, height, sigma, dt=1e-3, beta=1.0):
    Nsteps = int(T/dt)
    NstepsDeposite = int(Tdeposite/dt)
    trajectories = np.zeros((Nsteps+1, q0.shape[0], 2))
    print(trajectories.shape)
    q = q0
    qs = None
    eigenvectors = None
    for i in range(Nsteps):
        print(f'STEP {i+1}')
        trajectories[i, :] = q
        q = next_step(q, qs, eigenvectors, height, sigma, dt, beta)
        if (i+1)%NstepsDeposite==0:
            if qs is None:

                data = trajectories[:NstepsDeposite] # (N_steps, 1, 2)
                data = np.squeeze(data, axis=1) # (100, 2)
                mean_vector = np.array([np.mean(data, axis=0)])
                selected_eigenvectors = AE(data)
                qs = mean_vector
                eigenvectors = selected_eigenvectors
                # print(np.mean(trajectories[:NstepsDeposite], axis=0))
            else:
                data = trajectories[i-NstepsDeposite+1:i+1]
                data = np.squeeze(data, axis=1)  # (100, 2)
                mean_vector = np.array([np.mean(data, axis=0)])
                selected_eigenvectors = AE(data)
                qs = np.concatenate([mean_vector, qs], axis=0)
                eigenvectors = np.concatenate([selected_eigenvectors, eigenvectors], axis=1)
    trajectories[Nsteps, :] = q
    return trajectories, qs, eigenvectors

def next_step(qnow, qs, eigenvectors, height, sigma, dt=1e-3, beta=1.0):
    if qs is None:
        qnext = qnow + (- gradV(qnow)) * dt + np.sqrt(2 * dt / beta) * np.random.randn(*qnow.shape)
    else:
        qnext = qnow + (- (gradV(qnow)+gradGaussians(qnow, qs, eigenvectors, height, sigma))) * dt + np.sqrt(2 * dt / beta) * np.random.randn(*qnow.shape)
    # print(qnow.shape, qnext.shape, np.random.randn(*qnow.shape))
    return qnext

def run_simulation(method='PCA', T=10, height=0.1, sigma=0.2):
    xx = np.linspace(-2, 2, 100)
    yy = np.linspace(-1, 2, 100)
    [X, Y] = np.meshgrid(xx, yy)  # 100*100
    W = potential(X, Y)

    fig = plt.figure(figsize=(10,6))
    ax1 = fig.add_subplot(1, 3, 1)
    contourf_ = ax1.contourf(X, Y, W, levels=29)
    plt.colorbar(contourf_)

    # T = 10
    # height = 0.1  # set smaller, to 0.05
    # sigma = 0.2

    cmap = plt.get_cmap('plasma')
    for i in range(1):
        q0 = np.array([[0, -0.5]])
        if method == 'AE':
            trajectory, qs, eigenvectors = MD_AE(q0, T,  Tdeposite=0.5, height=height, sigma=sigma, dt=5e-3, beta=1.66) #(steps, bs, dim)
        else:
            trajectory, qs, eigenvectors = MD_PCA(q0, T,  Tdeposite=0.5, height=height, sigma=sigma, dt=5e-3, beta=1.66) #(steps, bs, dim)
        # print(trajectory.shape)
        indices = np.arange(trajectory.shape[0])
        ax1.scatter(trajectory[:,0, 0], trajectory[:,0, 1], c=indices, cmap=cmap)

    Gs = GaussiansPCA(np.concatenate([X.reshape(-1,1), Y.reshape(-1,1)], axis=1), qs, eigenvectors, height=height, sigma=sigma)
    ax2 = fig.add_subplot(1, 3, 2)
    contourf_2 = ax2.contourf(X, Y, Gs.reshape(100,100)+W, levels=29)
    plt.colorbar(contourf_2)
    indices = np.arange(qs.shape[0])
    cmap = plt.get_cmap('plasma')
    # ax2.scatter(qs[::-1, 0], qs[::-1, 1], c=indices, cmap=cmap)
    ax2.quiver(qs[:, 0], qs[:, 1], eigenvectors[0,:], eigenvectors[1,:])

    ax3 = fig.add_subplot(1, 3, 3)
    contourf_3 = ax3.contourf(X, Y, Gs.reshape(100,100), levels=29)
    plt.colorbar(contourf_3)

    # fig.colorbar(contourf_)
    plt.title(f'Local {method} dynamics')
    plt.show()

if __name__ == "__main__":
    run_simulation('PCA', T=10)
    run_simulation('AE', T=10)