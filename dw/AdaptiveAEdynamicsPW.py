import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import numpy.linalg as la
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from jax import vmap

# List to store models
global_models = []

# dimension the particles live in
d = 1

def potential(qx, qy, qn):
    V = 0.1*(qy +0.1*qx**3)**2 + 2*np.exp(-qx**2) + (qx**2+qy**2)/36 + np.sum(qn**2)/36
    return V


def gradV(q):
    qx = q[:, 0:1]
    qy = q[:, 1:2]
    qn = q[:, 2:]
    Vx = 0.1*2*(qy +0.1*qx**3)*3*0.1*qx**2 - 2*qx*2*np.exp(-qx**2) + 2*qx/36
    Vy = 0.1*2*(qy +0.1*qx**3) + 2*qy/36
    Vn = 2*qn/36
    return np.concatenate((Vx, Vy, Vn), axis=1)


def Jget_pairwise_distances(x):
    Natoms = int(x.shape[-1] / d)
    x = jnp.reshape(x, (Natoms, d))
    all_diffs = jnp.expand_dims(x, axis=1) - jnp.expand_dims(x, axis=0) # N * N * M
    sq_diffs = jnp.power(all_diffs, 2.)
    sum_sq_diffs = jnp.sum(sq_diffs, axis=-1)
    pairwise_distances = jnp.sqrt(sum_sq_diffs) # N * N
    pairwise_distances = pairwise_distances[jnp.triu_indices(Natoms, 1)]

    return pairwise_distances


def AE(data):
    global global_models

    mean_vector = np.mean(data, axis=0, keepdims=True)
    std_vector = np.std(data, axis=0, keepdims=True)
    
    input_dim = data.shape[1]
    print(f'input dim: {input_dim}')
    encoding_dim = 3 # Set the desired encoding dimension
    intermediate_dim = 64 # Set the width of the intermediate layer

    if len(global_models) > 0:
        # Load the saved model if it exists
        global_autoencoder = global_models[-1]
        encoder = Model(global_autoencoder.input, global_autoencoder.layers[2].output)

    else:
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

        global_autoencoder = Model(input_layer, decoded)
        global_autoencoder.compile(optimizer='adam', loss='mse')

    # Train the Autoencoder
    global_autoencoder.fit(data, data, epochs=300, batch_size=32, shuffle=True, validation_split=0.2)
    
    # Save the model after training
    global_models.append(global_autoencoder)

    return mean_vector

# what is this function doing?? Is this like SumGaussians?
@jax.jit
def GaussiansPW(q, qs, height, sigma):

    V = np.empty((q.shape[0], 1))
    for i in range(q.shape[0]):
        x_minus_centers = q[i:i + 1] - qs  # N * M
        # x_minus_centers = np.expand_dims(x_minus_centers, axis=1)  # N * 1 * M
        # x_projected = np.matmul(x_minus_centers, eigenvectors)
        # x_projected_ = x_projected * choose_eigenvalue_

        global_autoencoder = global_models[i]
        encoder = Model(global_autoencoder.input, global_autoencoder.layers[2].output)
        x_projected = encoder.predict(x_minus_centers)

        print(f'x_projected.shape: {x_projected.shape}')

        x_projected_sq_sum = np.sum((x_projected) ** 2, axis=(-2, -1))  # N

        V[i] = np.sum(height * np.exp(-np.expand_dims(x_projected_sq_sum, axis=1) / 2 / sigma ** 2), axis=0)

    return V

# NOTE: eliminated envelope gaussians
def SumGaussianPW_single(pw_x, pw_center, i, h, sigma):
    pw_x_minus_center = pw_x - pw_center  # D
    # pw_x_projected = jnp.matmul(pw_x_minus_center, eigenvectors)  # k
    global_autoencoder = global_models[i]
    encoder = Model(global_autoencoder.input, global_autoencoder.layers[2].output)
    pw_x_projected = encoder.predict(pw_x_minus_center)
    pw_x_projected_sq_sum = jnp.sum(pw_x_projected**2)  # scalar

    exps = h * jnp.exp(-pw_x_projected_sq_sum / (2 * sigma**2))  # scalar

    return exps  # scalar

@jax.jit
def JSumGaussianPW(pw_x, pw_centers, h, sigma):
    # x: 1 * M
    # centers: N * M
    i = np.arange(0, pw_centers.shape[0], 1, dtype=int)

    # Vectorize the single computation over the batch dimension N
    vmap_sum_gaussian_pw = vmap(SumGaussianPW_single, in_axes=(None, 0, 0, None, None))

    total_bias = vmap_sum_gaussian_pw(pw_x, pw_centers, i, h, sigma)  # N

    # TODO??: Normalize AND plot the size of normalization factor
    # Track the new sigma values that we calculate and use that for all calcs

    # TODO: variable sigma's dependent on the size of the eigenvalue. Larger eigenvalue = larger Gaussian
    # NEED that as it might potentially help the AE specifically

    return jnp.sum(total_bias)  # scalar

jax_SumGaussianPW = jax.grad(JSumGaussianPW)
jax_SumGaussianPW_jit = jax.jit(jax_SumGaussianPW)

def GradGaussian(x, centers_pw, h, sigma):
    # print(f'jax_SumGaussianPW_jit._cache_size: {jax_VSumGaussianPW_jit._cache_size()}')
    print(f'x.shape: {x.shape}')
    pw_x_jnp = jnp.array([Jget_pairwise_distances(x)])   # 1 * D
    pw_centers_jnp = centers_pw                                 # N * D

    pw_grad = jax_SumGaussianPW_jit(pw_x_jnp, pw_centers_jnp, h, sigma)

    # TODO: test that the below code is correct
    N = pw_grad.shape[0]
    M = x.shape[-1]
    Natoms = int(M / 3)
    D = int(Natoms * (Natoms - 1) / 2)

    x_grad = np.zeros((Natoms, 3))
    x_vals = np.reshape(x, (Natoms, 3))

    # TODO: sanity check the signs

    iu = np.triu_indices(Natoms, 1)
    for k in range(N):
        for l in range(D):
            # for j in range(i):
            i = iu[0][l]
            j = iu[1][l]
            rij = pw_x_jnp[0][l] if (pw_x_jnp[0][l] != 0) else 0.000001
            update_vec = pw_grad[k][l] * (x_vals[j] - x_vals[i]) / rij
            x_grad[i] -= update_vec
            x_grad[j] += update_vec

    x_grad = x_grad.flatten()   # M

    return x_grad


def MD(q0, T, Tdeposite, height, sigma, dt=1e-3, beta=1.0, n=0):
    Nsteps = int(T / dt)
    NstepsDeposite = int(Tdeposite / dt)
    trajectories = np.zeros((Nsteps + 1, q0.shape[0], 2 + n))

    print(trajectories.shape)

    variance = 0.7  # Threshhold for choosing number of eigenvectors
    q = q0

    qs = None

    for i in tqdm(range(Nsteps)):

        trajectories[i, :] = q
        q = next_step(q, qs, height, sigma, dt, beta)

        if (i + 1) % NstepsDeposite == 0:

            if qs is None:

                data = trajectories[:NstepsDeposite]  # (N_steps, 1, 2)
                data = np.squeeze(data, axis=1)  # (100, 2)
                mean_vector = AE(data)
                qs = mean_vector                #data[-2:-1]#mean_vector

            else:
                data = trajectories[i - NstepsDeposite + 1:i + 1]
                data = np.squeeze(data, axis=1)  # (100, 2)
                mean_vector = AE(data, ic_method)
                qs = np.concatenate([qs, mean_vector], axis=0)

    trajectories[Nsteps, :] = q
    return trajectories, qs


def next_step(qnow, qs, height, sigma, dt=1e-3, beta=1.0):
    if qs is None:
        qnext = qnow + (- gradV(qnow)) * dt + np.sqrt(2 * dt / beta) * np.random.randn(*qnow.shape)
    else:
        qnext = qnow + (- (gradV(qnow) + GradGaussian(qnow, qs, height, sigma))) * dt + np.sqrt(
            2 * dt / beta) * np.random.randn(*qnow.shape)
    # print(qnow.shape, qnext.shape, np.random.randn(*qnow.shape))
    return qnext


def findTSTime(trajectory):
    x_dimension = trajectory[:, 0, 0]
    # Find the indices where the first dimension is greater than 0
    indices = np.where(x_dimension > 0)[0]
    # Check if any such indices exist
    if indices.size > 0:
        # Get the first occurrence
        first_occurrence_index = indices[0]
        return f"The first time step where the first dimension is greater than 0 is: {first_occurrence_index}"
    else:
        return "There are no time steps where the first dimension is greater than 0."

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int, default=0)
    args = parser.parse_args()

    # number of extra dimensions
    n = 8

    xx = np.linspace(-10, 10, 200)
    yy = np.linspace(-25, 25, 200)
    [X, Y] = np.meshgrid(xx, yy)  # 100*100
    W = potential(X, Y, np.zeros(n))
    W1 = W.copy()
    W1[W > 5] = float('nan')

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    contourf_ = ax1.contourf(X, Y, W1, levels=29)
    plt.colorbar(contourf_)

    T = 400
    # T = 10
    dt = 1e-2
    beta = 4
    Tdeposite = 1
    height = 0.25
    sigma = 1.25
    ic_method = 'AE'

    foldername = 'Doublewell'

    cmap = plt.get_cmap('plasma')


    for i in range(1):
        q0 = np.concatenate((np.array([[-5.0, 12.0]]), np.array([np.random.rand(8)*40-20])), axis=1)
        trajectory, qs = MD(q0, T, Tdeposite=Tdeposite, height=height, sigma=sigma, dt=dt, beta=beta, n=n)  # (steps, bs, dim)
        # print(eigenvalues.shape)
        print(findTSTime(trajectory))
        indices = np.arange(trajectory.shape[0])
        ax1.scatter(trajectory[:, 0, 0], trajectory[:, 0, 1], c=indices, cmap=cmap)

        savename = 'results/T{}_Tdeposite{}_dt{}_height{}_sigma{}_beta{}_ic{}'.format(T, Tdeposite, dt, height, sigma, beta, ic_method)
        np.savez(savename, trajectory=trajectory, qs=qs, global_models=global_models)

    # # test derivative
    # eps = 0.0001
    # print('dev', gradGaussians(q0, qs, eigenvectors, choose_eigenvalue, height, sigma))
    # V0 = GaussiansPCA(q0, qs, eigenvectors, choose_eigenvalue, height, sigma)
    # for i in range(2):
    #     q = q0.copy()
    #     q[0,i] +=  eps
    #     print(q0, q)
    #     print(str(i) + ' compoenent dev: ', (GaussiansPCA(q, qs, eigenvectors, choose_eigenvalue, height, sigma) - V0)/eps)

    num_points = X.shape[0] * X.shape[1]
    Gs = JSumGaussianPW(np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1), np.zeros((num_points, n))], axis=1), qs, height=height, sigma=sigma)
    ax2 = fig.add_subplot(1, 3, 2)
    Sum = Gs.reshape(200, 200)+W1

    cnf2 = ax2.contourf(X, Y, Gs.reshape(200, 200), levels=29)
    plt.colorbar(cnf2)
    print(eigenvectors.shape)
    indices = np.arange(qs.shape[0])
    ax2.scatter(qs[:, 0], qs[:, 1], c=indices, cmap=cmap)
    ax2.quiver(qs[:, 0], qs[:, 1])
    ax2.axis('equal')
    indices = np.arange(trajectory.shape[0])
    # ax2.scatter(trajectory[:, 0, 0], trajectory[:, 0, 1], c=indices, cmap=cmap, alpha=0.1)

    # ax2.scatter(trajectory[:, 0], trajectory[:, 1], c=indices, cmap=cmap)
    ax3 = fig.add_subplot(1, 3, 3)
    cnf3 = ax3.contourf(X, Y, Sum, levels=29)

    # fig.colorbar(contourf_)
    plt.title('Local AE dynamics')
    plt.show()