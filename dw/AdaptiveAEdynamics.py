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
from functools import partial
import haiku as hk
import optax


# Global dimensionality of the system
n = 8


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


# Define the autoencoder
def autoencoder_fn(x):
    input_dim = x.shape[-1]
    intermediate_dim = 64
    encoding_dim = 3

    x = hk.Linear(intermediate_dim)(x)
    x = jax.nn.relu(x)
    x = hk.Linear(intermediate_dim)(x)
    x = jax.nn.relu(x)
    encoded = hk.Linear(encoding_dim)(x)

    x = hk.Linear(intermediate_dim)(encoded)
    x = jax.nn.relu(x)
    x = hk.Linear(intermediate_dim)(x)
    x = jax.nn.relu(x)
    decoded = jax.numpy.abs(hk.Linear(input_dim)(x))
    return decoded, encoded


# Define the autoencoder globally
autoencoder = hk.without_apply_rng(hk.transform(autoencoder_fn))

# # Initialize the model
# rng = jax.random.PRNGKey(42)
# sample = jnp.ones([1, 2 + n])  # Example input shape
# params = autoencoder.init(rng, sample)

# # Define optimizer
# learning_rate = 1e-3
# optimizer = optax.adam(learning_rate)
# opt_state = optimizer.init(params)

# # Register the optimizer state as a PyTree
# def opt_state_flatten(opt_state):
#     return (opt_state,), None

# def opt_state_unflatten(aux_data, children):
#     return children[0]

# jax.tree_util.register_pytree_node(
#     optax.OptState,
#     opt_state_flatten,
#     opt_state_unflatten
# )


def initialize_autoencoder(rng, sample):
    params = autoencoder.init(rng, sample)
    print("Params type:", type(params))
    # print("Params structure:", params)
    return params


def train_autoencoder(data, params, opt_state, optimizer, epochs=300, batch_size=32):
    for epoch in range(epochs):
        data = jax.random.permutation(jax.random.PRNGKey(epoch), data)
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            # Debug prints
            # print(f'Params type before train_step: {type(params)}')
            # print(f'Opt_state type before train_step: {type(opt_state)}')
            # print(f'Optimizer type before train_step: {type(optimizer)}')
            params, opt_state, loss = train_step(params, batch, opt_state, optimizer)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    return params, opt_state



# TODO: figure out why jax doesn't like the optimizer being passed in. Maybe use a global? PyTree approach didn't seem to work
# @jax.jit
def train_step(params, x, opt_state, optimizer):
    loss, grads = jax.value_and_grad(mse_loss)(params, x)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


def mse_loss(params, x):
    decoded, _ = autoencoder.apply(params, x)
    return jnp.mean((x - decoded) ** 2)


def AE(data, params, opt_state, optimizer):
    params, opt_state = train_autoencoder(data, params, opt_state, optimizer)
    return params, opt_state


def encode(params, x):
    _, encoded = autoencoder.apply(params, x)
    return encoded



# NOTE: eliminated envelope gaussians
# @partial(jax.jit, static_argnums=(2,))
def SumGaussian_single(x, center, i, encoder_params_list, h, sigma):
    x_minus_center = x - center  # D
    # pw_x_projected = jnp.matmul(pw_x_minus_center, eigenvectors)  # k
    # global_autoencoder = global_models[i]
    encoder_params = encoder_params_list[i]  # Select the appropriate encoder parameters
    # encoder = Model(global_autoencoder.input, global_autoencoder.layers[2].output)
    # x_projected = encoder.predict(x_minus_center)
    x_projected = encode(encoder_params, x_minus_center)
    x_projected_sq_sum = jnp.sum(x_projected**2)  # scalar

    exps = h * jnp.exp(-x_projected_sq_sum / (2 * sigma**2))  # scalar

    return exps  # scalar

@jax.jit
def JSumGaussian(x, centers, encoder_params_list, h, sigma):
    # x: 1 * M
    # centers: N * M
    i = jax.lax.iota(dtype=int, size=centers.shape[0])
    print(f'i.shape: {i.shape}')
    print(f'centers.shape: {centers.shape}')

    # Vectorize the single computation over the batch dimension N
    vmap_sum_gaussian = vmap(SumGaussian_single, in_axes=(None, 0, 0, None, None, None))

    total_bias = vmap_sum_gaussian(x, centers, i, encoder_params_list, h, sigma)  # N

    # TODO??: Normalize AND plot the size of normalization factor
    # Track the new sigma values that we calculate and use that for all calcs

    # TODO: variable sigma's dependent on the size of the eigenvalue. Larger eigenvalue = larger Gaussian
    # NEED that as it might potentially help the AE specifically

    return jnp.sum(total_bias)  # scalar

jax_SumGaussian = jax.grad(JSumGaussian)
jax_SumGaussian_jit = jax.jit(jax_SumGaussian)

def GradGaussian(x, centers, encoder_params_list, h, sigma):
    # print(f'jax_SumGaussianPW_jit._cache_size: {jax_VSumGaussianPW_jit._cache_size()}')
    print(f'x.shape: {x.shape}')
    # pw_x_jnp = jnp.array([Jget_pairwise_distances(x)])   # 1 * D
    # centers_jnp = centers                                 # N * D
    x_jnp = jnp.array(x)
    centers_jnp = jnp.array(centers)
    # encoder_params_list_jnp = jnp.array(encoder_params_list)
    # TODO: also convert encoder_params_list to a jnp thing??

    grad = jax_SumGaussian_jit(x_jnp, centers_jnp, encoder_params_list, h, sigma)

    return grad


def MD(q0, T, Tdeposite, height, sigma, dt=1e-3, beta=1.0, n=0):
    Nsteps = int(T / dt)
    NstepsDeposite = int(Tdeposite / dt)
    trajectories = np.zeros((Nsteps + 1, q0.shape[0], 2 + n))

    # variance = 0.7  # Threshhold for choosing number of eigenvectors
    q = q0
    qs = None


    # Sample data for initialization
    sample = jnp.ones((1, 2 + n))

    # Initialize the autoencoder
    rng = jax.random.PRNGKey(42)
    params = initialize_autoencoder(rng, sample)
    encoder_params_list = [params]
    
    # Initialize the optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    # # Register the optimizer state as a PyTree
    # def opt_state_flatten(opt_state):
    #     return (opt_state,), None

    # def opt_state_unflatten(aux_data, children):
    #     return children[0]

    # jax.tree_util.register_pytree_node(
    #     optax.OptState,
    #     opt_state_flatten,
    #     opt_state_unflatten
    # )


    for i in tqdm(range(Nsteps)):

        trajectories[i, :] = q
        q = next_step(q, qs, encoder_params_list, height, sigma, dt, beta)

        if (i + 1) % NstepsDeposite == 0:

            if qs is None:

                data = trajectories[:NstepsDeposite]  # (N_steps, 1, 2)
                data = np.squeeze(data, axis=1)  # (100, 2)
                mean_vector = np.mean(data, axis=0, keepdims=True)
                params, opt_state = AE(data, params, opt_state, optimizer)
                encoder_params_list[0] = params     # overwrite initializer values
                qs = mean_vector                #data[-2:-1]#mean_vector

            else:
                data = trajectories[i - NstepsDeposite + 1:i + 1]
                data = np.squeeze(data, axis=1)  # (100, 2)
                mean_vector = np.mean(data, axis=0, keepdims=True)
                params, opt_state = AE(data, params, opt_state, optimizer)
                encoder_params_list = jnp.concatenate([encoder_params_list, jnp.array([params])], axis=0)
                qs = np.concatenate([qs, mean_vector], axis=0)

    trajectories[Nsteps, :] = q
    return trajectories, qs


def next_step(qnow, qs, encoder_params_list, height, sigma, dt=1e-3, beta=1.0):
    if qs is None:
        qnext = qnow + (- gradV(qnow)) * dt + np.sqrt(2 * dt / beta) * np.random.randn(*qnow.shape)
    else:
        qnext = qnow + (- (gradV(qnow) + GradGaussian(qnow, qs, encoder_params_list, height, sigma))) * dt + np.sqrt(
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

    # # number of extra dimensions
    # n = 8

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

    # T = 400
    T = 10
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