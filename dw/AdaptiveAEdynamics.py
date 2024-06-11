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
    V = 0.1*(qy +0.1*qx**3)**2 + 2*jnp.exp(-qx**2) + (qx**2+qy**2)/36 + jnp.sum(qn**2)/36
    return V


def gradV(q):
    qx = q[:, 0:1]
    qy = q[:, 1:2]
    qn = q[:, 2:]
    Vx = 0.1*2*(qy +0.1*qx**3)*3*0.1*qx**2 - 2*qx*2*jnp.exp(-qx**2) + 2*qx/36
    Vy = 0.1*2*(qy +0.1*qx**3) + 2*qy/36
    Vn = 2*qn/36
    return jnp.concatenate((Vx, Vy, Vn), axis=1)


# Define the autoencoder
def autoencoder_fn(x, is_training=False):
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


def initialize_autoencoder(rng, sample):
    params = autoencoder.init(rng, sample, is_training=True)
    return params


@partial(jax.jit, static_argnums=(3, 4))
def train_step(params, x, opt_state, update_fn, is_training):
    (loss, grads) = jax.value_and_grad(mse_loss, has_aux=True)(params, x, is_training)
    updates, opt_state = update_fn(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


@partial(jax.jit, static_argnums=(3, 4))
def train_epoch(params, data, opt_state, update_fn, batch_size):
    def body_fun(i, carry):
        params, opt_state, loss = carry
        start = i * batch_size
        batch = jax.lax.dynamic_slice(data, (start, 0), (batch_size, data.shape[1]))
        params, opt_state, batch_loss = train_step(params, batch, opt_state, update_fn, is_training=True)
        loss += batch_loss[0]  # Extract the actual loss value
        return params, opt_state, loss

    num_batches = len(data) // batch_size
    params, opt_state, loss = jax.lax.fori_loop(0, num_batches, body_fun, (params, opt_state, 0.0))
    return params, opt_state, loss


def train_autoencoder(data, params, opt_state, optimizer, epochs=300, batch_size=32):
    update_fn = optimizer.update
    for epoch in range(epochs):
        data = jax.random.permutation(jax.random.PRNGKey(epoch), data)
        params, opt_state, loss = train_epoch(params, data, opt_state, update_fn, batch_size)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    return params, opt_state


# Returns a None placeholder that could contain auxiliary information
def mse_loss(params, x, is_training):
    decoded, encoded = autoencoder.apply(params, x, is_training=is_training)
    loss = jnp.mean((x - decoded) ** 2)
    return loss, None


def AE(data, params, opt_state, optimizer):
    params, opt_state = train_autoencoder(data, params, opt_state, optimizer)
    return params, opt_state


def encode(params, x):
    _, encoded = autoencoder.apply(params, x, is_training=False)
    return encoded


@jax.jit
def SumGaussian_single(x, center, encoder_params, h, sigma):
    x_minus_center = x - center
    x_projected = encode(encoder_params, x_minus_center)
    x_projected_sq_sum = jnp.sum(x_projected**2)
    exps = h * jnp.exp(-x_projected_sq_sum / (2 * sigma**2))
    return exps


@jax.jit
def JSumGaussian(x, centers, encoder_params_list, h, sigma):
    # x: 1 * M
    # centers: N * M

    # Ensure that all parameters in encoder_params_list are correctly batched
    def batch_params(params):
        return params

    encoder_params_batched = batch_params(encoder_params_list[0])
    
    # Vectorize the single computation over the batch dimension N
    vmap_sum_gaussian = vmap(SumGaussian_single, in_axes=(0, None, None, None, None))
    total_bias = vmap_sum_gaussian(x, centers, encoder_params_batched, h, sigma)  # N


    # TODO??: Normalize AND plot the size of normalization factor
    # Track the new sigma values that we calculate and use that for all calcs

    # TODO: variable sigma's dependent on the size of the eigenvalue. Larger eigenvalue = larger Gaussian
    # NEED that as it might potentially help the AE specifically

    return jnp.sum(total_bias)  # scalar


@jax.jit
def JSumGaussian_unsummed(x, centers, encoder_params_list, h, sigma):
    # x: 1 * M
    # centers: N * M

    # Ensure that all parameters in encoder_params_list are correctly batched
    def batch_params(params):
        return params

    encoder_params_batched = batch_params(encoder_params_list[0])
    
    # Vectorize the single computation over the batch dimension N
    vmap_sum_gaussian = vmap(SumGaussian_single, in_axes=(0, None, None, None, None))
    total_bias = vmap_sum_gaussian(x, centers, encoder_params_batched, h, sigma)  # N


    # TODO??: Normalize AND plot the size of normalization factor
    # Track the new sigma values that we calculate and use that for all calcs

    # TODO: variable sigma's dependent on the size of the eigenvalue. Larger eigenvalue = larger Gaussian
    # NEED that as it might potentially help the AE specifically

    return total_bias  # scalar


jax_SumGaussian = jax.grad(JSumGaussian)
jax_SumGaussian_jit = jax.jit(jax_SumGaussian)

@jax.jit
def GradGaussian(x, centers, encoder_params_list, h, sigma):
    # print(f'jax_SumGaussianPW_jit._cache_size: {jax_VSumGaussianPW_jit._cache_size()}')
    x_jnp = jnp.array(x)
    centers_jnp = jnp.array(centers)        # N * D
    encoder_params_list_jnp = jax.tree_map(lambda x: jnp.array(x), encoder_params_list)
    grad = jax_SumGaussian_jit(x_jnp, centers_jnp, encoder_params_list_jnp, h, sigma)
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
                encoder_params_list.append(params)
                qs = np.concatenate([qs, mean_vector], axis=0)

    trajectories[Nsteps, :] = q
    return trajectories, qs, encoder_params_list


@jax.jit
def next_step(qnow, qs, encoder_params_list, height, sigma, dt=1e-3, beta=1.0):
    if qs is None:
        qnext = qnow + (- gradV(qnow)) * dt + jnp.sqrt(2 * dt / beta) * jax.random.normal(jax.random.PRNGKey(0), shape=qnow.shape)
    else:
        qnext = qnow + (- (gradV(qnow) + GradGaussian(qnow, qs, encoder_params_list, height, sigma))) * dt + jnp.sqrt(
            2 * dt / beta) * jax.random.normal(jax.random.PRNGKey(0), shape=qnow.shape)
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
    W1 = W1.at[W > 5].set(float('nan'))  # Use JAX .at[] method

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
        trajectory, qs, encoder_params_list = MD(q0, T, Tdeposite=Tdeposite, height=height, sigma=sigma, dt=dt, beta=beta, n=n)  # (steps, bs, dim)
        # print(eigenvalues.shape)
        print(findTSTime(trajectory))
        indices = np.arange(trajectory.shape[0])
        ax1.scatter(trajectory[:, 0, 0], trajectory[:, 0, 1], c=indices, cmap=cmap)

        savename = 'results/T{}_Tdeposite{}_dt{}_height{}_sigma{}_beta{}_ic{}'.format(T, Tdeposite, dt, height, sigma, beta, ic_method)
        np.savez(savename, trajectory=trajectory, qs=qs, encoder_params_list=encoder_params_list)

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
    Gs = JSumGaussian_unsummed(jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1), jnp.zeros((num_points, n))], axis=1), qs, encoder_params_list, h=height, sigma=sigma)
    ax2 = fig.add_subplot(1, 3, 2)
    # Print the shape of Gs before reshaping
    print("Shape of Gs:", Gs.shape)
    Sum = Gs.reshape(200, 200) + np.array(W1)

    cnf2 = ax2.contourf(X, Y, Gs.reshape(200, 200), levels=29)
    plt.colorbar(cnf2)
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