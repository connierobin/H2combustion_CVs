import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax import vmap
import haiku as hk
import optax
import time

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
    input_dim = x.shape[-1]  # Capture the input dimension
    intermediate_dim = 64
    encoding_dim = 3

    # Encoder
    x = hk.Linear(intermediate_dim)(x)
    x = jax.nn.relu(x)
    x = hk.Linear(intermediate_dim)(x)
    x = jax.nn.relu(x)
    encoded = hk.Linear(encoding_dim)(x)

    # Decoder
    x = hk.Linear(intermediate_dim)(encoded)
    x = jax.nn.relu(x)
    x = hk.Linear(intermediate_dim)(x)
    x = jax.nn.relu(x)
    decoded = hk.Linear(input_dim)(x)  # Ensure the output shape matches the input shape
    return decoded, encoded

# Define the autoencoder globally
autoencoder = hk.transform(autoencoder_fn)

def initialize_autoencoder(rng, sample):
    params = autoencoder.init(rng, sample, is_training=True)
    print("Params type:", type(params))
    return params

def train_step(params, x, opt_state, optimizer, is_training):
    (loss, grads) = jax.value_and_grad(mse_loss, has_aux=True)(params, x, is_training)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

def train_autoencoder(data, params, opt_state, optimizer, epochs=300, batch_size=32):
    for epoch in range(epochs):
        data = jax.random.permutation(jax.random.PRNGKey(epoch), data)
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            params, opt_state, loss = train_step(params, batch, opt_state, optimizer, is_training=True)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    return params, opt_state

def mse_loss(params, x, is_training):
    decoded, encoded = autoencoder.apply(params, None, x, is_training=is_training)
    loss = jnp.mean((x - decoded) ** 2)
    return loss, None

def AE(data, params, opt_state, optimizer):
    params, opt_state = train_autoencoder(data, params, opt_state, optimizer)
    return params, opt_state

def encode(params, x):
    _, encoded = autoencoder.apply(params, None, x, is_training=False)
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
    def batch_params(params):
        return params

    encoder_params_batched = batch_params(encoder_params_list[0])
    
    vmap_sum_gaussian = vmap(SumGaussian_single, in_axes=(0, None, None, None, None))
    total_bias = vmap_sum_gaussian(x, centers, encoder_params_batched, h, sigma)
    return total_bias

@jax.jit
def GradGaussian(x, centers, encoder_params_list, h, sigma):
    x_jnp = jnp.array(x)
    centers_jnp = jnp.array(centers)
    encoder_params_list_jnp = jax.tree_map(lambda x: jnp.array(x), encoder_params_list)
    grad = jax.grad(lambda x: jnp.sum(JSumGaussian(x, centers_jnp, encoder_params_list_jnp, h, sigma)))(x_jnp)
    return grad

@jax.jit
def next_step(qnow, qs, encoder_params_list, height, sigma, dt=1e-3, beta=1.0):
    if qs is None:
        qnext = qnow + (- gradV(qnow)) * dt + jnp.sqrt(2 * dt / beta) * jax.random.normal(jax.random.PRNGKey(0), shape=qnow.shape)
    else:
        qnext = qnow + (- (gradV(qnow) + GradGaussian(qnow, qs, encoder_params_list, height, sigma))) * dt + jnp.sqrt(2 * dt / beta) * jax.random.normal(jax.random.PRNGKey(0), shape=qnow.shape)
    return qnext

def MD(q0, T, Tdeposite, height, sigma, dt=1e-3, beta=1.0, n=0):
    Nsteps = int(T / dt)
    NstepsDeposite = int(Tdeposite / dt)
    trajectories = np.zeros((Nsteps + 1, q0.shape[0], 2 + n))

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

    # Register the optimizer state as a PyTree
    def opt_state_flatten(opt_state):
        return (opt_state,), None

    def opt_state_unflatten(aux_data, children):
        return children[0]

    jax.tree_util.register_pytree_node(
        optax.OptState,
        opt_state_flatten,
        opt_state_unflatten
    )

    for i in tqdm(range(Nsteps)):
        trajectories[i, :] = q
        q = next_step(q, qs, encoder_params_list, height, sigma, dt, beta)

        if (i + 1) % NstepsDeposite == 0:
            if qs is None:
                data = trajectories[:NstepsDeposite]
                data = np.squeeze(data, axis=1)
                params, opt_state = AE(data, params, opt_state, optimizer)
                encoder_params_list[0] = params
                qs = np.mean(data, axis=0, keepdims=True)
            else:
                data = trajectories[i - NstepsDeposite + 1:i + 1]
                data = np.squeeze(data, axis=1)
                params, opt_state = AE(data, params, opt_state, optimizer)
                encoder_params_list.append(params)
                qs = np.concatenate([qs, np.mean(data, axis=0, keepdims=True)], axis=0)

    trajectories[Nsteps, :] = q
    return trajectories, qs, encoder_params_list

def findTSTime(trajectory):
    x_dimension = trajectory[:, 0, 0]
    indices = np.where(x_dimension > 0)[0]
    if indices.size > 0:
        first_occurrence_index = indices[0]
        return f"The first time step where the first dimension is greater than 0 is: {first_occurrence_index}"
    else:
        return "There are no time steps where the first dimension is greater than 0."


def test1():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int, default=0)
    args = parser.parse_args()

    xx = jnp.linspace(-10, 10, 200)
    yy = jnp.linspace(-25, 25, 200)
    X, Y = jnp.meshgrid(xx, yy)
    W = potential(X, Y, jnp.zeros(n))
    W1 = W.copy()
    W1 = W1.at[W > 5].set(float('nan'))  # Use JAX .at[] method

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    contourf_ = ax1.contourf(np.array(X), np.array(Y), np.array(W1), levels=29)
    plt.colorbar(contourf_)

    T = 400
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
        trajectory, qs, encoder_params_list = MD(q0, T, Tdeposite=Tdeposite, height=height, sigma=sigma, dt=dt, beta=beta, n=n)
        print(findTSTime(trajectory))
        indices = np.arange(trajectory.shape[0])
        ax1.scatter(trajectory[:, 0, 0], trajectory[:, 0, 1], c=indices, cmap=cmap)

        savename = 'results/T{}_Tdeposite{}_dt{}_height{}_sigma{}_beta{}_ic{}'.format(T, Tdeposite, dt, height, sigma, beta, ic_method)
        np.savez(savename, trajectory=trajectory, qs=qs)

    num_points = X.shape[0] * X.shape[1]
    Gs = JSumGaussian(jnp.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1), jnp.zeros((num_points, n))], axis=1), qs, encoder_params_list, h=height, sigma=sigma)
    
    # Print the shape of Gs before reshaping
    print("Shape of Gs:", Gs.shape)

    ax2 = fig.add_subplot(1, 3, 2)
    Sum = Gs.reshape(200, 200) + np.array(W1)

    cnf2 = ax2.contourf(np.array(X), np.array(Y), np.array(Gs).reshape(200, 200), levels=29)
    plt.colorbar(cnf2)
    indices = np.arange(qs.shape[0])
    ax2.scatter(np.array(qs[:, 0]), np.array(qs[:, 1]), c=indices, cmap=cmap)
    ax2.quiver(np.array(qs[:, 0]), np.array(qs[:, 1]))
    ax2.axis('equal')
    indices = np.arange(trajectory.shape[0])
    ax3 = fig.add_subplot(1, 3, 3)
    cnf3 = ax3.contourf(np.array(X), np.array(Y), np.array(Sum), levels=29)

    plt.title('Local AE dynamics')
    plt.show()


def test2():
    # Initialize parameters for the test
    rng = jax.random.PRNGKey(42)
    sample = jnp.ones((1, 2 + n))
    params = initialize_autoencoder(rng, sample)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    # Create dummy data for the test
    data = jax.random.normal(rng, (100, 2 + n))

    # Measure time for training step
    start_time = time.time()
    params, opt_state = train_autoencoder(data, params, opt_state, optimizer, epochs=10, batch_size=10)
    end_time = time.time()
    print(f"Time taken for training (10 epochs): {end_time - start_time:.4f} seconds")

    # Measure time for a single next_step call
    q0 = jax.random.normal(rng, (1, 2 + n))
    qs = jax.random.normal(rng, (10, 2 + n))
    encoder_params_list = [params]

    start_time = time.time()
    for _ in range(1000):
        q0 = next_step(q0, qs, encoder_params_list, height=0.25, sigma=1.25, dt=1e-2, beta=4)
    end_time = time.time()
    print(f"Time taken for 1000 next_step calls: {end_time - start_time:.4f} seconds")




if __name__ == '__main__':
    test1()
    # test2()

    