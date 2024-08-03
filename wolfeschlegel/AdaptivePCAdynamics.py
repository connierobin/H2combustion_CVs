import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import numpy.linalg as la

def potential(qx, qy, qn):
    V = 10 * (qx**4 + qy**4 - 2 * qx**2 - 4 * qy**2 + qx * qy + 0.2 * qx + 0.1 * qy + np.sum(qn**2))
    return V

def gradV(q):
    qx = q[:, 0:1]
    qy = q[:, 1:2]
    qn = q[:, 2:]
    Vx = 10 * (4 * qx**3 - 4 * qx + qy + 0.2)
    Vy = 10 * (4 * qy**3 - 8 * qy + qx + 0.1)
    Vn = 10 * 2*qn
    grad = np.concatenate((Vx, Vy, Vn), axis=1)
    return grad

def getCVs(data, ic_method='PCA'):
    return PCA(data)

def PCA(data):  # datasize: N * dim
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
    # k = 1  # Set the desired number of components

    # Step 4.7: Retain the top k components
    selected_eigenvectors = eigenvectors[:, :]

    # Step 4.8: Transform your data to the new lower-dimensional space
    transformed_data = np.dot(centered_data, selected_eigenvectors)

    # ##### using the last configuration
    # mean_vector = data[-1:]

    # print(np.dot(centered_data, selected_eigenvectors)-np.matmul(centered_data, selected_eigenvectors))
    # print(eigenvalues, eigenvalues.shape)
    return mean_vector, selected_eigenvectors, eigenvalues

def GaussiansPCA(q, qs, eigenvectors, choose_eigenvalue, height, sigma):
    choose_eigenvalue_ = np.expand_dims(choose_eigenvalue, axis=1)

    V = np.empty((q.shape[0], 1))
    for i in range(q.shape[0]):
        x_minus_centers = q[i:i + 1] - qs  # N * M
        x_minus_centers = np.expand_dims(x_minus_centers, axis=1)  # N * 1 * M
        x_projected = np.matmul(x_minus_centers, eigenvectors)
        x_projected_ = x_projected * choose_eigenvalue_


        x_projected_sq_sum = np.sum((x_projected_) ** 2, axis=(-2, -1))  # N

        V[i] = np.sum(height * np.exp(-np.expand_dims(x_projected_sq_sum, axis=1) / 2 / sigma ** 2), axis=0)

    return V


def gradGaussians(q, qs, eigenvectors, choose_eigenvalue, height, sigma):
    # print(q.shape, qs.shape, choose_eigenvalue.shape)
    # x: 1 * M
    # centers: N * M
    # eigenvectors: N * M * k
    # choose_eigenvalue: N*M

    choose_eigenvalue_ = np.expand_dims(choose_eigenvalue, axis=1) # N*1*M

    x_minus_centers = q - qs  # N * M
    x_minus_centers = np.expand_dims(x_minus_centers, axis=1)  # N * 1 * M
    x_projected = np.matmul(x_minus_centers, eigenvectors)  # N * 1 * k

    x_projected_ = x_projected * choose_eigenvalue_
    eigenvectors_ = eigenvectors * choose_eigenvalue_

    x_projected_sq_sum = np.sum((x_projected_)** 2, axis=(-2, -1))  # N
    exps = -height / sigma ** 2 * np.exp(-np.expand_dims(x_projected_sq_sum, axis=1) / 2 / sigma ** 2)  # N * 1
    PTPx = np.matmul(eigenvectors_, np.transpose(x_projected_, axes=(0, 2, 1)))  # N * M * 1
    PTPx = np.squeeze(PTPx, axis=2)  # N * M
    grad = np.sum(exps * PTPx, axis=0, keepdims=True)  # 1 * M

    # Gs_x = height * np.sum(
    #     eigenvectors.T[:, 0:1] * (-np.sum((q - qs) * eigenvectors.T, axis=1, keepdims=True)) / sigma ** 2 * (
    #         np.exp(-np.sum((q - qs) * eigenvectors.T, axis=1, keepdims=True) ** 2 / 2 / sigma ** 2)), axis=0,
    #     keepdims=True)
    # Gs_y = height * np.sum(
    #     eigenvectors.T[:, 1:2] * (-np.sum((q - qs) * eigenvectors.T, axis=1, keepdims=True)) / sigma ** 2 * (
    #         np.exp(-np.sum((q - qs) * eigenvectors.T, axis=1, keepdims=True) ** 2 / 2 / sigma ** 2)), axis=0,
    #     keepdims=True)

    # print(f'grad: {grad}')

    return grad#np.concatenate((Gs_x, Gs_y), axis=1)


def MD(q0, T, Tdeposite, height, sigma, dt=1e-3, beta=1.0, n=0, ic_method='PCA'):
    Nsteps = int(T / dt)
    NstepsDeposite = int(Tdeposite / dt)
    trajectories = np.zeros((Nsteps + 1, q0.shape[0], 2 + n))

    # print(trajectories.shape)

    variance = 0.7  # Threshhold for choosing number of eigenvectors
    q = q0

    qs = None
    eigenvectors = None
    save_eigenvalues = None
    choose_eigenvalue = None

    for i in tqdm(range(Nsteps)):

        trajectories[i, :] = q
        q = next_step(q, qs, eigenvectors, choose_eigenvalue, height, sigma, dt, beta)

        if (i + 1) % NstepsDeposite == 0:

            if qs is None:

                data = trajectories[:NstepsDeposite]  # (N_steps, 1, 2)
                data = np.squeeze(data, axis=1)  # (100, 2)
                mean_vector, selected_eigenvectors, eigenvalues = getCVs(data, ic_method)
                qs = mean_vector#data[-2:-1]#mean_vector
                eigenvectors = np.expand_dims(selected_eigenvectors, axis=0)
                save_eigenvalues = np.expand_dims(eigenvalues, axis=0)

                eigenvalues = np.expand_dims(eigenvalues, axis=0)
                choose_eigenvalue_tmp = np.zeros((1, 2 + n))
                cumsum = np.cumsum(eigenvalues, axis=1)
                var_ratio = cumsum / np.sum(save_eigenvalues)
                idx = np.argmax(var_ratio > variance)

                for s in range(idx + 1):
                    choose_eigenvalue_tmp[0, s] = 1
                choose_eigenvalue = choose_eigenvalue_tmp
                # print(choose_eigenvalue_tmp)

            else:
                data = trajectories[i - NstepsDeposite + 1:i + 1]
                data = np.squeeze(data, axis=1)  # (100, 2)
                mean_vector, selected_eigenvectors, eigenvalues = getCVs(data, ic_method)
                # print(data[-2:-1].shape, mean_vector.shape)
                # qs = np.concatenate([qs, data[-2:-1]], axis=0)
                qs = np.concatenate([qs, mean_vector], axis=0)
                eigenvectors = np.concatenate([eigenvectors, np.expand_dims(selected_eigenvectors, axis=0)], axis=0)
                save_eigenvalues = np.concatenate([save_eigenvalues, np.expand_dims(eigenvalues, axis=0)], axis=0)

                eigenvalues = np.expand_dims(eigenvalues, axis=0)
                choose_eigenvalue_tmp = np.zeros((1, 2 + n))
                cumsum = np.cumsum(eigenvalues, axis=1)
                var_ratio = cumsum / np.sum(eigenvalues)
                idx = np.argmax(var_ratio > variance)

                for s in range(idx + 1):
                    choose_eigenvalue_tmp[0, s] = 1
                choose_eigenvalue = np.concatenate([choose_eigenvalue, choose_eigenvalue_tmp], axis=0)
                # print(choose_eigenvalue_tmp)

            # print('eigenvectors shape: ', eigenvectors.shape)

    trajectories[Nsteps, :] = q
    return trajectories, qs, eigenvectors, save_eigenvalues, choose_eigenvalue


def next_step(qnow, qs, eigenvectors, choose_eigenvalue, height, sigma, dt=1e-3, beta=1.0, step_cap=0.3):
    if qs is None:
        qnext = qnow + (- gradV(qnow)) * dt + np.sqrt(2 * dt / beta) * np.random.randn(*qnow.shape)
    else:
        qnext = qnow + (- (gradV(qnow) + gradGaussians(qnow, qs, eigenvectors, choose_eigenvalue, height, sigma))) * dt + np.sqrt(
            2 * dt / beta) * np.random.randn(*qnow.shape)
    # print(qnow.shape, qnext.shape, np.random.randn(*qnow.shape))
    return qnext

def next_step(qnow, qs, eigenvectors, choose_eigenvalue, height, sigma, dt=1e-3, beta=1.0, step_cap=0.3):
    if qs is None:
        step = (-gradV(qnow)) * dt + np.sqrt(2 * dt / beta) * np.random.randn(*qnow.shape)
        step_size = np.linalg.norm(step)
        capped_step = np.where(step_size > step_cap, step * (step_cap / step_size), step)
        qnext = qnow + capped_step
    else:
        step = (- (gradV(qnow) + gradGaussians(qnow, qs, eigenvectors, choose_eigenvalue, height, sigma))) * dt + np.sqrt(
            2 * dt / beta) * np.random.randn(*qnow.shape)
        step_size = np.linalg.norm(step)
        capped_step = np.where(step_size > step_cap, step * (step_cap / step_size), step)
        qnext = qnow + capped_step

    return qnext

def findTSTime(trajectory):
    x_dimension = trajectory[:, 0, 0]
    y_dimension = trajectory[:, 0, 1]

    first_occurrence_index_1 = -1
    first_occurrence_index_2 = -1
    first_occurrence_index_3 = -1

    # Find the indices where the first dimension is greater than 0
    indices_1 = np.where(x_dimension > 0)[0]
    # Check if any such indices exist
    if indices_1.size > 0:
        # Get the first occurrence
        first_occurrence_index_1 = indices_1[0]
        print(f"The first time step where the first dimension is greater than 0 is: {first_occurrence_index_1}")
    else:
        print("There are no time steps where the first dimension is greater than 0.")

    # Find the indices where the second dimension is greater than 0
    indices_2 = np.where(y_dimension < 0)[0]
    # Check if any such indices exist
    if indices_2.size > 0:
        # Get the first occurrence
        first_occurrence_index_2 = indices_2[0]
        print(f"The first time step where the second dimension is less than 0 is: {first_occurrence_index_2}")
    else:
        print("There are no time steps where the second dimension is less than 0.")

    # Find the indices where the second dimension is greater than 0
    indices_3 = np.where((x_dimension > 0) & (y_dimension < 0))[0]
    # Check if any such indices exist
    if indices_3.size > 0:
        # Get the first occurrence
        first_occurrence_index_3 = indices_3[0]
        print(f"The first time step where the first dimension is greater than zero and the second dimension is less than 0 is: {first_occurrence_index_3}")
    else:
        print("There are no time steps where the first dimension is greater than zero and the second dimension is less than 0.")

    return first_occurrence_index_1, first_occurrence_index_2, first_occurrence_index_3


def run(T=100):
    # number of extra dimensions
    n = 0

    xx = np.linspace(-3, 3, 200)
    yy = np.linspace(-5, 5, 200)
    [X, Y] = np.meshgrid(xx, yy)  # 100*100
    W = potential(X, Y, np.zeros(n))
    W1 = W.copy()
    W1[W > 5] = float('nan')

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    contourf_ = ax1.contourf(X, Y, W1, levels=29)
    plt.colorbar(contourf_)

    # T = 100
    # T = 10
    T = 30
    dt = 1e-2
    beta = 50
    Tdeposite = 1
    height = 4.
    sigma = 0.7
    ic_method = 'PCA'

    foldername = 'WolfeSchlegel'

    cmap = plt.get_cmap('plasma')


    max_qn_val = 20
    q0 = np.concatenate((np.array([[-2.0, 2.0]]), np.array([np.random.rand(n)*(2*max_qn_val) - max_qn_val])), axis=1)
    trajectory, qs, eigenvectors, eigenvalues, choose_eigenvalue = MD(q0, T, Tdeposite=Tdeposite, height=height, sigma=sigma, dt=dt, beta=beta, n=n, ic_method=ic_method)  # (steps, bs, dim)
    
    first_occurrence_index_1, first_occurrence_index_2, first_occurrence_index_3 = findTSTime(trajectory)

    savename = 'PCA_results/T{}_Tdeposite{}_dt{}_height{}_sigma{}_beta{}_ic{}'.format(T, Tdeposite, dt, height, sigma, beta, ic_method)
    np.savez(savename, trajectory=trajectory, qs=qs, save_eigenvector=eigenvectors, save_eigenvalue=eigenvalues, choose_eigenvalue=choose_eigenvalue)

    print(trajectory)

    plotting = True
    if plotting:
        indices = np.arange(trajectory.shape[0])
        ax1.scatter(trajectory[:, 0, 0], trajectory[:, 0, 1], c=indices, cmap=cmap)

        # print(trajectory.shape, choose_eigenvalue)

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
        Gs = GaussiansPCA(np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1), np.zeros((num_points, n))], axis=1), qs, eigenvectors, choose_eigenvalue, height=height,
                          sigma=sigma)
        ax2 = fig.add_subplot(1, 3, 2)
        Sum = Gs.reshape(200, 200)+W1

        cnf2 = ax2.contourf(X, Y, Gs.reshape(200, 200), levels=29)
        plt.colorbar(cnf2)
        # print(eigenvectors.shape)
        indices = np.arange(qs.shape[0])
        ax2.scatter(qs[:, 0], qs[:, 1], c=indices, cmap=cmap)
        ax2.quiver(qs[:, 0], qs[:, 1], eigenvectors[:, 0, 0], eigenvectors[:, 1, 0])
        ax2.axis('equal')
        indices = np.arange(trajectory.shape[0])
        # ax2.scatter(trajectory[:, 0, 0], trajectory[:, 0, 1], c=indices, cmap=cmap, alpha=0.1)

        # ax2.scatter(trajectory[:, 0], trajectory[:, 1], c=indices, cmap=cmap)
        ax3 = fig.add_subplot(1, 3, 3)
        cnf3 = ax3.contourf(X, Y, Sum, levels=29)

        # fig.colorbar(contourf_)
        plt.title(f'Local {ic_method} dynamics')
        plt.show()

    return first_occurrence_index_1, first_occurrence_index_2, first_occurrence_index_3


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int, default=0)
    args = parser.parse_args()

    results = []
    
    for _ in range(1):
        i_1, i_2, i_3 = run(T=100)
        results.append((i_1, i_2, i_3))

    # Print results in a way that's easy to copy and paste
    for result in results:
        print(f"{result[0]}\t{result[1]}\t{result[2]}")



