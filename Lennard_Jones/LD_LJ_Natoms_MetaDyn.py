import numpy as np
from tqdm import tqdm

def Psi(dist):

    sigma = 1
    epsilon = 1

    return 4*epsilon*(sigma**12/dist**12 - sigma**6/dist**6)

def PsiSq(distSq):

    sigma = 1
    epsilon = 1

    return 4*epsilon*(sigma**12/distSq**6 - sigma**6/distSq**3)

def GradPsi(atom1, atom2):
    sigma = 1
    epsilon = 1

    distSq = np.sum((atom1-atom2)**2)
    DPsi_DdistSq = 4*epsilon*(-6*sigma**12/distSq**7 + 3*sigma**6/distSq**4)
    return 2*(atom1-atom2)*DPsi_DdistSq

def LJpotential(r): ## r size 1*M M is a multiple of 3
    M = r.shape[1]
    Natoms = M // 3
    V = 0
    for i in range(Natoms-1):
        for j in range(i+1, Natoms):
            atom1 = r[0, i*3:i*3+3]
            atom2 = r[0, j*3:j*3+3]
            V += PsiSq(np.sum((atom1-atom2)**2))
    return V

def GradLJpotential(r): ## r size 1*M M is a multiple of 3
    M = r.shape[1]
    grad = np.zeros((1,M))
    Natoms = M // 3

    for i in range(Natoms):
        lst = np.arange(Natoms).tolist()
        lst.remove(i)
        # print(lst)
        for j in lst:
            atom1 = r[0, i * 3:i * 3 + 3]
            atom2 = r[0, j * 3:j * 3 + 3]
            grad[0,i * 3:i * 3 + 3] += GradPsi(atom1, atom2)
    return grad

def PCA(data):  # datasize: N * dim
    # Step 4.1: Compute the mean of the data
    data_z = data  # bs*3

    mean_vector = np.mean(data_z, axis=0, keepdims=True)

    # Step 4.2: Center the data by subtracting the mean
    centered_data = (data_z - mean_vector)

    # Step 4.3: Compute the covariance matrix of the centered data
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # Step 4.4: Perform eigendecomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 4.5: Sort the eigenvectors based on eigenvalues (descending order)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 4.6: Choose the number of components (optional)
    k = 4  # Set the desired number of components

    # Step 4.7: Retain the top k components
    selected_eigenvectors = eigenvectors[:, 0:k]

    return mean_vector, selected_eigenvectors

def GradGuassian(x, centers, eigenvectors, h, sigma):
    # x: 1 * M
    # centers: N * M
    # eigenvectors: N * M * k
    x_minus_centers = x - centers # N * M
    x_minus_centers = np.expand_dims(x_minus_centers, axis=1) # N * 1 * M
    x_projected = np.matmul(x_minus_centers, eigenvectors) # N * 1 * k
    x_projected_sq_sum = np.sum(x_projected**2, axis=(-2, -1)) # N
    exps = -h/sigma**2*np.exp(-np.expand_dims(x_projected_sq_sum, axis=1)/2/sigma**2) # N * 1
    PTPx = np.matmul(eigenvectors, np.transpose(x_projected, axes=(0,2,1))) # N * M * 1
    PTPx = np.squeeze(PTPx, axis=2) # N * M
    grad = np.sum(exps * PTPx, axis=0, keepdims=True) # 1 * M

    return grad

def SumGuassian(x, centers, eigenvectors, h, sigma):
    # x: 1 * M
    # centers: N * M
    # eigenvectors: N * M * k
    x_minus_centers = x - centers # N * M
    x_minus_centers = np.expand_dims(x_minus_centers, axis=1) # N * 1 * M
    x_projected = np.matmul(x_minus_centers, eigenvectors) # N * 1 * k
    x_projected_sq_sum = np.sum(x_projected**2, axis=(-2, -1)) # N
    exps = h*np.exp(-np.expand_dims(x_projected_sq_sum, axis=1)/2/sigma**2) # N * 1
    grad = np.sum(exps, axis=0, keepdims=True) # 1 * M

    return grad

def LD_MetaDyn(M, T, Tdeposite, dt, h, sigma, kbT):
    # M: dim
    r = np.random.randn(1, M)*1

    Nsteps = round(T / dt)
    NstepsDeposite = round(Tdeposite / dt)
    print(NstepsDeposite)
    trajectories4PCA = np.zeros((NstepsDeposite, 1, M))

    rcenters = None
    eigenvectors = None
    # Smallest = float('inf')

    for i in tqdm(range(Nsteps)):
        print(LJpotential(r))
        r = next_step(r, rcenters, eigenvectors, h, sigma, kbT, dt)
        # print(trajectories4PCA.shape, r.shape)
        trajectories4PCA[i % NstepsDeposite, :] = r
        # print(trajectories4PCA)
        if (i + 1) % NstepsDeposite == 0:
            # print(r, LJpotential(r))
            # r = next_step(r, rcenters, eigenvectors, h, sigma, kbT, dt)
            if rcenters is None:
                ### conducting PCA ###
                data = trajectories4PCA

                data = np.squeeze(data, axis=1)
                mean_vector, selected_eigenvectors = PCA(data)
                # print(selected_eigenvectors.shape, eigenvalues.shape)
                rcenters = mean_vector
                eigenvectors = np.expand_dims(selected_eigenvectors, axis=0)

                ### reset the PCA dataset
                trajectories4PCA = np.zeros((NstepsDeposite, 1, M))
            else:
                ### conducting PCA ###
                data = trajectories4PCA

                data = np.squeeze(data, axis=1)
                mean_vector, selected_eigenvectors = PCA(data)

                rcenters = np.concatenate([rcenters, mean_vector], axis=0)
                eigenvectors = np.concatenate([eigenvectors, np.expand_dims(selected_eigenvectors, axis=0)], axis=0)
                # print(rcenters.shape, eigenvectors.shape)

                # if rcenters.shape[0]>20:
                #     rcenters = rcenters[-50:]
                #     eigenvectors = eigenvectors[-50:]
                # print(rcenters.shape, eigenvectors.shape)
                ### reset the PCA dataset
                trajectories4PCA = np.zeros((NstepsDeposite, 1, M))

    return None

def next_LD(r, dt, kbT):

    rnew = r - (GradLJpotential(r)) * dt + np.sqrt(2 * dt *kbT) * np.random.randn(*r.shape)

    return rnew

def next_LD_Gaussian(r, dt, rcenters, eigenvectors, h, sigma, kbT):

    rnew = r - (GradLJpotential(r) + GradGuassian(r, rcenters, eigenvectors, h, sigma)) * dt + np.sqrt(2 * dt * kbT) * np.random.randn(*r.shape)

    return rnew

def next_step(r, rcenters, eigenvectors, h, sigma, kbT, dt):

    if rcenters is None:
        r = next_LD(r, dt, kbT)
    else:
        r = next_LD_Gaussian(r, dt, rcenters, eigenvectors, h, sigma, kbT)
    return r

M = 30
T = 20
Tdeposite = 0.05
dt = 0.001
h = 0.1
sigma = 0.1
kbT = 0.01

LD_MetaDyn(M, T, Tdeposite, dt, h, sigma, kbT)

# N = 5
# M = 7
# k = 2
# h = 0.1
# sigma = 0.2
#
# x = np.random.rand(1, M)
# centers = np.random.rand(N, M)
# eigenvectors = np.random.rand(N, M, k)
# print(GradGuassian(x, centers, eigenvectors, h, sigma))
# for i in range(M):
#     shift = 0.0001
#     e=np.zeros((1,M))
#     e[0, i]= shift
#     print((SumGuassian(x+e, centers, eigenvectors, h, sigma)-SumGuassian(x, centers, eigenvectors, h, sigma))/shift)
# LD(24, 0.001, 200)
# #
# # Set up basinhopping optimization
# minimizer_kwargs = {"method": "L-BFGS-B"}
# result = basinhopping(LJpotential1, x0=[0.0, 0.0, 0.0, -0.1, 0.1], minimizer_kwargs=minimizer_kwargs, niter=10000)
# print(result)
# print(result.fun)
#
# r = np.zeros((1,10))
# print(LJpotential(np.array([[0.1, -0.2, 0.1, -0.1, 0.1]])))