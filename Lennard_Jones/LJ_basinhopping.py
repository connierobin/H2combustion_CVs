import numpy as np
from scipy.optimize import basinhopping

sigma = 1
epsilon = 1

def Psi(dist):
    return 4*epsilon*(sigma**12/dist**12 - sigma**6/dist**6)

def PsiSq(distSq):
    return 4*epsilon*(sigma**12/distSq**6 - sigma**6/distSq**3)

def LJpotential(r): ## r size 1*M M is a multiple of 3
    M = r.shape[1]
    Natoms = M//3
    V = 0
    for i in range(Natoms-1):
        for j in range(i+1,Natoms):
            atom1 = r[0, i*3:i*3+3]
            atom2 = r[0, j*3:j*3+3]
            V += PsiSq(np.sum((atom1-atom2)**2))
    return V

def LJpotential1(r): ### r size: M
    r = np.array(r).reshape(1,-1)
    return LJpotential(r)

# Set up basinhopping optimization
minimizer_kwargs = {"method": "L-BFGS-B"}
result = basinhopping(LJpotential1, x0=np.random.randn(60), minimizer_kwargs=minimizer_kwargs, niter=1000)
print(result)
print(result.fun)

# r = np.zeros((1,10))
# print(LJpotential1(np.array([[[14.64611025, 16.38061715, 17.64513847, 15.51730755, 16.77449607]]])))