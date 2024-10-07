# H2combustion_CVs

** temporarily set to public, do not share **

Repo: Research code for testing autoencoders as a replacement for principle component analysis in metadynamics. 

See **contents** and **figures** below to get an overview of this work. 

## Contents:

`H2COMBUSTION_DATA-main-2`: data from the public dataset "A Benchmark Data Set for Hydrogen Combustion" used here to test metadynamics on real systems with just a few atoms

`Lennard_Jones`: tests with a collection of identical Lennard-Jones particles, an analytic approximation to atomic potentials that are easy to calculate but not entirely accurate. Found here to produce shallow energy wells that are not a realistic test of the method. 

`MTDPCA_Reaction09`: data from the hydrogen combustion dataset, tests involving reaction 9

`Reaction2`: data from the hydrogen combustion dataset, tests involving reaction 2

`any_potential`: setup for a generalized testing framework, with results using the Wolfe-Schlegel potential

`dw`: tests on a double well potential

`rastrigin`: tests on a system with periodic potential wells, with auxiliary parabolic dimensions

`rosenbrock`: tests with the "banana" potential

`rotated_rastrigin`: rastrigin potential rotated by 45 degrees to prevent bias along Cartesian directions (most recent work is here!)

`wolfeschlegel`: tests on a potential with 4 potential wells and auxiliary dimensions

## Figures

### Big Picture

The purpose of using autoencoders is to encode more complex molecular movements than can be expressed with PCA. 
![pca_v_ae_plot](https://github.com/user-attachments/assets/b300ae6b-4093-4638-af13-ab2f0d384c88)

### Wolfe Schlegel

Potential function:

$10(x^4 + y^4 - 2x^2 - 4y^2 + xy + 0.2x + 0.1y + 10z_i^2)$

where $z_i$ are auxiliary dimensions included to test whether the method explores many dimensions simultaneously, or if this causes interference. 

Compare the plots of simulations using the existing method with principle component analysis (PCA) to simulations using the new method introduced here using autoencoders (AE). 

#### Discussion
- PCA discovers fewer wells
- The center plot is a better inverse of the left plot with AE
- AE on average takes fewer iterations to explore the wells

#### Plots
PCA simulation with 3 auxiliary dimensions:
![pca-ws-n3-1-zoom](https://github.com/user-attachments/assets/61e1beb0-aba1-4c9f-87e9-151a46aedc30)

Autoencoder (AE) simulation with 3 auxiliary dimensions:
![ae-ws-n3-2-zoom](https://github.com/user-attachments/assets/054724f0-bf75-4d09-a5d0-d8a6f2de3c5d)

AE outperforms PCA:
![avg_well_iters_1](https://github.com/user-attachments/assets/f5d13738-a417-4d8d-81db-40fbd8ce9eef)


### Rotated Rastrigin

Potential function (before rotation):

$10d + \sum_{i=1}^d (0.5 x_i^2 + 10 \cos (2 \pi x_i))$

where $d$ is the total number of dimensions. 

#### Discussion
- PCA has a side effect of extended inaccurate potential contributions
- PCA trends in one direction whereas AE explores the lower energy potentials first

#### Plots
PCA results with $d=10$:
![pca_n8_12](https://github.com/user-attachments/assets/185a7943-2c26-4d1f-8aa0-f52526d120c8)

AE results with $d=10$:
![ae_n8_k2_3](https://github.com/user-attachments/assets/0af1b005-405c-4462-b782-17398f8345c9)


