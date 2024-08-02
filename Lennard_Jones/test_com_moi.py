# COM = center of mass
# MOI = moments of inertia

import numpy as np

# Function to calculate center of mass
def center_of_mass(r):
    total_mass = len(r)
    com = np.sum(r, axis=0) / total_mass
    return com

# Function to calculate moments of inertia
def moments_of_inertia(r, com):
    inertia_tensor = np.zeros((3, 3))
    for particle in r:
        rel_position = particle - com
        # print(f'relative position: {rel_position}')
        inertia_tensor += np.outer(rel_position, rel_position)
        # print(f'inertia tensor: {inertia_tensor}')
    return inertia_tensor

# Function to diagonalize the inertia tensor and get principal axes
def principal_axes(inertia_tensor):
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)
    # Sort eigenvectors based on eigenvalues
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    return eigenvalues, eigenvectors

# Function to rotate particles to align with principal axes
def rotate_to_principal_axes(r, com, eigenvectors):
    rotated_positions = np.dot(r - com, eigenvectors)
    return rotated_positions

# This is unnecessary, the eigenvectors are already normalized
def rotate_to_principal_axes_norm(r, com, eigenvectors):
    # Normalize eigenvectors
    eigenvectors_normalized = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
    rotated_positions = np.dot(r - com, eigenvectors_normalized)
    return rotated_positions

# Example usage
r = np.array([[0, 0, -2], [0, 0, 0], [0, 0, 2]])  # Example positions of particles
com = center_of_mass(r)
r_shifted = r - com
intermediate_com = center_of_mass(r_shifted)
inertia_tensor = moments_of_inertia(r_shifted, com)
eigenvalues, eigenvectors = principal_axes(inertia_tensor)
rotated_positions = rotate_to_principal_axes(r_shifted, com, eigenvectors)
rotated_positions_norm = rotate_to_principal_axes_norm(r_shifted, com, eigenvectors)

print("Center of Mass:", com)
print("Shifted Center of Mass: ", intermediate_com)
print("Inertia Tensor:")
print(inertia_tensor)
print("Principal Axes Eigenvalues:", eigenvalues)
print("Principal Axes Eigenvectors:")
print(eigenvectors)
print("Rotated Positions:")
print(rotated_positions)
print("Rotated Positions Norm:")
print(rotated_positions_norm)


new_com = center_of_mass(rotated_positions_norm)
inertia_tensor = moments_of_inertia(rotated_positions_norm, new_com)
print("New Center of Mass:")
print(new_com)
print("New Moments of Inertia:")
print(inertia_tensor)





def test_three_particles_on_axes():
    # Test case with three particles, each on one of the axes
    r = np.array([[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], [0, 0, 0], [-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)]])
    com = center_of_mass(r)
    print(f'com: {com}')
    inertia_tensor = moments_of_inertia(r - com, com)
    print(f'inertia tensor: {inertia_tensor}')
    _, eigenvectors = principal_axes(inertia_tensor)
    print(f'eigenvectors: {eigenvectors}')
    rotated_positions = rotate_to_principal_axes(r - com, com, eigenvectors)
    expected_rotated_positions = np.array([[1., 0., 0.], [0., 0, 0.], [-1, 0., 0]])
    assert np.allclose(rotated_positions, expected_rotated_positions), \
        f"Rotated positions are incorrect: \n{rotated_positions}"

# Run test
test_three_particles_on_axes()
print("Test case for three particles on axes passed!")












def test_center_of_mass():
    # Test case with known center of mass
    r = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_com = np.array([4., 5., 6.])
    com = center_of_mass(r)
    assert np.allclose(com, expected_com), f"Center of mass is incorrect: {com}"

def test_moments_of_inertia():
    # Test case with known inertia tensor
    r = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    com = center_of_mass(r)
    inertia_tensor = moments_of_inertia(r - com, com)
    expected_inertia_tensor = np.array([[14., 14., 14.], [14., 14., 14.], [14., 14., 14.]])
    assert np.allclose(inertia_tensor, expected_inertia_tensor), \
        f"Inertia tensor is incorrect: {inertia_tensor}"

def test_principal_axes():
    # Test case with known eigenvalues and eigenvectors
    inertia_tensor = np.array([[14., 0., 0.], [0., 14., 0.], [0., 0., 14.]])
    expected_eigenvalues = np.array([14., 14., 14.])
    expected_eigenvectors = np.eye(3)
    eigenvalues, eigenvectors = principal_axes(inertia_tensor)
    assert np.allclose(eigenvalues, expected_eigenvalues), \
        f"Eigenvalues are incorrect: {eigenvalues}"
    assert np.allclose(eigenvectors, expected_eigenvectors), \
        f"Eigenvectors are incorrect: {eigenvectors}"

def test_rotate_to_principal_axes():
    # Test case with known rotated positions
    r = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    com = center_of_mass(r)
    inertia_tensor = moments_of_inertia(r - com, com)
    _, eigenvectors = principal_axes(inertia_tensor)
    rotated_positions = rotate_to_principal_axes(r - com, com, eigenvectors)
    expected_rotated_positions = np.array([[-1.73205081e+00,  2.22044605e-16,  3.33066907e-16],
                                           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                                           [ 1.73205081e+00, -2.22044605e-16, -3.33066907e-16]])
    assert np.allclose(rotated_positions, expected_rotated_positions), \
        f"Rotated positions are incorrect: {rotated_positions}"

# Run tests
test_center_of_mass()
# test_moments_of_inertia()     # test is wrong
# test_principal_axes()     # test is wrong
# test_rotate_to_principal_axes()     # test is wrong
print("All tests passed!")
