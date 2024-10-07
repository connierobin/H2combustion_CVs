import numpy as np
import jax.numpy as jnp

def givens_rotation_matrix(i, j, theta, dim):
    rotation = np.eye(dim)
    rotation[i, i] = np.cos(theta)
    rotation[j, j] = np.cos(theta)
    rotation[i, j] = -np.sin(theta)
    rotation[j, i] = np.sin(theta)
    return rotation

def full_rotation_matrix(angle, dim):
    rotation = np.eye(dim)
    for i in range(0, dim - 1, 2):
        j = i + 1
        rotation_step = givens_rotation_matrix(i, j, angle, dim)
        rotation = rotation @ rotation_step

        print(f"Givens rotation matrix for coordinates {i} and {j} with angle {angle}:")
        print(rotation_step)
        print("Accumulated rotation matrix so far:")
        print(rotation)
    
    return rotation

def trig_rotation(q, angle):
    dim = len(q)
    q_rotated = np.array(q)
    
    for i in range(0, dim - 1, 2):
        j = i + 1
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        
        qi, qj = q_rotated[i], q_rotated[j]
        
        q_rotated[i] = cos_theta * qi - sin_theta * qj
        q_rotated[j] = sin_theta * qi + cos_theta * qj

        print(f"After rotating coordinates {i} and {j} with angle {angle}:")
        print(f"q_rotated[{i}] = {q_rotated[i]}")
        print(f"q_rotated[{j}] = {q_rotated[j]}")
        print(q_rotated)

    return q_rotated

# OLD and only half fixed
# def rotated_rastringin_potential(q, angle):
#     # print(f'q.shape: {q.shape}')
#     # one_point = False
#     # if q.ndim == 1:
#     #     one_point = True
#     #     q = q.reshape(1, -1)  # Ensure q is a 2D array with shape (1, dim)
    
#     dim = q.shape[-1]
#     print(f'dim: {dim}')
#     rotation = full_rotation_matrix(angle, dim)
#     q_rotated = rotation @ q
#     # q_rotated = q @ rotation.T  # Apply rotation to each point
#     A = 10
#     B = 0.5
#     d = q_rotated.shape[0]
#     xsq = B * jnp.power(q_rotated, 2)
#     wave = jnp.cos(2 * np.pi * q_rotated)
#     result = A * d + jnp.sum(xsq - A * wave, axis=0)
#     # if one_point:
#     #     result = jnp.sum(result, axis =0)
#     return result

# def grad_rotated_rastringin(q, angle):
#     print(f'q.shape: {q.shape}')
#     if q.ndim == 1:
#         q = q.reshape(1, -1)  # Ensure q is a 2D array with shape (1, dim)

#     dim = q.shape[-1]
#     rotation = full_rotation_matrix(angle, dim)
#     q_rotated = q @ rotation.T  # Apply rotation to each point
#     B = 0.5
#     V = B * 2 * q_rotated + 10 * jnp.sin(2 * jnp.pi * q_rotated) * 2 * jnp.pi
#     return -rotation.T @ V

def test_full_rotation_matrix():
    q = np.array([1.0, 1.0, 1.0, 1.0])
    angle = np.pi / 4  # Angle for Givens rotations
    
    rotation = full_rotation_matrix(angle, len(q))
    q_rotated_matrix = rotation @ q
    q_rotated_trig = trig_rotation(q, angle)
    
    print("Rotation matrix result:")
    print(q_rotated_matrix)
    
    print("Trig rotation result:")
    print(q_rotated_trig)
    
    return np.allclose(q_rotated_matrix, q_rotated_trig)

# Example usage for testing the rotation matrix
# if test_full_rotation_matrix():
#     print("The full_rotation_matrix function passes the test!")
# else:
#     print("The full_rotation_matrix function fails the test.")

# # Example usage for rotated Rastringin potential
# q = np.array([1.0, 1.0, 1.0, 1.0])
# angle = np.pi / 4

# potential = rotated_rastringin_potential(q, angle)
# gradient = grad_rotated_rastringin(q, angle)

# print("Rotated Rastringin Potential:", potential)
# print("Gradient of Rotated Rastringin Potential:", gradient)










def full_rotation_matrix(angle, dim):
    def givens_rotation_matrix(i, j, theta, dim):
        rotation = np.eye(dim)
        rotation[i, i] = np.cos(theta)
        rotation[j, j] = np.cos(theta)
        rotation[i, j] = -np.sin(theta)
        rotation[j, i] = np.sin(theta)
        return rotation
    
    rotation = np.eye(dim)
    for i in range(0, dim - 1, 2):
        j = i + 1
        rotation_step = givens_rotation_matrix(i, j, angle, dim)
        rotation = rotation @ rotation_step
    return rotation

def rotated_rastringin_potential(q, angle):
    need_swap = False
    if q.ndim > 1:
        need_swap = True
        # Move the last dimension to the first position
        q = np.moveaxis(q, -1, 0)
    dim = len(q)
    rotation = full_rotation_matrix(angle, dim)
    # print(f'rotation: {rotation}')
    q_rotated = rotation @ q
    A = 10
    B = 0.5
    d = q_rotated.shape[0]
    xsq = B * jnp.power(q_rotated, 2)
    wave = jnp.cos(2 * np.pi * q_rotated)
    result = A * d + jnp.sum(xsq - A * wave, axis=0)
    if need_swap:
        result = np.moveaxis(result, 0, -1)
    return result

def grad_rotated_rastringin(q, angle):
    need_swap = False
    if q.ndim > 1:
        need_swap = True
        # Move the last dimension to the first position
        q = np.moveaxis(q, -1, 0)
    dim = len(q)
    rotation = full_rotation_matrix(angle, dim)
    q_rotated = rotation @ q

    A = 10
    B = 0.5
    V = B * 2 * q_rotated + A * jnp.sin(2 * jnp.pi * q_rotated) * 2 * jnp.pi
    result = -rotation.T @ V
    if need_swap:
        result = np.moveaxis(result, 0, -1)
    return result

def alt_grad_rotated_rastringin(q, angle):
    dim = len(q)
    rotation = full_rotation_matrix(angle, dim)
    q_rotated = rotation @ q
    return -rotation.T @ grad_rastrigin(q_rotated)

def alt_rotated_rastringin_potential(q, angle):
    dim = len(q)
    rotation = full_rotation_matrix(angle, dim)
    q_rotated = rotation @ q
    return rastrigin_potential(q_rotated)

def rastrigin_potential(q):
    need_swap = False
    if q.ndim > 1:
        need_swap = True
        # Move the last dimension to the first position
        q = np.moveaxis(q, -1, 0)
    A = 10
    B = 0.5
    d = q.shape[0]
    xsq = B * jnp.power(q, 2)
    wave = jnp.cos(2 * np.pi * q)
    result = A * d + jnp.sum(xsq - A * wave, axis=0)
    if need_swap:
        result = np.moveaxis(result, 0, -1)
    return result

def grad_rastrigin(q):
    need_swap = False
    if q.ndim > 1:
        need_swap = True
        # Move the last dimension to the first position
        q = np.moveaxis(q, -1, 0)
    A = 10
    B = 0.5
    xsq = B * 2 * q
    wave = jnp.sin(2 * np.pi * q) * 2 * np.pi
    # V = B * 2 * q + 10 * jnp.sin(2 * jnp.pi * q) * 2 * jnp.pi
    V = xsq + A * wave
    if need_swap:
        V = np.moveaxis(V, 0, -1)
    return -V


def check_gradient_finite_difference(q, angle, epsilon=1e-2):
    dim = len(q)
    rotation = full_rotation_matrix(angle, dim)
    
    # Analytical gradient
    grad_analytical = grad_rotated_rastringin(q, angle)

    # Finite difference gradient
    grad_finite_diff = np.zeros_like(q)
    for i in range(len(q)):
        q_pos = np.array(q, dtype=float)
        q_neg = np.array(q, dtype=float)
        q_pos[i] += epsilon
        q_neg[i] -= epsilon

        potential_pos = rotated_rastringin_potential(q_pos, angle)
        # print(f'potential_pos: {potential_pos}')
        potential_neg = rotated_rastringin_potential(q_neg, angle)

        # print(f'potential_neg: {potential_neg}')

        grad_finite_diff[i] = (potential_pos - potential_neg) / (2 * epsilon)

    print(f'Function Value: {rotated_rastringin_potential(q, angle)}')
    print("Analytical Gradient:", grad_analytical)
    print("Finite Difference Gradient:", grad_finite_diff)

    return np.allclose(grad_analytical, grad_finite_diff, atol=1e-4)

def alt_check_gradient_finite_difference(q, angle, epsilon=1e-2):
    dim = len(q)
    rotation = full_rotation_matrix(angle, dim)
    
    # Analytical gradient
    grad_analytical = alt_grad_rotated_rastringin(q, angle)

    # Finite difference gradient
    grad_finite_diff = np.zeros_like(q)
    for i in range(len(q)):
        q_pos = np.array(q, dtype=float)
        q_neg = np.array(q, dtype=float)
        q_pos[i] += epsilon
        q_neg[i] -= epsilon

        potential_pos = alt_rotated_rastringin_potential(q_pos, angle)
        potential_neg = alt_rotated_rastringin_potential(q_neg, angle)

        grad_finite_diff[i] = (potential_pos - potential_neg) / (2 * epsilon)

    print("Analytical Gradient:", grad_analytical)
    print("Finite Difference Gradient:", grad_finite_diff)

    return np.allclose(grad_analytical, grad_finite_diff, atol=1e-4)


angle = np.pi / 4
# q = np.array([[ 0.,0.,         0.09666521]])
q = np.array([[-0.5491254, -0.603084,   0.4924881]])



print(f'result: {rastrigin_potential(q)}')
print(f'correct result: {np.array([rastrigin_potential(q[0])])}')
print(f'grad: {grad_rastrigin(q)}')
print(f'correct grad: {np.array([grad_rastrigin(q[0])])}')
print(f'rotated result: {rotated_rastringin_potential(q, angle)}')
print(f'correct rotated result: {np.array([rotated_rastringin_potential(q[0], angle)])}')
print(f'rotated grad: {grad_rotated_rastringin(q, angle)}')
print(f'correct rotated grad: {np.array([grad_rotated_rastringin(q[0], angle)])}')

print('ALT')
# Example usage
q = np.array([1.0, 1.0, 1.0, 1.0])
angle = 0
if alt_check_gradient_finite_difference(q, angle):
    print("The gradient is correct.")
else:
    print("The gradient is incorrect.")


# Example usage
q = np.array([1.0, 1.0, 1.0, 1.0])
angle = np.pi / 4
if alt_check_gradient_finite_difference(q, angle):
    print("The gradient is correct.")
else:
    print("The gradient is incorrect.")

print('REGULAR')

# Example usage
q = np.array([1.0, 1.0, 1.0, 1.0])
angle = 0
if check_gradient_finite_difference(q, angle):
    print("The gradient is correct.")
else:
    print("The gradient is incorrect.")


# Example usage
q = np.array([1.0, 1.0, 1.0, 1.0])
angle = np.pi / 4
if check_gradient_finite_difference(q, angle):
    print("The gradient is correct.")
else:
    print("The gradient is incorrect.")


# Example usage
q = np.array([0.1, 0.1, 0.1, 0.1])
angle = np.pi / 4
if check_gradient_finite_difference(q, angle):
    print("The gradient is correct.")
else:
    print("The gradient is incorrect.")



