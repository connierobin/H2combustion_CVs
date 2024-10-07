import numpy as np
import jax.numpy as jnp


def check_gradient_finite_difference_original(q, epsilon=1e-2):
    def rastrigin_potential(q):
        A = 10
        B = 0.5
        d = q.shape[0]
        xsq = B * jnp.power(q, 2)
        wave = jnp.cos(2 * np.pi * q)
        return A * d + jnp.sum(xsq - A * wave, axis=0)

    def grad_rastrigin(q):
        B = 0.5
        xsq = B * 2 * q
        wave = jnp.sin(2 * np.pi * q) * 2 * np.pi
        # V = B * 2 * q + 10 * jnp.sin(2 * jnp.pi * q) * 2 * jnp.pi
        V = xsq + A * wave
        return -V

    # Potential value
    potential_value = rastrigin_potential(q)

    # Analytical gradient
    grad_analytical = grad_rastrigin(q)

    # Finite difference gradient
    grad_finite_diff = np.zeros_like(q)
    for i in range(len(q)):
        q_pos = np.array(q, dtype=float)
        q_neg = np.array(q, dtype=float)
        q_pos[i] += epsilon
        q_neg[i] -= epsilon

        potential_pos = rastrigin_potential(q_pos)
        potential_neg = rastrigin_potential(q_neg)
        potential_val = rastrigin_potential(q)

        print(f'potential_pos: {potential_pos}')
        print(f'potential_neg: {potential_neg}')
        print(f'potential: {potential_val}')

        grad_finite_diff[i] = (potential_pos - potential_neg) / (2 * epsilon)

    print("potential value: ", potential_value)
    print("Analytical Gradient (Original):", grad_analytical)
    print("Finite Difference Gradient (Original):", grad_finite_diff)

    return np.allclose(grad_analytical, grad_finite_diff, atol=1e-4)

# Example usage for original Rastrigin potential
q = np.array([1.0])
if check_gradient_finite_difference_original(q):
    print("The original gradient is correct.")
else:
    print("The original gradient is incorrect.")

# Example usage for original Rastrigin potential
q = np.array([1.0, 1.0, 1.0, 1.0])
if check_gradient_finite_difference_original(q):
    print("The original gradient is correct.")
else:
    print("The original gradient is incorrect.")


# Example usage for original Rastrigin potential
q = np.array([46.234, 2.64, -0.34, 6.234])
if check_gradient_finite_difference_original(q):
    print("The original gradient is correct.")
else:
    print("The original gradient is incorrect.")
