import jax.numpy as jnp
import jax

def potential(qx, qy, qn):
    V = 100 * (qy - qx**2)**2 + (1 - qx)**2 + 100 * jnp.sum(qn**2)
    return V

def gradV(q):
    qx = q[:, 0:1]
    qy = q[:, 1:2]
    qn = q[:, 2:]
    Vx = -400 * qx * (qy - qx**2) - 2 * (1 - qx)
    Vy = 200 * (qy - qx**2)
    Vn = 100 * 2 * qn
    grad = jnp.concatenate((Vx, Vy, Vn), axis=1)
    return grad

def test_gradient(q, h=1e-5):
    analytical_grad = gradV(q)
    numerical_grad = jnp.zeros_like(analytical_grad)
    
    for i in range(q.shape[1]):
        # Create perturbation vectors
        e_i = jnp.zeros_like(q)
        e_i = e_i.at[:, i].set(h)
        
        # Compute finite differences
        V_plus = potential(q[:, 0] + e_i[:, 0], q[:, 1] + e_i[:, 1], q[:, 2:] + e_i[:, 2:])
        V_minus = potential(q[:, 0] - e_i[:, 0], q[:, 1] - e_i[:, 1], q[:, 2:] - e_i[:, 2:])
        numerical_grad = numerical_grad.at[:, i].set((V_plus - V_minus) / (2 * h))
    
    # Compare analytical and numerical gradients
    error = jnp.linalg.norm(analytical_grad - numerical_grad)
    return error, analytical_grad, numerical_grad

# Example usage:
q = jnp.array([[1.0, 2.0, 3.0, 4.0]])  # Example input
error, analytical_grad, numerical_grad = test_gradient(q)
print("Error:", error)
print("Analytical Gradient:", analytical_grad)
print("Numerical Gradient:", numerical_grad)

error, analytical_grad, numerical_grad = test_gradient(q, h=1e-4)
print("Error:", error)
print("Analytical Gradient:", analytical_grad)
print("Numerical Gradient:", numerical_grad)

error, analytical_grad, numerical_grad = test_gradient(q, h=1e-3)
print("Error:", error)
print("Analytical Gradient:", analytical_grad)
print("Numerical Gradient:", numerical_grad)
