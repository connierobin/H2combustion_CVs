import numpy as np
import jax.numpy as jnp
from jax import grad

# Define the potential function, term by term
def potential_term1(qx, qy, qn):
    return 10 * (qx**4)

def potential_term2(qx, qy, qn):
    return 10 * (qy**4)

def potential_term3(qx, qy, qn):
    return 10 * (-2 * qx**2)

def potential_term4(qx, qy, qn):
    return 10 * (-4 * qy**2)

def potential_term5(qx, qy, qn):
    return 10 * (qx * qy)

def potential_term6(qx, qy, qn):
    return 10 * (0.2 * qx)

def potential_term7(qx, qy, qn):
    return 10 * (0.1 * qy)

def potential_term8(qx, qy, qn):
    print(f'qn: {qn}')
    print(f'qn^2: {qn**2}')
    sum = qn**2
    print(f'sum[0] + sum[1]: {sum[0][0] + sum[0][1]}')
    print(f'sum(qn^2): {jnp.sum(qn**2, axis=1)}')
    return 10 * (jnp.sum(qn**2, axis=1))

# Complete potential function
def potential(q):
    qx = q[:, 0]
    qy = q[:, 1]
    qn = q[:, 2:]
    return (potential_term1(qx, qy, qn) + potential_term2(qx, qy, qn) + 
            potential_term3(qx, qy, qn) + potential_term4(qx, qy, qn) + 
            potential_term5(qx, qy, qn) + potential_term6(qx, qy, qn) + 
            potential_term7(qx, qy, qn) + potential_term8(qx, qy, qn))

# Define the analytical gradient function, term by term
def grad_term1(q):
    qx = q[:, 0:1]
    grad_x = 40 * qx**3
    grad_y = jnp.zeros_like(qx)
    grad_n = jnp.zeros_like(q[:, 2:])
    return jnp.concatenate((grad_x, grad_y, grad_n), axis=1)

def grad_term2(q):
    qy = q[:, 1:2]
    grad_x = jnp.zeros_like(qy)
    grad_y = 40 * qy**3
    grad_n = jnp.zeros_like(q[:, 2:])
    return jnp.concatenate((grad_x, grad_y, grad_n), axis=1)

def grad_term3(q):
    qx = q[:, 0:1]
    grad_x = -40 * qx
    grad_y = jnp.zeros_like(qx)
    grad_n = jnp.zeros_like(q[:, 2:])
    return jnp.concatenate((grad_x, grad_y, grad_n), axis=1)

def grad_term4(q):
    qy = q[:, 1:2]
    grad_x = jnp.zeros_like(qy)
    grad_y = -80 * qy
    grad_n = jnp.zeros_like(q[:, 2:])
    return jnp.concatenate((grad_x, grad_y, grad_n), axis=1)

def grad_term5(q):
    qx = q[:, 0:1]
    qy = q[:, 1:2]
    grad_x = 10 * qy
    grad_y = 10 * qx
    grad_n = jnp.zeros_like(q[:, 2:])
    return jnp.concatenate((grad_x, grad_y, grad_n), axis=1)

def grad_term6(q):
    grad_x = 2 * jnp.ones_like(q[:, 0:1])
    grad_y = jnp.zeros_like(q[:, 1:2])
    grad_n = jnp.zeros_like(q[:, 2:])
    return jnp.concatenate((grad_x, grad_y, grad_n), axis=1)

def grad_term7(q):
    grad_x = jnp.zeros_like(q[:, 0:1])
    grad_y = jnp.ones_like(q[:, 1:2])
    grad_n = jnp.zeros_like(q[:, 2:])
    return jnp.concatenate((grad_x, grad_y, grad_n), axis=1)

def grad_term8(q):
    qn = q[:, 2:]
    grad_x = jnp.zeros_like(q[:, 0:1])
    grad_y = jnp.zeros_like(q[:, 1:2])
    grad_n = 20 * qn
    print(f'qn: {qn}')
    print(f'20 * qn: {20 * qn}')
    return jnp.concatenate((grad_x, grad_y, grad_n), axis=1)

# Combine all gradient terms into a single analytical gradient function
def gradV_sum(q):
    return (grad_term1(q) + grad_term2(q) + grad_term3(q) + grad_term4(q) +
            grad_term5(q) + grad_term6(q) + grad_term7(q) + grad_term8(q))

def gradV(q):
    qx = q[:, 0:1]
    qy = q[:, 1:2]
    qn = q[:, 2:]
    Vx = 10 * (4 * qx**3 - 4 * qx + qy + 0.2)
    Vy = 10 * (4 * qy**3 - 8 * qy + qx + 0.1)
    Vn = 10 * 2*qn
    grad = jnp.concatenate((Vx, Vy, Vn), axis=1)
    return grad

# Function to calculate numerical gradient
def numerical_grad(f, x, epsilon=1e-3):
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.shape[1]):
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        x_plus[:, i] += epsilon
        x_minus[:, i] -= epsilon
        grad[:, i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
    return grad

# Wrap each potential term for numerical gradient comparison
def wrap_term(term_func):
    def wrapped(q):
        qx = q[:, 0]
        qy = q[:, 1]
        qn = q[:, 2:]
        return term_func(qx, qy, qn).reshape(-1, 1)
    return wrapped

# Test function for individual terms
def test_grad_term(term_func, grad_func, term_name):
    # Test point
    q = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=float)
    
    # Analytical gradient for the term
    analytical_grad = grad_func(q)
    
    # Numerical gradient for the term
    numerical_grad_result = numerical_grad(wrap_term(term_func), q)
    
    # Print the gradients for debugging
    print(f"Testing {term_name}")
    print(f"Analytical gradient: {analytical_grad}")
    print(f"Numerical gradient: {numerical_grad_result}")
    
    # NOTE: this is a very loose check. The reason the results aren't very close is that JAX uses single precision, 
    # which is generally good for ML purposes. 
    # Check if the gradients are close
    assert np.allclose(analytical_grad, numerical_grad_result, atol=1e-1), \
        f"Gradients do not match for {term_name}: analytical {analytical_grad}, numerical {numerical_grad_result}"

# Run the tests for each term
test_grad_term(potential_term1, grad_term1, "Term 1")
test_grad_term(potential_term2, grad_term2, "Term 2")
test_grad_term(potential_term3, grad_term3, "Term 3")
test_grad_term(potential_term4, grad_term4, "Term 4")
test_grad_term(potential_term5, grad_term5, "Term 5")
test_grad_term(potential_term6, grad_term6, "Term 6")
test_grad_term(potential_term7, grad_term7, "Term 7")
test_grad_term(potential_term8, grad_term8, "Term 8")

# Test function for complete gradient
def test_gradV():
    # Test point
    q = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=float)
    
    # Analytical gradient
    analytical_grad = gradV(q)
    
    # Numerical gradient
    numerical_grad_result = numerical_grad(potential, q)
    
    # Compare each term and print the gradients
    terms = [potential_term1, potential_term2, potential_term3, potential_term4,
             potential_term5, potential_term6, potential_term7, potential_term8]
    
    for term in terms:
        numerical_term_grad = numerical_grad(wrap_term(term), q)
        print(f"Numerical gradient for {term.__name__}: {numerical_term_grad}")
    
    print(f"Analytical gradient: {analytical_grad}")
    print(f"Numerical gradient: {numerical_grad_result}")
    
    # Check if the gradients are close
    assert np.allclose(analytical_grad, numerical_grad_result, atol=1e-2), \
        f"Gradients do not match: analytical {analytical_grad}, numerical {numerical_grad_result}"

# Run the complete gradient test
test_gradV()
