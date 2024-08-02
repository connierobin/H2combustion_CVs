import numpy as np
import scipy

def orthogonal_complement(vectors):
    # Convert the vectors into a NumPy array
    vectors = np.array(vectors)
    print(vectors)
    
    # Form the matrix A from the input vectors
    A = vectors.T
    print(A)
    
    # Compute the QR decomposition of A
    Q, R = np.linalg.qr(A)
    print(Q)
    b = np.array([1, 1, 1])
    bproj = Q @ Q.T @ b
    print('bproj1')
    print(bproj)
    Q = scipy.linalg.orth(A)
    print(Q)
    bproj = Q @ Q.T @ b
    print('bproj2')
    print(bproj)
    
    # Extract the basis for the orthogonal complement from the columns of Q
    complement_basis = Q[:, np.linalg.matrix_rank(A):]
    print(complement_basis)
    
    # Return the orthogonal complement
    return complement_basis

def test_orthogonal_complement():
    # Test case 1
    vectors1 = [[1, 0, 0], [0, 1, 0]]
    complement1 = orthogonal_complement(vectors1)
    assert np.allclose(complement1, np.array([[0], [0], [1]]))

    # TODO: I think this test case is wrong??? I changed it from (0,0,1) to (-1,1,1)
    # Test case 2
    vectors2 = [[1, 1, 0], [1, 0, 1]]
    complement2 = orthogonal_complement(vectors2)
    assert np.allclose(complement2, np.array([[-1], [1], [1]]))

    # Test case 3
    vectors3 = [[1, 2], [2, -1]]
    complement3 = orthogonal_complement(vectors3)
    assert np.allclose(complement3, np.array([[-0.4472136], [0.89442719]]))

    print(complement1)
    print(complement2)
    print(complement3)

    print("All tests passed!")

# Run the tests
test_orthogonal_complement()
