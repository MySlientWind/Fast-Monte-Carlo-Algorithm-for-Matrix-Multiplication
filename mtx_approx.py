import numpy as np
from typing import Tuple



'''
@params:
    A: m x n matrix
    B: n x p matrix
    c: a positive integer
    ps: a list of probabilities (size n)
@return:
    Matrix C and R such that CR ≈ AB
    C: m x c matrix
    R: c x p matrix
'''
def matrix_multi_approx(A: np.ndarray, B: np.ndarray, c: int)-> Tuple[np.ndarray, np.ndarray]:
    assert A.shape[1] == B.shape[0], "Shape not match"
    n = A.shape[1]
    probs = compute_ps(A, B)
    its = np.random.choice(n, size=c, p=probs)
    C = A[:, its]/np.sqrt(c*probs[its]).reshape(1, -1)
    R = B[its, :]/np.sqrt(c*probs[its]).reshape(-1, 1)
    return C, R


'''
@params:
    A: m x n matrix
    B: n x p matrix
@return:
    a list of probabilities that minimize E(||AB-CR||^2) (Frobenius norm)
'''
def compute_ps(A: np.ndarray, B: np.ndarray)-> np.ndarray:
    assert A.shape[1] == B.shape[0], "Shape not match"
    C = np.sqrt(np.square(A).sum(axis=0) * np.square(B).sum(axis=1))
    return C/np.sum(C)

if __name__ == "__main__":

    # Generate random matrices
    A = np.random.uniform(size=(1000, 1000))
    B = np.random.uniform(size=(1000, 1000))

    # Do approximation
    C, R = matrix_multi_approx(A, B, 500)

    AB = A@B
    CR = C@R

    print("The median of |1-CR/AB|: ", np.median(np.abs(1-(CR/AB))))

