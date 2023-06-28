import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from mtx_approx import *


if __name__ == "__main__":
    f1 = []
    f2 = []

    for _ in range(100):
        # Generate random matrices
        A = np.random.uniform(size=(1000, 1000))
        B = np.random.uniform(size=(1000, 1000))

        # Do approximation
        C, R = matrix_multi_approx(A, B, 500, replace=True)
        C2, R2 = matrix_multi_approx(A, B, 500, replace=False)

        f1.append(np.linalg.norm(A@B-C@R, ord='fro'))
        f2.append(np.linalg.norm(A@B-C2@R2, ord='fro'))

    # replace=False is significantly better than replace=True
    print("replace=True: ", np.mean(f1), "+/-", np.std(f1))
    print("replace=False: ", np.mean(f2), "+/-", np.std(f2))