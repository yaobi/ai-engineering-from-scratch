import numpy as np


def main():
    print("=== NumPy Eigen Demo ===")

    A = np.array([
        [2, 1],
        [1, 2],
    ], dtype=float)

    eigenvalues, eigenvectors = np.linalg.eig(A)

    print("Matrix A:")
    print(A)

    print()
    print("Eigenvalues:")
    print(eigenvalues)

    print()
    print("Eigenvectors are columns of this matrix:")
    print(eigenvectors)

    print()
    print("Check A @ v = lambda * v")

    for i in range(len(eigenvalues)):
        lam = eigenvalues[i]
        v = eigenvectors[:, i]

        print()
        print(f"Eigenvalue {i + 1}: {lam:.4f}")
        print("v =", np.round(v, 4))
        print("A @ v =", np.round(A @ v, 4))
        print("lambda * v =", np.round(lam * v, 4))

    print()
    print("=== Eigendecomposition ===")

    V = eigenvectors
    D = np.diag(eigenvalues)
    V_inv = np.linalg.inv(V)

    reconstructed = V @ D @ V_inv

    print("V:")
    print(np.round(V, 4))

    print()
    print("D:")
    print(np.round(D, 4))

    print()
    print("V @ D @ V^-1:")
    print(np.round(reconstructed, 4))

    print()
    print("Original A:")
    print(A)

    print()
    print("Reconstruction close to A:", np.allclose(A, reconstructed))


if __name__ == "__main__":
    main()
