import numpy as np


def rotation_2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s],
        [s, c],
    ])


def rotation_3d_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])


def rotation_3d_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])


def main():
    print("=== NumPy 2D Transformation ===")

    theta = np.pi / 4
    R = rotation_2d(theta)

    point = np.array([1.0, 0.0])
    rotated = R @ point

    print("point:", point)
    print("Rotate 45 degrees:", np.round(rotated, 4))

    print()
    print("=== Composition ===")

    S = np.diag([2.0, 3.0])
    composed = S @ R

    print("Scale after rotate:")
    print(np.round(composed @ point, 4))

    print()
    print("=== Eigenvalues and Eigenvectors ===")

    A = np.array([
        [2, 1],
        [1, 2],
    ], dtype=float)

    eigenvalues, eigenvectors = np.linalg.eig(A)

    print("A:")
    print(A)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:")
    print(np.round(eigenvectors, 4))

    for i in range(len(eigenvalues)):
        lam = eigenvalues[i]
        v = eigenvectors[:, i]

        print()
        print(f"lambda {i + 1}:", round(lam, 4))
        print("A @ v:", np.round(A @ v, 4))
        print("lambda * v:", np.round(lam * v, 4))

    print()
    print("=== Eigendecomposition ===")

    B = np.array([
        [3, 1],
        [0, 2],
    ], dtype=float)

    vals, vecs = np.linalg.eig(B)
    D = np.diag(vals)
    V = vecs

    reconstructed = V @ D @ np.linalg.inv(V)

    print("Original B:")
    print(B)
    print("Reconstructed B:")
    print(np.round(reconstructed, 4))
    print("Close:", np.allclose(B, reconstructed))

    print()
    print("=== 3D Rotation ===")

    point_3d = np.array([1.0, 0.0, 0.0])

    rotated_z = rotation_3d_z(np.pi / 2) @ point_3d
    rotated_x = rotation_3d_x(np.pi / 2) @ point_3d

    print("3D point:", point_3d)
    print("Rotate 90 around z-axis:", np.round(rotated_z, 4))
    print("Rotate 90 around x-axis:", np.round(rotated_x, 4))


if __name__ == "__main__":
    main()
