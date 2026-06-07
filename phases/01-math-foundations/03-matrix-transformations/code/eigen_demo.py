from transformations_2d import mat_vec_mul


def eigenvalues_2x2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]

    trace = a + d
    det = a * d - b * c

    discriminant = trace ** 2 - 4 * det

    if discriminant < 0:
        real = trace / 2
        imag = (-discriminant) ** 0.5 / 2
        return complex(real, imag), complex(real, -imag)

    sqrt_disc = discriminant ** 0.5

    lambda1 = (trace + sqrt_disc) / 2
    lambda2 = (trace - sqrt_disc) / 2

    return lambda1, lambda2


def eigenvector_2x2(matrix, eigenvalue):
    a, b = matrix[0]
    c, d = matrix[1]

    if abs(b) > 1e-10:
        v = [b, eigenvalue - a]
    elif abs(c) > 1e-10:
        v = [eigenvalue - d, c]
    else:
        if abs(a - eigenvalue) < 1e-10:
            v = [1, 0]
        else:
            v = [0, 1]

    magnitude = (v[0] ** 2 + v[1] ** 2) ** 0.5

    return [v[0] / magnitude, v[1] / magnitude]


def print_vector(name, vector):
    print(f"{name}: [{vector[0]:.4f}, {vector[1]:.4f}]")


def main():
    print("=== Eigenvalues and Eigenvectors Demo ===")

    A = [
        [2, 1],
        [1, 2],
    ]

    print("Matrix A:")
    for row in A:
        print(" ", row)

    eigenvalues = eigenvalues_2x2(A)

    print()
    print(f"Eigenvalues: {eigenvalues[0]:.4f}, {eigenvalues[1]:.4f}")

    print()
    for eigenvalue in eigenvalues:
        eigenvector = eigenvector_2x2(A, eigenvalue)

        transformed = mat_vec_mul(A, eigenvector)
        scaled = [
            eigenvalue * eigenvector[0],
            eigenvalue * eigenvector[1],
        ]

        print(f"lambda = {eigenvalue:.4f}")
        print_vector("eigenvector v", eigenvector)
        print_vector("A @ v", transformed)
        print_vector("lambda * v", scaled)
        print("A @ v equals lambda * v, so v keeps the same direction.")
        print()


if __name__ == "__main__":
    main()
