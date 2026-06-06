import numpy as np


def main():
    print("=== Vector Operations with NumPy ===")

    a = np.array([1, 2, 3], dtype=float)
    b = np.array([4, 5, 6], dtype=float)

    print("a =", a)
    print("b =", b)
    print("a + b =", a + b)
    print("a · b =", np.dot(a, b))
    print("|a| =", round(np.linalg.norm(a), 4))

    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print("cosine similarity =", round(cosine, 4))

    print()
    print("=== Matrix Multiplication ===")

    W = np.array([
        [0.1, -0.2, 0.3],
        [0.4, 0.5, -0.1],
    ])

    x = np.array([1.0, 0.5, -0.3])

    print("W =", W)
    print("x =", x)
    print("W @ x =", W @ x)

    print()
    print("=== Rank ===")

    A = np.array([
        [1, 2],
        [2, 4],
    ])

    print("A =", A)
    print("rank(A) =", np.linalg.matrix_rank(A))
    print("Because row 2 = 2 * row 1, this matrix has rank 1.")

    print()
    print("=== Projection ===")

    p = np.array([3, 4], dtype=float)
    direction = np.array([1, 0], dtype=float)

    projection = (np.dot(p, direction) / np.dot(direction, direction)) * direction

    print("p =", p)
    print("direction =", direction)
    print("projection =", projection)

    print()
    print("=== QR Decomposition ===")

    random_matrix = np.random.randn(3, 3)
    Q, R = np.linalg.qr(random_matrix)

    print("Q @ Q.T close to identity:", np.allclose(Q @ Q.T, np.eye(3)))
    print("R is upper triangular:", np.allclose(R, np.triu(R)))


if __name__ == "__main__":
    main()
