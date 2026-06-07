import numpy as np


def main():
    print("=== NumPy Matrix Operations ===")

    A = np.array([[1, 2], [3, 4]], dtype=float)
    B = np.array([[5, 6], [7, 8]], dtype=float)

    print("A + B =")
    print(A + B)

    print()
    print("A * B element-wise =")
    print(A * B)

    print()
    print("A @ B matrix multiply =")
    print(A @ B)

    print()
    print("A transpose =")
    print(A.T)

    print()
    print("det(A) =", np.linalg.det(A))

    print()
    print("A inverse =")
    print(np.linalg.inv(A))

    print()
    print("A @ inv(A) =")
    print(A @ np.linalg.inv(A))

    print()
    print("Identity matrix =")
    print(np.eye(2))

    print()
    print("=== Neural Network Layer with NumPy ===")

    inputs = np.array([[0.5], [0.8], [0.2]])
    weights = np.random.uniform(-1, 1, size=(2, 3))
    bias = np.array([[0.1], [0.1]])

    pre_activation = weights @ inputs + bias
    output = np.maximum(0, pre_activation)

    print("Input shape:", inputs.shape)
    print("Weight shape:", weights.shape)
    print("Bias shape:", bias.shape)
    print("Output shape:", output.shape)
    print("Output:")
    print(output)

    print()
    print("=== Broadcasting Demo ===")

    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    bias_vector = np.array([10, 20, 30])

    print("matrix =")
    print(matrix)
    print("bias_vector =", bias_vector)
    print("matrix + bias_vector =")
    print(matrix + bias_vector)


if __name__ == "__main__":
    main()
