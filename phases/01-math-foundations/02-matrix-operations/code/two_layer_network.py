import random

from matrix_operations import Matrix, relu_matrix


def random_matrix(rows, cols):
    return Matrix([
        [random.uniform(-1, 1) for _ in range(cols)]
        for _ in range(rows)
    ])


def main():
    random.seed(42)

    print("=== Two-Layer Neural Network with Matrix Class ===")

    # input: 3 features, one sample
    x = Matrix([
        [0.5],
        [0.8],
        [0.2],
    ])

    # hidden layer: 3 inputs -> 4 hidden units
    W1 = random_matrix(4, 3)
    b1 = Matrix([
        [0.1],
        [0.1],
        [0.1],
        [0.1],
    ])

    # output layer: 4 hidden units -> 2 outputs
    W2 = random_matrix(2, 4)
    b2 = Matrix([
        [0.1],
        [0.1],
    ])

    hidden_pre = W1.matmul(x) + b1
    hidden = relu_matrix(hidden_pre)

    output = W2.matmul(hidden) + b2

    print("Input shape:", x.shape)
    print("W1 shape:", W1.shape)
    print("Hidden shape:", hidden.shape)
    print("W2 shape:", W2.shape)
    print("Output shape:", output.shape)
    print("Output:", output.data)

    print()
    print("This is a two-layer network:")
    print("hidden = relu(W1 @ x + b1)")
    print("output = W2 @ hidden + b2")


if __name__ == "__main__":
    main()
