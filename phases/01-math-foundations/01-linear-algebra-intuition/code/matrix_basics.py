from vector_basics import Vector


class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(
                    self.rows[i][j] * other.components[j]
                    for j in range(self.shape[1])
                )
                for i in range(self.shape[0])
            ])

        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                value = sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                )
                row.append(value)
            rows.append(row)

        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"


def main():
    rotation_90 = Matrix([
        [0, -1],
        [1, 0],
    ])

    point = Vector([3, 1])
    rotated = rotation_90 @ point

    print("Original point:", point)
    print("After 90-degree rotation:", rotated)

    weights = Matrix([
        [0.1, -0.2, 0.3],
        [0.4, 0.5, -0.1],
    ])

    input_vector = Vector([1.0, 0.5, -0.3])
    output = weights @ input_vector

    print()
    print("Input vector:", input_vector)
    print("Weight matrix:", weights)
    print("Output vector:", output)
    print("This is what a neural network layer does: output = W @ x")


if __name__ == "__main__":
    main()
