import math

from transformations_2d import rotation_2d, scaling_2d, mat_vec_mul


def mat_mul(a, b):
    rows_a = len(a)
    cols_a = len(a[0])
    cols_b = len(b[0])

    return [
        [
            sum(a[i][k] * b[k][j] for k in range(cols_a))
            for j in range(cols_b)
        ]
        for i in range(rows_a)
    ]


def print_matrix(name, matrix):
    print(name)
    for row in matrix:
        print("  ", [round(x, 4) for x in row])


def print_point(name, point):
    print(f"{name}: ({point[0]:.4f}, {point[1]:.4f})")


def main():
    print("=== Composition of Transformations ===")

    point = [1.0, 0.0]

    R = rotation_2d(math.pi / 2)  # 90 degrees
    S = scaling_2d(2, 0.5)

    print()
    print("Original point:")
    print_point("point", point)

    print()
    print("Rotation matrix R:")
    print_matrix("R =", R)

    print()
    print("Scaling matrix S:")
    print_matrix("S =", S)

    print()
    print("Case 1: rotate first, then scale")
    rotate_then_scale_matrix = mat_mul(S, R)
    result1 = mat_vec_mul(rotate_then_scale_matrix, point)

    print_matrix("S @ R =", rotate_then_scale_matrix)
    print_point("result1", result1)

    print()
    print("Case 2: scale first, then rotate")
    scale_then_rotate_matrix = mat_mul(R, S)
    result2 = mat_vec_mul(scale_then_rotate_matrix, point)

    print_matrix("R @ S =", scale_then_rotate_matrix)
    print_point("result2", result2)

    print()
    print("Are they the same?", result1 == result2)
    print("Conclusion: transformation order matters.")


if __name__ == "__main__":
    main()
