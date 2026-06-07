import math

from transformations_2d import rotation_2d, scaling_2d, shearing_2d, reflection_y


def det_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def main():
    print("=== Determinant as Area Scaling ===")

    R = rotation_2d(math.pi / 4)
    S = scaling_2d(2, 3)
    Sh = shearing_2d(1, 0)
    RefY = reflection_y()
    singular = [
        [1, 2],
        [2, 4],
    ]

    print(f"det(rotation 45°) = {det_2x2(R):.4f}")
    print("Meaning: rotation preserves area.")

    print()

    print(f"det(scale sx=2, sy=3) = {det_2x2(S):.1f}")
    print("Meaning: area becomes 2 * 3 = 6 times larger.")

    print()

    print(f"det(shear kx=1) = {det_2x2(Sh):.1f}")
    print("Meaning: shear changes shape, but preserves area.")

    print()

    print(f"det(reflect across y-axis) = {det_2x2(RefY):.1f}")
    print("Meaning: reflection preserves area, but flips orientation.")

    print()

    print(f"det(singular matrix) = {det_2x2(singular):.1f}")
    print("Meaning: space collapses to a line, so the matrix has no inverse.")


if __name__ == "__main__":
    main()
