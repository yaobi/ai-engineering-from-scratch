import numpy as np


def rotation_2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s],
        [s, c],
    ])


def scaling_2d(sx, sy):
    return np.array([
        [sx, 0],
        [0, sy],
    ])


def shearing_2d(kx, ky):
    return np.array([
        [1, kx],
        [ky, 1],
    ])


def transform_points(matrix, points):
    return [matrix @ p for p in points]


def distance(a, b):
    return np.linalg.norm(a - b)


def print_points(title, points):
    print(title)
    for i, p in enumerate(points):
        print(f"  p{i}: ({p[0]:.4f}, {p[1]:.4f})")


def main():
    print("=== Square Transformation Exercise ===")

    square = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.0, 1.0]),
    ]

    print_points("Original square:", square)

    R = rotation_2d(np.pi / 4)
    S = scaling_2d(2, 0.5)
    Sh = shearing_2d(0.5, 0)

    rotated = transform_points(R, square)
    scaled = transform_points(S, square)
    sheared = transform_points(Sh, square)

    print()
    print_points("Rotated 45 degrees:", rotated)

    print()
    print_points("Scaled sx=2, sy=0.5:", scaled)

    print()
    print_points("Sheared kx=0.5:", sheared)

    print()
    print("=== Distance Check ===")

    original_edge = distance(square[0], square[1])
    rotated_edge = distance(rotated[0], rotated[1])

    print("Original edge length:", round(original_edge, 4))
    print("Rotated edge length:", round(rotated_edge, 4))
    print("Rotation preserves distance:", np.isclose(original_edge, rotated_edge))

    print()
    print("=== Determinant Check ===")

    print("det(rotation):", round(np.linalg.det(R), 4))
    print("det(scale):", round(np.linalg.det(S), 4))
    print("det(shear):", round(np.linalg.det(Sh), 4))

    print()
    print("Meaning:")
    print("rotation det=1: area unchanged")
    print("scale det=1.0 here because 2 * 0.5 = 1")
    print("shear det=1: area unchanged, shape becomes slanted")


if __name__ == "__main__":
    main()
