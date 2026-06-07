import math


def rotation_2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [
        [c, -s],
        [s, c],
    ]


def scaling_2d(sx, sy):
    return [
        [sx, 0],
        [0, sy],
    ]


def shearing_2d(kx, ky):
    return [
        [1, kx],
        [ky, 1],
    ]


def reflection_x():
    return [
        [1, 0],
        [0, -1],
    ]


def reflection_y():
    return [
        [-1, 0],
        [0, 1],
    ]


def mat_vec_mul(matrix, vector):
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]


def print_point(name, point):
    print(f"{name}: ({point[0]:.4f}, {point[1]:.4f})")


def main():
    print("=== 2D Matrix Transformations ===")

    point = [1.0, 0.0]

    print()
    print("Original point:")
    print_point("point", point)

    print()
    print("1. Rotate point by 45 degrees")
    R = rotation_2d(math.pi / 4)
    rotated = mat_vec_mul(R, point)
    print_point("rotated", rotated)

    print()
    print("2. Scale point [1, 1] by sx=2, sy=3")
    S = scaling_2d(2, 3)
    scaled = mat_vec_mul(S, [1.0, 1.0])
    print_point("scaled", scaled)

    print()
    print("3. Shear point [1, 1] with kx=1, ky=0")
    Sh = shearing_2d(1, 0)
    sheared = mat_vec_mul(Sh, [1.0, 1.0])
    print_point("sheared", sheared)

    print()
    print("4. Reflect point [2, 1] across y-axis")
    RefY = reflection_y()
    reflected = mat_vec_mul(RefY, [2.0, 1.0])
    print_point("reflected", reflected)


if __name__ == "__main__":
    main()
