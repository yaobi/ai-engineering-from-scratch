def hessian_2d(f, x, y, h=1e-5):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)

    fxy = (
        f(x + h, y + h)
        - f(x + h, y - h)
        - f(x - h, y + h)
        + f(x - h, y - h)
    ) / (4 * h ** 2)

    return [
        [fxx, fxy],
        [fxy, fyy],
    ]


def saddle(x, y):
    return x ** 2 - y ** 2


def bowl(x, y):
    return x ** 2 + y ** 2


def print_matrix(name, matrix):
    print(name)
    for row in matrix:
        print(" ", [round(x, 4) for x in row])


def main():
    print("=== Hessian Demo ===")

    H_saddle = hessian_2d(saddle, 0.0, 0.0)
    H_bowl = hessian_2d(bowl, 0.0, 0.0)

    print_matrix("Saddle Hessian:", H_saddle)
    print("Meaning: one direction curves up, one direction curves down.")
    print()

    print_matrix("Bowl Hessian:", H_bowl)
    print("Meaning: both directions curve up, so this is a minimum.")


if __name__ == "__main__":
    main()
