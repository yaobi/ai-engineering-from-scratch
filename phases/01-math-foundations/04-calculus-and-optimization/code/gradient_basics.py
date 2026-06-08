def numerical_gradient(f, point, h=1e-7):
    gradient = []

    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)

        point_plus[i] += h
        point_minus[i] -= h

        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)

    return gradient


def f_multi(point):
    x, y = point
    return x ** 2 + 3 * x * y + y ** 2


def main():
    print("=== Partial Derivatives and Gradient Demo ===")
    print("Function: f(x, y) = x^2 + 3xy + y^2")
    print()

    point = [1.0, 2.0]

    numerical = numerical_gradient(f_multi, point)

    x, y = point
    analytical = [
        2 * x + 3 * y,
        3 * x + 2 * y,
    ]

    print(f"Point: ({x}, {y})")
    print(f"Numerical gradient: {[round(g, 4) for g in numerical]}")
    print(f"Analytical gradient: {analytical}")

    print()
    print("Meaning:")
    print("df/dx = 2x + 3y")
    print("df/dy = 3x + 2y")
    print("At (1, 2):")
    print("df/dx = 2*1 + 3*2 = 8")
    print("df/dy = 3*1 + 2*2 = 7")
    print("gradient = [8, 7]")


if __name__ == "__main__":
    main()
