def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)


def f(x):
    return x ** 2


def main():
    print("=== Numerical Derivative Demo ===")
    print("Function: f(x) = x^2")
    print()

    for x in [-2, -1, 0, 1, 2]:
        numerical = numerical_derivative(f, x)
        analytical = 2 * x

        print(
            f"x={x:2d}  "
            f"f'(x) numerical={numerical:.6f}  "
            f"analytical={analytical:.1f}"
        )


if __name__ == "__main__":
    main()
