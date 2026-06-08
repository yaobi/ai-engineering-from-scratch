import math


def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)


def main():
    print("=== Numerical vs Analytical Derivatives ===")

    test_functions = [
        ("x^2", lambda x: x ** 2, lambda x: 2 * x),
        ("x^3", lambda x: x ** 3, lambda x: 3 * x ** 2),
        ("sin(x)", lambda x: math.sin(x), lambda x: math.cos(x)),
        ("e^x", lambda x: math.exp(x), lambda x: math.exp(x)),
        ("1/x", lambda x: 1 / x, lambda x: -1 / x ** 2),
    ]

    x = 2.0

    print(f"{'Function':<12} {'Numerical':>12} {'Analytical':>12} {'Error':>12}")
    print("-" * 50)

    for name, f, df in test_functions:
        numerical = numerical_derivative(f, x)
        analytical = df(x)
        error = abs(numerical - analytical)

        print(
            f"{name:<12} "
            f"{numerical:12.6f} "
            f"{analytical:12.6f} "
            f"{error:12.2e}"
        )


if __name__ == "__main__":
    main()
