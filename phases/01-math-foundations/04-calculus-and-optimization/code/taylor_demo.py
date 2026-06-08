import math


def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)

    if order >= 1:
        result += f_prime(x0) * h

    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2

    return result


def main():
    print("=== Taylor Approximation Demo ===")
    print("Approximate sin(h) near x0 = 0")
    print()

    x0 = 0.0

    for h in [0.1, 0.5, 1.0, 2.0]:
        true_value = math.sin(h)

        order1 = taylor_approx(
            math.sin,
            math.cos,
            lambda x: -math.sin(x),
            x0,
            h,
            order=1,
        )

        order2 = taylor_approx(
            math.sin,
            math.cos,
            lambda x: -math.sin(x),
            x0,
            h,
            order=2,
        )

        print(
            f"h={h:.1f}  "
            f"sin(h)={true_value:.4f}  "
            f"order1={order1:.4f}  "
            f"order2={order2:.4f}"
        )

    print()
    print("Meaning:")
    print("Taylor approximation works well near x0.")
    print("When h is large, the approximation becomes worse.")
    print("This is why gradient descent needs a reasonable learning rate.")


if __name__ == "__main__":
    main()
