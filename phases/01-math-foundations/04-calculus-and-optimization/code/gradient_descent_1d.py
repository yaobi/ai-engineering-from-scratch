def main():
    print("=== Gradient Descent on f(x) = x^2 ===")

    x = 5.0
    lr = 0.1

    for step in range(20):
        grad = 2 * x
        x = x - lr * grad
        loss = x ** 2

        print(f"step {step:2d}  x={x:8.4f}  f(x)={loss:10.6f}")


if __name__ == "__main__":
    main()
