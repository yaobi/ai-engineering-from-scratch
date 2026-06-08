import random


def main():
    print("=== Linear Model Training from Scratch ===")
    print("Target function: y = 2x + 1")
    print()

    random.seed(42)

    w = random.gauss(0, 1)
    b = random.gauss(0, 1)
    lr = 0.01

    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [3.0, 5.0, 7.0, 9.0, 11.0]

    for epoch in range(200):
        total_loss = 0.0
        dw = 0.0
        db = 0.0

        for x, y in zip(xs, ys):
            pred = w * x + b
            error = pred - y

            total_loss += error ** 2

            dw += 2 * error * x
            db += 2 * error

        dw /= len(xs)
        db /= len(xs)
        total_loss /= len(xs)

        w -= lr * dw
        b -= lr * db

        if epoch % 40 == 0 or epoch == 199:
            print(
                f"epoch {epoch:3d}  "
                f"w={w:.4f}  "
                f"b={b:.4f}  "
                f"loss={total_loss:.6f}"
            )

    print()
    print(f"Learned: y = {w:.2f}x + {b:.2f}")
    print("Actual:  y = 2x + 1")


if __name__ == "__main__":
    main()
