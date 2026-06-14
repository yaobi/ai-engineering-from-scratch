import math

from optimizer_compare import (
    GradientDescent,
    rosenbrock,
    rosenbrock_gradient,
)


def safe_optimize(lr, start, steps=5000):
    optimizer = GradientDescent(lr=lr)
    params = list(start)

    for _ in range(steps):
        try:
            grads = rosenbrock_gradient(params)
            params = optimizer.step(params, grads)
            loss = rosenbrock(params)

            if not math.isfinite(loss):
                return params, float("inf"), "diverged"

        except OverflowError:
            return params, float("inf"), "diverged"

    return params, rosenbrock(params), "ok"


def main():
    print("=== Learning Rate Sweep for Gradient Descent ===")
    print()

    start = [-1.0, 1.0]

    learning_rates = [
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
    ]

    print("Start point:", start)
    print("Target minimum: x=1, y=1, loss=0")
    print()

    print(f"{'lr':>10} {'x':>12} {'y':>12} {'loss':>16} {'status':>12}")
    print("-" * 70)

    for lr in learning_rates:
        final, loss, status = safe_optimize(
            lr=lr,
            start=start,
            steps=5000,
        )

        if math.isfinite(loss):
            loss_text = f"{loss:.8f}"
        else:
            loss_text = "inf"

        print(
            f"{lr:10.4g} "
            f"{final[0]:12.6f} "
            f"{final[1]:12.6f} "
            f"{loss_text:>16} "
            f"{status:>12}"
        )

    print()
    print("Meaning:")
    print("A small learning rate is stable but slow.")
    print("A medium learning rate may converge faster.")
    print("A large learning rate can make the optimization diverge.")


if __name__ == "__main__":
    main()
