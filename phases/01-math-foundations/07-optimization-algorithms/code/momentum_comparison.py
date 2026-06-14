from optimizer_compare import (
    SGDMomentum,
    rosenbrock,
    rosenbrock_gradient,
)


def optimize_with_check(optimizer, start, steps=5000):
    params = list(start)
    losses = []

    for step in range(steps):
        grads = rosenbrock_gradient(params)
        params = optimizer.step(params, grads)
        loss = rosenbrock(params)
        losses.append(loss)

        if loss > 1e20:
            return params, loss, step + 1, "diverged"

    return params, losses[-1], steps, "ok"


def main():
    print("=== Momentum Comparison ===")
    print()

    start = [-1.0, 1.0]
    momentum_values = [0.0, 0.5, 0.9, 0.99]

    print("Start point:", start)
    print("Target minimum: x=1, y=1, loss=0")
    print()

    print(f"{'momentum':>10} {'x':>12} {'y':>12} {'loss':>16} {'steps':>8} {'status':>12}")
    print("-" * 80)

    for momentum in momentum_values:
        optimizer = SGDMomentum(
            lr=0.0001,
            momentum=momentum,
        )

        final, loss, steps, status = optimize_with_check(
            optimizer=optimizer,
            start=start,
            steps=5000,
        )

        print(
            f"{momentum:10.2f} "
            f"{final[0]:12.6f} "
            f"{final[1]:12.6f} "
            f"{loss:16.8f} "
            f"{steps:8d} "
            f"{status:>12}"
        )

    print()
    print("Meaning:")
    print("momentum=0.0 is almost vanilla gradient descent.")
    print("moderate momentum can accelerate convergence.")
    print("very large momentum may overshoot or become unstable.")


if __name__ == "__main__":
    main()
