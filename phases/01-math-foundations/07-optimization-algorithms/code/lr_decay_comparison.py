from optimizer_compare import (
    GradientDescent,
    rosenbrock,
    rosenbrock_gradient,
)


class GradientDescentWithDecay:
    def __init__(self, lr=0.001, decay=0.999):
        self.initial_lr = lr
        self.lr = lr
        self.decay = decay
        self.step_count = 0

    def step(self, params, grads):
        self.step_count += 1
        self.lr = self.initial_lr * (self.decay ** self.step_count)

        return [
            p - self.lr * g
            for p, g in zip(params, grads)
        ]


def optimize(optimizer, start, steps=5000):
    params = list(start)

    for _ in range(steps):
        grads = rosenbrock_gradient(params)
        params = optimizer.step(params, grads)

    loss = rosenbrock(params)
    return params, loss


def main():
    print("=== Learning Rate Decay Comparison ===")
    print()

    start = [-1.0, 1.0]
    steps = 5000

    experiments = [
        ("fixed", GradientDescent(lr=0.001)),
        ("decay=0.999", GradientDescentWithDecay(lr=0.001, decay=0.999)),
        ("decay=0.9999", GradientDescentWithDecay(lr=0.001, decay=0.9999)),
        ("decay=0.99999", GradientDescentWithDecay(lr=0.001, decay=0.99999)),
    ]

    print(f"{'setting':>15} {'x':>12} {'y':>12} {'loss':>16} {'final_lr':>14}")
    print("-" * 75)

    for name, optimizer in experiments:
        params, loss = optimize(
            optimizer=optimizer,
            start=start,
            steps=steps,
        )

        final_lr = getattr(optimizer, "lr", 0.001)

        print(
            f"{name:>15} "
            f"{params[0]:12.6f} "
            f"{params[1]:12.6f} "
            f"{loss:16.8f} "
            f"{final_lr:14.10f}"
        )

    print()
    print("Meaning:")
    print("If decay is too strong, the learning rate becomes too small too early.")
    print("If decay is gentle, it may keep enough movement while becoming more stable later.")
    print("Learning rate schedules must be tuned, just like the initial learning rate.")


if __name__ == "__main__":
    main()
