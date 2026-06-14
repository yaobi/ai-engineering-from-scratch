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
    print("=== Learning Rate Decay Demo ===")
    print()

    start = [-1.0, 1.0]

    fixed_optimizer = GradientDescent(lr=0.001)
    decay_optimizer = GradientDescentWithDecay(lr=0.001, decay=0.999)

    fixed_params, fixed_loss = optimize(
        fixed_optimizer,
        start=start,
        steps=5000,
    )

    decay_params, decay_loss = optimize(
        decay_optimizer,
        start=start,
        steps=5000,
    )

    print("Fixed learning rate:")
    print(
        f"  x={fixed_params[0]:.6f}, "
        f"y={fixed_params[1]:.6f}, "
        f"loss={fixed_loss:.8f}"
    )
    print()

    print("Learning rate decay:")
    print(
        f"  x={decay_params[0]:.6f}, "
        f"y={decay_params[1]:.6f}, "
        f"loss={decay_loss:.8f}"
    )
    print(f"  final lr={decay_optimizer.lr:.10f}")
    print()

    print("Meaning:")
    print("A fixed learning rate keeps the same step size during training.")
    print("Learning rate decay starts with a larger step and gradually makes it smaller.")
    print("This can help training become more stable near the minimum.")


if __name__ == "__main__":
    main()
