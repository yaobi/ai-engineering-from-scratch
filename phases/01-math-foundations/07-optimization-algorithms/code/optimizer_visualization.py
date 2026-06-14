import os
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from optimizer_compare import (
    GradientDescent,
    SGDMomentum,
    Adam,
    rosenbrock,
    rosenbrock_gradient,
    optimize,
)


def main():
    print("=== Optimizer Path Visualization ===")

    os.makedirs("outputs", exist_ok=True)

    start = [-1.0, 1.0]

    gd_history = optimize(
        GradientDescent(lr=0.0005),
        rosenbrock,
        rosenbrock_gradient,
        start,
        steps=5000,
    )

    momentum_history = optimize(
        SGDMomentum(lr=0.0001, momentum=0.9),
        rosenbrock,
        rosenbrock_gradient,
        start,
        steps=5000,
    )

    adam_history = optimize(
        Adam(lr=0.01),
        rosenbrock,
        rosenbrock_gradient,
        start,
        steps=5000,
    )

    histories = {
        "GD": gd_history,
        "Momentum": momentum_history,
        "Adam": adam_history,
    }

    # Draw contour background
    xs = [i / 100 for i in range(-150, 151)]
    ys = [i / 100 for i in range(-50, 201)]

    X = []
    Y = []
    Z = []

    for y in ys:
        row_x = []
        row_y = []
        row_z = []

        for x in xs:
            row_x.append(x)
            row_y.append(y)

            z = rosenbrock([x, y])
            # log scale makes the narrow valley easier to see
            row_z.append(math.log10(z + 1e-6))

        X.append(row_x)
        Y.append(row_y)
        Z.append(row_z)

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=30)

    for name, history in histories.items():
        # Plot every 50th point to keep the figure clean
        sampled = history[::50]
        path_x = [p[0] for p in sampled]
        path_y = [p[1] for p in sampled]

        plt.plot(path_x, path_y, marker="o", markersize=2, label=name)

    plt.scatter([1.0], [1.0], marker="*", s=150, label="Minimum (1,1)")
    plt.title("Optimizer Paths on Rosenbrock Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("outputs/optimizer_paths.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved: outputs/optimizer_paths.png")
    print()
    print("Meaning:")
    print("GD moves slowly along the valley.")
    print("Momentum follows a smoother path.")
    print("Adam reaches the minimum more directly.")


if __name__ == "__main__":
    main()
