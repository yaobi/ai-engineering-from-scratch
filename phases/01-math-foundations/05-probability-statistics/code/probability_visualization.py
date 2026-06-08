import math
import random
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)


def sample_categorical(probs, n=1):
    cumulative = []
    total = 0.0

    for p in probs:
        total += p
        cumulative.append(total)

    samples = []

    for _ in range(n):
        r = random.random()

        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break

    return samples


def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []

    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))

    return averages


def die_roll():
    return random.randint(1, 6)


def main():
    random.seed(42)

    os.makedirs("outputs", exist_ok=True)

    print("=== Probability Visualization ===")

    # 1. Normal PDF
    xs = [i / 100 for i in range(-400, 401)]
    ys = [normal_pdf(x, 0, 1) for x in xs]

    plt.figure()
    plt.plot(xs, ys)
    plt.title("Standard Normal PDF")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.savefig("outputs/probability_normal_pdf.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved: outputs/probability_normal_pdf.png")

    # 2. Categorical sampling
    probs = [0.6, 0.3, 0.1]
    samples = sample_categorical(probs, n=10000)

    counts = [samples.count(i) for i in range(len(probs))]
    proportions = [c / len(samples) for c in counts]

    plt.figure()
    plt.bar(["class 0", "class 1", "class 2"], proportions)
    plt.title("Categorical Sampling Proportions")
    plt.xlabel("class")
    plt.ylabel("proportion")
    plt.savefig("outputs/probability_categorical_sampling.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved: outputs/probability_categorical_sampling.png")

    # 3. CLT histogram
    averages = demonstrate_clt(
        dist_fn=die_roll,
        n_samples=30,
        n_averages=10000,
    )

    plt.figure()
    plt.hist(averages, bins=30, density=True)
    plt.title("Central Limit Theorem: Averages of Die Rolls")
    plt.xlabel("sample average")
    plt.ylabel("density")
    plt.savefig("outputs/probability_clt.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved: outputs/probability_clt.png")

    print()
    print("Meaning:")
    print("The normal PDF shows the bell curve.")
    print("The categorical sampling plot shows sampled proportions close to true probabilities.")
    print("The CLT plot shows that averages become concentrated and bell-shaped.")


if __name__ == "__main__":
    main()
