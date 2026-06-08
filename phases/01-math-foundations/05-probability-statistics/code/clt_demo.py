import random


def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []

    for _ in range(n_averages):
        samples = [
            dist_fn()
            for _ in range(n_samples)
        ]

        averages.append(sum(samples) / len(samples))

    return averages


def summarize(values):
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std = variance ** 0.5

    return mean, std, min(values), max(values)


def die_roll():
    return random.randint(1, 6)


def main():
    random.seed(42)

    print("=== Central Limit Theorem Demo ===")
    print("Source distribution: fair die roll")
    print("Original outcomes: 1, 2, 3, 4, 5, 6")
    print()

    for n_samples in [1, 2, 5, 10, 30, 100]:
        averages = demonstrate_clt(
            dist_fn=die_roll,
            n_samples=n_samples,
            n_averages=10000,
        )

        mean, std, min_value, max_value = summarize(averages)

        print(f"Sample size per average: {n_samples}")
        print(f"  mean of averages: {mean:.4f}")
        print(f"  std of averages:  {std:.4f}")
        print(f"  min average:      {min_value:.4f}")
        print(f"  max average:      {max_value:.4f}")
        print(f"  first 10 averages: {[round(x, 2) for x in averages[:10]]}")
        print()

    print("Meaning:")
    print("When n_samples is small, the averages are more spread out.")
    print("When n_samples becomes larger, the averages become more stable.")
    print("The distribution of averages becomes closer to a normal distribution.")


if __name__ == "__main__":
    main()
