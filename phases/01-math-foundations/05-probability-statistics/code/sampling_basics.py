import math
import random


def sample_bernoulli(p, n=1):
    return [
        1 if random.random() < p else 0
        for _ in range(n)
    ]


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


def sample_normal_box_muller(mu, sigma, n=1):
    samples = []

    for _ in range(n):
        u1 = random.random()
        u2 = random.random()

        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

        samples.append(mu + sigma * z)

    return samples


def count_values(samples):
    counts = {}

    for s in samples:
        counts[s] = counts.get(s, 0) + 1

    return counts


def main():
    random.seed(42)

    print("=== Sampling Demo ===")
    print()

    print("1. Bernoulli Sampling")
    bernoulli_samples = sample_bernoulli(p=0.7, n=20)
    print("20 samples:", bernoulli_samples)

    large_bernoulli = sample_bernoulli(p=0.7, n=10000)
    proportion_ones = sum(large_bernoulli) / len(large_bernoulli)
    print("Proportion of 1 in 10000 samples:", round(proportion_ones, 4))
    print()

    print("2. Categorical Sampling")
    probs = [0.6, 0.3, 0.1]
    categorical_samples = sample_categorical(probs, n=20)
    print("20 samples:", categorical_samples)

    large_categorical = sample_categorical(probs, n=10000)
    counts = count_values(large_categorical)

    print("Counts in 10000 samples:")
    for k in sorted(counts):
        print(f"  class {k}: {counts[k]}  proportion={counts[k] / 10000:.4f}")
    print()

    print("3. Normal Sampling with Box-Muller")
    normal_samples = sample_normal_box_muller(mu=0, sigma=1, n=10)
    print("10 samples:", [round(x, 4) for x in normal_samples])

    large_normal = sample_normal_box_muller(mu=0, sigma=1, n=10000)
    mean = sum(large_normal) / len(large_normal)
    variance = sum((x - mean) ** 2 for x in large_normal) / len(large_normal)
    std = variance ** 0.5

    print("Mean of 10000 normal samples:", round(mean, 4))
    print("Std of 10000 normal samples:", round(std, 4))

    print()
    print("Meaning:")
    print("Sampling means generating random values according to a distribution.")
    print("With many samples, observed proportions become close to theoretical probabilities.")


if __name__ == "__main__":
    main()
