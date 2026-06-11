def bayes(prior, likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    posterior = likelihood * prior / evidence
    return posterior


def main():
    print("=== Bayes Theorem Demo ===")
    print()

    prior = 0.0001
    likelihood = 0.99
    false_positive_rate = 0.01

    result = bayes(
        prior=prior,
        likelihood=likelihood,
        false_positive_rate=false_positive_rate,
    )

    print("Disease test example")
    print(f"Prior P(sick): {prior}")
    print(f"Likelihood P(positive | sick): {likelihood}")
    print(f"False positive rate P(positive | healthy): {false_positive_rate}")
    print()
    print(f"Posterior P(sick | positive) = {result:.4f}")
    print()

    print("Meaning:")
    print("Even if the test is 99% accurate,")
    print("when the disease is extremely rare,")
    print("a positive result does not mean the person is almost certainly sick.")
    print()
    print("This is called the base rate effect.")


if __name__ == "__main__":
    main()
