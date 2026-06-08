def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))


def variance(values, probabilities):
    mu = expected_value(values, probabilities)

    return sum(
        p * (v - mu) ** 2
        for v, p in zip(values, probabilities)
    )


def main():
    print("=== Expected Value and Variance Demo ===")
    print()

    die_values = [1, 2, 3, 4, 5, 6]
    die_probs = [1 / 6] * 6

    mu = expected_value(die_values, die_probs)
    var = variance(die_values, die_probs)
    sd = var ** 0.5

    print("Example: fair die")
    print("Values:", die_values)
    print("Probabilities:", [round(p, 4) for p in die_probs])
    print()

    print(f"E[X] = {mu:.4f}")
    print(f"Var(X) = {var:.4f}")
    print(f"SD(X) = {sd:.4f}")
    print()

    print("Meaning:")
    print("Expected value is the long-run average.")
    print("Variance measures how spread out the outcomes are.")
    print("Standard deviation is the square root of variance.")

    print()
    print("=== Loaded Die Example ===")

    loaded_probs = [0.05, 0.05, 0.10, 0.10, 0.20, 0.50]

    loaded_mu = expected_value(die_values, loaded_probs)
    loaded_var = variance(die_values, loaded_probs)
    loaded_sd = loaded_var ** 0.5

    print("Loaded probabilities:", loaded_probs)
    print(f"E[X] = {loaded_mu:.4f}")
    print(f"Var(X) = {loaded_var:.4f}")
    print(f"SD(X) = {loaded_sd:.4f}")
    print()

    print("Meaning:")
    print("The loaded die has a higher expected value because 6 is more likely.")


if __name__ == "__main__":
    main()
