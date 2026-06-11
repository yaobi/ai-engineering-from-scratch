def beta_mean(a, b):
    return a / (a + b)


def beta_update(a, b, successes, failures):
    new_a = a + successes
    new_b = b + failures

    return new_a, new_b


def print_beta(name, a, b):
    print(f"{name}: Beta({a}, {b})")
    print(f"  mean = {beta_mean(a, b):.4f}")
    print(f"  confidence strength a+b = {a + b}")
    print()


def main():
    print("=== Beta Prior and Bayesian Updating ===")
    print()

    print("Example: estimating coin head probability")
    print()

    # Day 1: no data
    a, b = 1, 1
    print_beta("Day 1 prior", a, b)

    # Day 2: observe 7 heads, 3 tails
    a, b = beta_update(a, b, successes=7, failures=3)
    print_beta("Day 2 posterior after 7 heads and 3 tails", a, b)

    # Day 3: observe 5 heads, 5 tails
    a, b = beta_update(a, b, successes=5, failures=5)
    print_beta("Day 3 posterior after 5 more heads and 5 more tails", a, b)

    print("Batch update check:")
    batch_a, batch_b = beta_update(1, 1, successes=12, failures=8)
    print_beta("Update all data at once", batch_a, batch_b)

    print("Meaning:")
    print("Sequential updating and batch updating give the same final posterior.")
    print("Yesterday's posterior can become today's prior.")
    print("This is the core idea of online Bayesian learning.")


if __name__ == "__main__":
    main()
