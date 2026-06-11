import random


def beta_mean(a, b):
    return a / (a + b)


def beta_update(prior_a, prior_b, successes, failures):
    return prior_a + successes, prior_b + failures


def estimate_probability_b_better(a_a, b_a, a_b, b_b, n_samples=100000):
    b_better_count = 0

    for _ in range(n_samples):
        sample_a = random.betavariate(a_a, b_a)
        sample_b = random.betavariate(a_b, b_b)

        if sample_b > sample_a:
            b_better_count += 1

    return b_better_count / n_samples


def main():
    random.seed(42)

    print("=== Bayesian A/B Testing Demo ===")
    print()

    # Prior: Beta(1, 1) for both variants
    prior_a = 1
    prior_b = 1

    # Observed data
    views_A = 1000
    clicks_A = 50
    failures_A = views_A - clicks_A

    views_B = 1000
    clicks_B = 65
    failures_B = views_B - clicks_B

    # Posterior update
    post_A_a, post_A_b = beta_update(
        prior_a,
        prior_b,
        successes=clicks_A,
        failures=failures_A,
    )

    post_B_a, post_B_b = beta_update(
        prior_a,
        prior_b,
        successes=clicks_B,
        failures=failures_B,
    )

    print("Variant A:")
    print(f"  clicks/views = {clicks_A}/{views_A}")
    print(f"  raw conversion rate = {clicks_A / views_A:.4f}")
    print(f"  posterior = Beta({post_A_a}, {post_A_b})")
    print(f"  posterior mean = {beta_mean(post_A_a, post_A_b):.4f}")
    print()

    print("Variant B:")
    print(f"  clicks/views = {clicks_B}/{views_B}")
    print(f"  raw conversion rate = {clicks_B / views_B:.4f}")
    print(f"  posterior = Beta({post_B_a}, {post_B_b})")
    print(f"  posterior mean = {beta_mean(post_B_a, post_B_b):.4f}")
    print()

    prob_b_better = estimate_probability_b_better(
        post_A_a,
        post_A_b,
        post_B_a,
        post_B_b,
        n_samples=100000,
    )

    print(f"Estimated P(B > A) = {prob_b_better:.4f}")
    print()

    if prob_b_better > 0.95:
        decision = "Ship B"
    elif prob_b_better < 0.05:
        decision = "Ship A"
    else:
        decision = "Keep collecting data"

    print("Decision:", decision)
    print()

    print("Meaning:")
    print("Bayesian A/B testing gives a direct probability statement.")
    print("Instead of only saying B has a higher observed rate,")
    print("we estimate the probability that B is truly better than A.")


if __name__ == "__main__":
    main()
