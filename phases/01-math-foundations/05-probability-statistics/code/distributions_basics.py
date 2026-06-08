import math


def factorial(n):
    result = 1

    for i in range(2, n + 1):
        result *= i

    return result


def bernoulli_pmf(k, p):
    if k == 1:
        return p
    elif k == 0:
        return 1 - p
    else:
        return 0


def categorical_pmf(k, probs):
    return probs[k]


def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)


def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0


def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)


def main():
    print("=== PMF and PDF Demo ===")
    print()

    print("1. Bernoulli PMF")
    print("Example: coin flip, p(head)=0.7")
    print("P(X=1) =", bernoulli_pmf(1, 0.7))
    print("P(X=0) =", bernoulli_pmf(0, 0.7))
    print()

    print("2. Categorical PMF")
    print("Example: class probabilities [cat, dog, rabbit] = [0.6, 0.3, 0.1]")
    probs = [0.6, 0.3, 0.1]
    print("P(class=0 cat) =", categorical_pmf(0, probs))
    print("P(class=1 dog) =", categorical_pmf(1, probs))
    print("P(class=2 rabbit) =", categorical_pmf(2, probs))
    print()

    print("3. Poisson PMF")
    print("Example: average 3 events per hour")
    lam = 3
    for k in range(6):
        print(f"P(X={k}) = {poisson_pmf(k, lam):.4f}")
    print()

    print("4. Uniform PDF")
    print("Example: uniform distribution from 0 to 10")
    print("pdf(5) =", uniform_pdf(5, 0, 10))
    print("pdf(11) =", uniform_pdf(11, 0, 10))
    print()

    print("5. Normal PDF")
    print("Example: standard normal distribution, mu=0, sigma=1")
    for x in [-2, -1, 0, 1, 2]:
        print(f"pdf({x}) = {normal_pdf(x, 0, 1):.4f}")

    print()
    print("Meaning:")
    print("PMF gives probability for discrete outcomes.")
    print("PDF gives density for continuous values.")
    print("For PDF, probability comes from area under the curve, not a single point.")


if __name__ == "__main__":
    main()
