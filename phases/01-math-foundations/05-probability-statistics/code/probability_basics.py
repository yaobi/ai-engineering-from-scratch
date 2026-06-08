def factorial(n):
    result = 1

    for i in range(2, n + 1):
        result *= i

    return result


def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))


def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b


def main():
    print("=== Probability Basics ===")
    print()

    print("Factorial:")
    print("5! =", factorial(5))
    print()

    print("Combinations:")
    print("C(5, 2) =", combinations(5, 2))
    print("Meaning: choose 2 objects from 5 objects.")
    print()

    print("Conditional Probability:")
    print("Formula: P(A | B) = P(A and B) / P(B)")
    print()

    # A standard deck has 52 cards.
    # Face cards are J, Q, K.
    # There are 12 face cards in total.
    # There are 4 kings.
    p_king_and_face = 4 / 52
    p_face = 12 / 52

    p_king_given_face = conditional_probability(p_king_and_face, p_face)

    print("Question: P(King | Face card)")
    print("P(King and Face card) = 4 / 52")
    print("P(Face card) = 12 / 52")
    print(f"P(King | Face card) = {p_king_given_face:.4f}")
    print()

    print("Meaning:")
    print("If we already know the card is a face card,")
    print("then only J, Q, K are possible.")
    print("Among 12 face cards, 4 are kings.")
    print("So the answer is 4 / 12 = 1 / 3.")


if __name__ == "__main__":
    main()
