import math


def softmax(logits):
    max_logit = max(logits)

    shifted = [
        z - max_logit
        for z in logits
    ]

    exps = [
        math.exp(z)
        for z in shifted
    ]

    total = sum(exps)

    return [
        e / total
        for e in exps
    ]


def log_softmax(logits):
    max_logit = max(logits)

    shifted = [
        z - max_logit
        for z in logits
    ]

    log_sum_exp = max_logit + math.log(
        sum(math.exp(z) for z in shifted)
    )

    return [
        z - log_sum_exp
        for z in logits
    ]


def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)

    return -log_probs[target_index]


def main():
    print("=== Softmax and Cross-Entropy Demo ===")
    print()

    logits = [2.0, 1.0, 0.1]
    target_index = 0

    probs = softmax(logits)
    log_probs = log_softmax(logits)
    loss = cross_entropy_loss(logits, target_index)

    print("Logits:", logits)
    print("Softmax probabilities:", [round(p, 4) for p in probs])
    print("Sum of probabilities:", round(sum(probs), 4))
    print()

    print("Log probabilities:", [round(lp, 4) for lp in log_probs])
    print()

    print("Target class index:", target_index)
    print("Target class probability:", round(probs[target_index], 4))
    print("Cross-entropy loss:", round(loss, 4))
    print()

    print("Meaning:")
    print("Softmax turns raw scores into probabilities.")
    print("Cross-entropy loss is small when the target class probability is high.")
    print("Cross-entropy loss is large when the target class probability is low.")

    print()
    print("=== Compare Different Predictions ===")

    examples = [
        ([5.0, 1.0, 0.1], 0, "very confident and correct"),
        ([2.0, 1.0, 0.1], 0, "moderately correct"),
        ([0.1, 1.0, 5.0], 0, "confident but wrong"),
    ]

    for logits, target, description in examples:
        probs = softmax(logits)
        loss = cross_entropy_loss(logits, target)

        print()
        print(description)
        print("logits:", logits)
        print("probs:", [round(p, 4) for p in probs])
        print("target probability:", round(probs[target], 4))
        print("loss:", round(loss, 4))


if __name__ == "__main__":
    main()
