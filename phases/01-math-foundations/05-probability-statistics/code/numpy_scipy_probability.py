import numpy as np


def manual_softmax(logits):
    logits = np.array(logits, dtype=float)
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def manual_log_softmax(logits):
    logits = np.array(logits, dtype=float)
    shifted = logits - np.max(logits)
    log_sum_exp = np.max(logits) + np.log(np.sum(np.exp(shifted)))
    return logits - log_sum_exp


def main():
    print("=== NumPy Probability Demo ===")
    print()

    # Normal sampling
    samples = np.random.normal(loc=0, scale=1, size=10000)

    print("Normal samples:")
    print(f"Mean: {np.mean(samples):.4f}")
    print(f"Std:  {np.std(samples):.4f}")
    print()

    # Categorical sampling
    probs = np.array([0.6, 0.3, 0.1])
    categorical_samples = np.random.choice(
        a=[0, 1, 2],
        size=10000,
        p=probs,
    )

    counts = np.bincount(categorical_samples, minlength=3)
    proportions = counts / len(categorical_samples)

    print("Categorical sampling:")
    print("True probs:", probs)
    print("Sample proportions:", np.round(proportions, 4))
    print()

    # Softmax
    logits = np.array([2.0, 1.0, 0.1])

    softmax_probs = manual_softmax(logits)
    log_probs = manual_log_softmax(logits)

    print("Softmax:")
    print("Logits:", logits)
    print("Softmax probs:", np.round(softmax_probs, 4))
    print("Log-softmax:", np.round(log_probs, 4))
    print()

    # Cross entropy
    target_index = 0
    loss = -log_probs[target_index]

    print("Cross-entropy:")
    print("Target index:", target_index)
    print("Target probability:", round(softmax_probs[target_index], 4))
    print("Loss:", round(loss, 4))
    print()

    print("Meaning:")
    print("NumPy lets us compute probability operations faster and more concisely.")
    print("The formulas are the same as the scratch implementations.")


if __name__ == "__main__":
    main()
