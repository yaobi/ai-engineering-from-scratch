import math
from collections import defaultdict


class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_word_totals = defaultdict(int)
        self.vocab = set()

    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = doc.lower().split()

            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

    def word_probability(self, word, cls):
        vocab_size = len(self.vocab)
        count = self.word_counts[cls].get(word, 0)
        total = self.class_word_totals[cls]

        return (
            (count + self.smoothing)
            / (total + self.smoothing * vocab_size)
        )


def main():
    print("=== Smoothing Impact Demo ===")
    print()

    train_docs = [
        "win free money now",
        "free lottery ticket winner",
        "claim your prize today free",
        "urgent offer free cash",
        "congratulations you won free",
        "meeting tomorrow at noon",
        "project update attached",
        "can we schedule a call",
        "quarterly report review",
        "lunch on thursday sounds good",
        "team standup notes attached",
        "please review the pull request",
    ]

    train_labels = [
        "spam", "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham", "ham", "ham", "ham",
    ]

    words_to_check = ["free", "money", "meeting", "attached", "unknownword"]
    smoothing_values = [0.01, 0.1, 1.0, 10.0]

    for smoothing in smoothing_values:
        clf = NaiveBayes(smoothing=smoothing)
        clf.train(train_docs, train_labels)

        print(f"Smoothing = {smoothing}")
        for word in words_to_check:
            p_spam = clf.word_probability(word, "spam")
            p_ham = clf.word_probability(word, "ham")

            print(
                f"  word={word:<12} "
                f"P(word|spam)={p_spam:.6f}  "
                f"P(word|ham)={p_ham:.6f}"
            )

        print()

    print("Meaning:")
    print("Small smoothing keeps the model close to raw word counts.")
    print("Large smoothing makes word probabilities more uniform.")
    print("Smoothing prevents unseen words from getting probability zero.")


if __name__ == "__main__":
    main()
