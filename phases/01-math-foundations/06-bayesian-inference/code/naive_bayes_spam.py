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

    def predict_with_scores(self, document):
        words = document.lower().split()

        total_docs = sum(self.class_counts.values())
        vocab_size = len(self.vocab)

        scores = {}

        for cls in self.class_counts:
            # log P(class)
            score = math.log(self.class_counts[cls] / total_docs)

            for word in words:
                count = self.word_counts[cls].get(word, 0)
                total = self.class_word_totals[cls]

                # Laplace smoothing:
                # P(word | class) = (count + smoothing) / (total + smoothing * vocab_size)
                word_prob = (
                    (count + self.smoothing)
                    / (total + self.smoothing * vocab_size)
                )

                score += math.log(word_prob)

            scores[cls] = score

        return scores

    def predict(self, document):
        scores = self.predict_with_scores(document)

        best_class = max(scores, key=scores.get)

        return best_class


def show_top_words(classifier, cls, n=5):
    vocab_size = len(classifier.vocab)
    total = classifier.class_word_totals[cls]

    probs = {}

    for word in classifier.vocab:
        count = classifier.word_counts[cls].get(word, 0)

        probs[word] = (
            (count + classifier.smoothing)
            / (total + classifier.smoothing * vocab_size)
        )

    sorted_words = sorted(
        probs.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for word, prob in sorted_words[:n]:
        print(f"  {word}: {prob:.4f}")


def main():
    print("=== Naive Bayes Spam Classifier ===")
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

    classifier = NaiveBayes(smoothing=1.0)
    classifier.train(train_docs, train_labels)

    print("Class counts:")
    print(dict(classifier.class_counts))
    print()

    print("Vocabulary size:")
    print(len(classifier.vocab))
    print()

    test_messages = [
        "free money waiting for you",
        "meeting rescheduled to friday",
        "you won a free prize",
        "please review the attached report",
    ]

    print("Predictions:")

    for msg in test_messages:
        pred = classifier.predict(msg)
        scores = classifier.predict_with_scores(msg)

        print()
        print(f"Message: {msg}")
        print(f"Prediction: {pred}")
        print("Log scores:")

        for cls, score in scores.items():
            print(f"  {cls}: {score:.4f}")

    print()
    print("Top spam words:")
    show_top_words(classifier, "spam")

    print()
    print("Top ham words:")
    show_top_words(classifier, "ham")

    print()
    print("Meaning:")
    print("Words like free, prize, money push the score toward spam.")
    print("Words like meeting, review, attached push the score toward ham.")
    print("Log probabilities are used instead of multiplying many tiny probabilities.")


if __name__ == "__main__":
    main()
