from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


def main():
    print("=== Scikit-learn Naive Bayes Demo ===")
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

    test_messages = [
        "free money waiting for you",
        "meeting rescheduled to friday",
        "you won a free prize",
        "please review the attached report",
    ]

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_docs)

    clf = MultinomialNB()
    clf.fit(X_train, train_labels)

    X_test = vectorizer.transform(test_messages)
    predictions = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)

    print("Vocabulary size:", len(vectorizer.vocabulary_))
    print()

    print("Predictions:")

    for msg, pred, prob in zip(test_messages, predictions, probabilities):
        print()
        print(f"Message: {msg}")
        print(f"Prediction: {pred}")

        for cls, p in zip(clf.classes_, prob):
            print(f"  P({cls}) = {p:.4f}")

    print()
    print("Meaning:")
    print("CountVectorizer turns text into word-count vectors.")
    print("MultinomialNB trains a Naive Bayes classifier using word counts.")
    print("This is the library version of the classifier we wrote from scratch.")


if __name__ == "__main__":
    main()
