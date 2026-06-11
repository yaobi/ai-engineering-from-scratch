# Bayesian Inference Tutor

## 1. Bayes theorem

Bayes theorem updates belief after seeing evidence.

Formula:

P(H | E) = P(E | H) * P(H) / P(E)

Meaning:

posterior = likelihood * prior / evidence

---

## 2. Prior

Prior means the belief before seeing new evidence.

Example:

If a disease affects 1 in 10,000 people, then:

P(sick) = 0.0001

This prior matters a lot when interpreting a positive test.

---

## 3. Likelihood

Likelihood means how likely the evidence is if the hypothesis is true.

Example:

P(positive | sick) = 0.99

This says the test is positive 99% of the time when the person is truly sick.

---

## 4. False positive

False positive means the test says positive, but the true state is negative.

Example:

P(positive | healthy) = 0.01

When the disease is very rare, false positives can outnumber true positives.

---

## 5. Posterior

Posterior means the updated belief after seeing evidence.

Example:

Even if a test is 99% accurate, a positive result may still imply a low probability of disease when the disease is extremely rare.

This is the base rate effect.

---

## 6. Naive Bayes

Naive Bayes is a simple probabilistic classifier.

It estimates:

P(class | words)

For spam classification:

P(spam | words)
P ham | words

The model predicts the class with the higher score.

---

## 7. Naive assumption

Naive Bayes assumes features are independent given the class.

For words:

P(free, money, prize | spam)
≈ P(free | spam) * P(money | spam) * P(prize | spam)

This assumption is not always true, but the method often works well.

---

## 8. Laplace smoothing

Laplace smoothing prevents zero probabilities.

Without smoothing, an unseen word can make the whole probability zero.

Formula idea:

P(word | class) = (count + smoothing) / (total + smoothing * vocab_size)

Small smoothing keeps the model close to raw counts.

Large smoothing makes probabilities more uniform.

---

## 9. Log probability

Multiplying many small probabilities can cause numerical underflow.

Using log probabilities turns multiplication into addition:

log(a * b) = log(a) + log(b)

Naive Bayes commonly works in log space.

---

## 10. Scikit-learn Naive Bayes

CountVectorizer converts text into word-count vectors.

MultinomialNB trains a Naive Bayes classifier using word counts.

This is the library version of the from-scratch Naive Bayes classifier.

---

## 11. Conjugate priors

A conjugate prior means the prior and posterior belong to the same distribution family.

This makes Bayesian updating simple.

For Bernoulli or Binomial data, the Beta distribution is a common conjugate prior.

---

## 12. Beta distribution

Beta(a, b) represents belief about a probability.

Mean:

a / (a + b)

a acts like success count.

b acts like failure count.

---

## 13. Beta update

Prior:

Beta(a, b)

Observed data:

s successes, f failures

Posterior:

Beta(a + s, b + f)

This is simple Bayesian updating by addition.

---

## 14. Sequential Bayesian updating

Bayesian updating can be sequential.

Yesterday's posterior becomes today's prior.

Sequential update and batch update give the same final posterior if the data is the same.

This is useful for online learning and streaming systems.

---

## 15. Bayesian A/B testing

Bayesian A/B testing estimates:

P(B > A)

Example:

A: 50 clicks out of 1000 views
B: 65 clicks out of 1000 views

A posterior:

Beta(51, 951)

B posterior:

Beta(66, 936)

Monte Carlo sampling can estimate the probability that B is truly better than A.

---

## 16. AI connections

Naive Bayes is a classic text classification method.

Bayesian updating supports online learning.

Beta distributions can model click-through rates and conversion rates.

Bayesian A/B testing gives direct probability statements such as:

There is a 92% chance B is better than A.

Log probabilities are important in language models and sequence modeling.
