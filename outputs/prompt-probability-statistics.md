# Probability and Statistics Tutor

## 1. Probability basics

Probability measures uncertainty.

A probability is between 0 and 1.

0 means impossible.
1 means certain.

---

## 2. Conditional probability

Conditional probability means the probability of A given that B has already happened.

Formula:

P(A | B) = P(A and B) / P(B)

Example:

P(King | Face card) = 4 / 12 = 1 / 3

The sample space changes after the condition is known.

---

## 3. PMF

PMF means probability mass function.

It is used for discrete variables.

Example:

coin flip
die roll
class labels

PMF gives the probability of each exact outcome.

---

## 4. PDF

PDF means probability density function.

It is used for continuous variables.

Example:

height
weight
temperature
normal distribution

For continuous variables, probability comes from the area under the curve over an interval.

---

## 5. Expected value

Expected value is the long-run average.

Formula idea:

E[X] = sum(value * probability)

Example:

A fair die has expected value 3.5.

---

## 6. Variance and standard deviation

Variance measures spread.

It measures how far outcomes are from the expected value.

Standard deviation is the square root of variance.

High variance means more uncertainty and instability.

---

## 7. Sampling

Sampling means generating random values according to a probability distribution.

Examples:

Bernoulli sampling generates 0 or 1.
Categorical sampling generates class labels.
Normal sampling generates bell-shaped continuous values.

With many samples, observed frequencies become close to theoretical probabilities.

---

## 8. Softmax

Softmax turns logits into probabilities.

Logits are raw model scores.

Softmax output has two properties:

1. every value is between 0 and 1
2. all values sum to 1

This makes it a valid probability distribution.

---

## 9. Log probability

Log probability is used for numerical stability.

Products of many small probabilities can become too small.

Using logs turns multiplication into addition:

log(a * b) = log(a) + log(b)

This is important for language models and sequence models.

---

## 10. Cross-entropy

Cross-entropy measures how wrong a predicted probability distribution is.

For one correct class:

cross entropy = -log(probability of correct class)

If the model gives high probability to the correct class, loss is small.

If the model gives low probability to the correct class, loss is large.

---

## 11. Central Limit Theorem

The Central Limit Theorem says:

The average of many independent samples tends to follow a normal distribution.

Even if the original distribution is not normal, sample averages become more stable and bell-shaped.

This explains why batch averages in machine learning are more stable than single-sample estimates.

---

## 12. AI connections

Classification models use softmax to output class probabilities.

Language models sample the next token from a probability distribution.

Cross-entropy is the standard loss for classification and language modeling.

Log probabilities prevent numerical underflow in long sequences.

Expected value and variance help describe model uncertainty and data noise.
