# Optimizer Guide

## 1. What is an optimizer?

An optimizer decides how to update model parameters using gradients.

The goal is to reduce the loss.

Basic idea:

parameter = parameter - learning_rate * gradient

---

## 2. Gradient Descent

Gradient Descent uses the current gradient only.

Formula:

p = p - lr * g

It is simple and stable with a good learning rate.

But it can be slow, especially in narrow curved valleys.

---

## 3. Learning Rate

Learning rate controls the step size.

If learning rate is too small, training is stable but slow.

If learning rate is too large, training can overshoot, diverge, or produce inf/nan loss.

Learning rate is one of the most important hyperparameters.

---

## 4. Momentum

Momentum accumulates past gradients into a velocity.

Formula idea:

velocity = momentum * old_velocity + gradient
parameter = parameter - lr * velocity

Momentum helps reduce oscillation and accelerates movement in consistent directions.

It behaves like a ball rolling downhill.

---

## 5. SGD with Momentum

SGD with Momentum is often stronger than plain gradient descent.

In this lesson, larger momentum improved convergence on the Rosenbrock function.

But too much momentum can overshoot in some problems.

Common value:

momentum = 0.9

---

## 6. Adam

Adam tracks two moving averages:

m: average direction of gradients
v: average of squared gradients

Adam uses these to adapt the step size for each parameter.

It often works well with little tuning.

Common starting point:

Adam(lr=0.001)

---

## 7. AdamW

AdamW is Adam with decoupled weight decay.

It is widely used for transformer models.

Common starting point:

AdamW(lr=0.001, weight_decay=0.01)

Weight decay discourages overly large parameters and may improve generalization.

---

## 8. Learning Rate Schedule

A learning rate schedule changes lr over time.

Large steps early can speed up learning.

Smaller steps later can improve stability near the minimum.

But decay must be tuned.

If decay is too strong, lr becomes too small too early and training stalls.

---

## 9. PyTorch training loop

A standard PyTorch training loop:

pred = model(x)
loss = loss_fn(pred, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()

Meaning:

1. predict
2. compute loss
3. clear old gradients
4. compute new gradients
5. update parameters

---

## 10. Rules of thumb

Start with Adam or AdamW for most neural network tasks.

Use AdamW for transformers.

Use SGD with momentum when you want strong final accuracy and can tune carefully.

If loss becomes nan or inf, reduce the learning rate.

If loss decreases too slowly, try increasing the learning rate or using momentum / Adam.

Use a learning rate schedule for longer training runs.

---

## 11. Key lesson

GD uses the current gradient.

Momentum uses the current gradient plus past direction.

Adam uses gradient direction and gradient scale to adapt step sizes.

All optimizers answer the same question:

Given the gradient, how should the parameters move next?
