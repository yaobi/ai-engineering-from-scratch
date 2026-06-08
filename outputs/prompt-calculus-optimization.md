# Calculus and Optimization Tutor

## 1. Derivative

A derivative measures the rate of change of a function at a point.

For:

f(x) = x^2

The derivative is:

f'(x) = 2x

Meaning:

The derivative tells how fast the function changes when x changes slightly.

---

## 2. Numerical derivative

Numerical derivative approximates the slope by evaluating nearby points.

Formula:

(f(x + h) - f(x - h)) / (2h)

It is useful for understanding and checking gradients.

---

## 3. Partial derivative

For a function with multiple variables, a partial derivative measures the change with respect to one variable while holding others fixed.

Example:

f(x, y) = x^2 + 3xy + y^2

df/dx = 2x + 3y
df/dy = 3x + 2y

---

## 4. Gradient

A gradient is a vector of partial derivatives.

gradient = [df/dx, df/dy]

It points in the direction where the function increases fastest.

To reduce a function, move in the opposite direction of the gradient.

---

## 5. Gradient descent

Gradient descent updates parameters by moving opposite to the gradient.

Formula:

parameter = parameter - learning_rate * gradient

In neural networks:

w = w - lr * dw
b = b - lr * db

---

## 6. Learning rate

Learning rate controls the step size.

If it is too large, training may diverge.

If it is too small, training becomes slow.

---

## 7. Hessian

The Hessian is a matrix of second derivatives.

It describes curvature.

A bowl-shaped function has positive curvature.

A saddle function has mixed curvature.

---

## 8. Taylor approximation

Taylor approximation estimates a function near a point using derivatives.

First-order idea:

f(x + h) ≈ f(x) + f'(x)h

This explains why gradient descent works best with small steps.

---

## 9. Linear model training

A simple model:

pred = w * x + b

Training loop:

1. predict
2. compute error
3. compute loss
4. compute gradients
5. update parameters

This is the simplest version of neural network training.

---

## 10. PyTorch connection

Manual gradient update:

w -= lr * dw
b -= lr * db

PyTorch version:

loss.backward()
optimizer.step()

PyTorch automatically computes gradients, but the logic is the same.
