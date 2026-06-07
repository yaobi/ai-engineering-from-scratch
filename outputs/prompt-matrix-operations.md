# Matrix Operations Tutor

## 1. Vector

A vector is an ordered list of numbers.

In AI, a vector can represent:

- input features
- word embeddings
- image embeddings
- user preferences
- hidden states in a neural network

Important operations:

- addition
- subtraction
- scalar multiplication
- dot product
- magnitude

---

## 2. Matrix

A matrix is a linear transformation.

It maps vectors from one space to another.

In AI, matrices often represent:

- neural network weights
- attention matrices
- embedding tables
- projection layers

---

## 3. Element-wise multiplication vs matrix multiplication

Element-wise multiplication:

Multiply matching positions.

Example:

A * B

Matrix multiplication:

Each output value is a dot product between one row of the first matrix and one column of the second matrix.

Example:

A @ B

This is the core operation in neural networks.

---

## 4. Transpose

Transpose swaps rows and columns.

A matrix with shape:

(m, n)

becomes:

(n, m)

Transpose is used in:

- matrix multiplication shape alignment
- backpropagation
- attention: Q @ K.T

---

## 5. Determinant

The determinant measures how much a matrix transformation scales area or volume.

For a 2x2 matrix:

[[a, b],
 [c, d]]

det = a*d - b*c

If det = 0, the matrix crushes space into a lower dimension and has no inverse.

---

## 6. Inverse

The inverse matrix undoes a matrix transformation.

A @ A^-1 = I

The inverse only exists when the determinant is not zero.

---

## 7. Identity matrix

The identity matrix acts like the number 1 for matrix multiplication.

A @ I = A

I @ A = A

---

## 8. Dense neural network layer

A dense layer computes:

output = relu(W @ x + b)

Where:

- x is the input vector
- W is the weight matrix
- b is the bias
- ReLU keeps positive values and turns negative values into 0

A two-layer network computes:

hidden = relu(W1 @ x + b1)
output = W2 @ hidden + b2

---

## 9. NumPy

NumPy performs the same matrix operations much faster than pure Python.

Common operations:

A + B
A * B
A @ B
A.T
np.linalg.det(A)
np.linalg.inv(A)
np.eye(n)

---

## 10. Broadcasting

Broadcasting lets NumPy automatically expand smaller arrays to match larger arrays.

Example:

matrix + bias_vector

This is how bias addition works in neural networks.
