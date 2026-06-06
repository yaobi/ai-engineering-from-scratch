# Linear Algebra Tutor

## 1. Vector

A vector is a list of numbers.

It can represent:

- a point
- a direction
- a word embedding
- an image embedding
- a user preference vector

Important operations:

- vector addition
- vector subtraction
- dot product
- magnitude
- normalization
- cosine similarity

---

## 2. Dot Product

Formula:

a · b = a1*b1 + a2*b2 + ... + an*bn

Meaning:

The dot product measures how aligned two vectors are.

- positive: similar direction
- zero: perpendicular
- negative: opposite direction

AI connection:

Dot product is used in:

- embedding similarity
- semantic search
- recommendation systems
- RAG retrieval
- Transformer attention scores

---

## 3. Cosine Similarity

Formula:

cosine = dot(a, b) / (|a| * |b|)

Meaning:

Cosine similarity measures direction similarity while reducing the effect of vector length.

AI connection:

It is commonly used to compare text embeddings.

---

## 4. Matrix

A matrix is a transformation.

It maps one vector to another vector.

Example:

output = W @ x

AI connection:

A neural network layer is mainly matrix multiplication.

Input vector x goes through weight matrix W and becomes output vector.

---

## 5. Projection

Projection means keeping the component of one vector in another direction.

Example:

Project [3, 4] onto [1, 0]:

result = [3, 0]

AI connection:

Projection appears in:

- linear regression
- PCA
- dimensionality reduction
- attention query/key/value projections

---

## 6. Rank

Rank means how many independent directions a matrix contains.

If a matrix has low rank, it means some rows or columns contain redundant information.

AI connection:

Rank appears in:

- feature redundancy
- multicollinearity
- low-rank approximation
- LoRA fine-tuning

---

## 7. Gram-Schmidt

Gram-Schmidt converts independent vectors into an orthonormal basis.

Orthonormal means:

- every vector has length 1
- every pair of vectors is perpendicular

AI connection:

It is related to QR decomposition and stable numerical computation.

---

## 8. PyTorch Autograd

PyTorch tensors can track gradients.

requires_grad=True tells PyTorch to record operations.

backward() computes gradients automatically.

Training workflow:

forward pass
loss calculation
backward()
optimizer step

This is the foundation of neural network training.
