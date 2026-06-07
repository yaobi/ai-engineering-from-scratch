# Matrix Transformations Tutor

## 1. Matrix as a transformation

A matrix is not just a table of numbers.

It is a machine that transforms points in space.

In 2D, a 2x2 matrix tells where the basis vectors [1, 0] and [0, 1] move.

---

## 2. Rotation

A rotation matrix moves points around the origin.

It preserves:

- distance
- angle
- area

For 2D rotation:

R = [[cos(theta), -sin(theta)],
     [sin(theta),  cos(theta)]]

det(rotation) = 1

---

## 3. Scaling

A scaling matrix stretches or compresses each axis.

Example:

S = [[sx, 0],
     [0, sy]]

If sx = 2 and sy = 3, area becomes 2 * 3 = 6 times larger.

det(scale) = sx * sy

---

## 4. Shearing

Shearing tilts space.

It can turn a square into a parallelogram.

Example:

Shx = [[1, k],
       [0, 1]]

A shear changes shape but usually preserves area.

det(shear) = 1

---

## 5. Reflection

Reflection flips points across an axis.

Across y-axis:

[[-1, 0],
 [ 0, 1]]

Reflection preserves area but flips orientation.

det(reflection) = -1

---

## 6. Composition

Multiple transformations can be chained by matrix multiplication.

B @ A @ point means:

1. apply A first
2. apply B second

Order matters.

Usually:

S @ R != R @ S

So rotating then scaling is not the same as scaling then rotating.

---

## 7. Determinant

The determinant tells how much a transformation scales area in 2D or volume in 3D.

Examples:

det = 1: area preserved
det = 2: area doubled
det = 0: space collapsed to a lower dimension
det = -1: area preserved but orientation flipped

If det = 0, the matrix has no inverse.

---

## 8. Eigenvectors and eigenvalues

Most vectors change direction after a matrix transformation.

Eigenvectors are special directions that do not change direction.

Formula:

A @ v = lambda * v

v is the eigenvector.

lambda is the eigenvalue.

The eigenvalue tells how much the eigenvector direction is stretched or compressed.

---

## 9. Eigendecomposition

If a matrix has enough independent eigenvectors, it can be decomposed as:

A = V @ D @ V^-1

V contains eigenvectors.

D contains eigenvalues.

This means:

1. change into eigenvector coordinates
2. scale along eigenvector directions
3. change back to the original coordinates

---

## 10. AI connections

PCA:

Eigenvectors of the covariance matrix are principal directions.

Eigenvalues tell how much variance each direction explains.

RNN stability:

Large eigenvalues can cause exploding states or gradients.

Small eigenvalues can cause vanishing states or gradients.

Spectral methods:

Graph learning and spectral clustering use eigenvectors of graph matrices.

Data augmentation:

Image rotation, scaling, shearing, and reflection are matrix transformations.
