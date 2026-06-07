from matrix_operations import Matrix

A = Matrix([[1, 2], [2, 4]])

print("A =", A.data)
print("det(A) =", A.determinant())

try:
    print("A^-1 =", A.inverse_2x2().data)
except ValueError as e:
    print("Error:", e)

print("Because row 2 = 2 * row 1, this matrix loses one dimension and has no inverse.")
