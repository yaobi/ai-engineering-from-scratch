from vector_basics import Vector


def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])


def gram_schmidt(vectors):
    orthonormal = []

    for v in vectors:
        w = v

        for u in orthonormal:
            proj = project(w, u)
            w = w - proj

        if w.magnitude() < 1e-10:
            continue

        orthonormal.append(w.normalize())

    return orthonormal


def main():
    print("=== Projection Demo ===")

    a = Vector([3, 4])
    b = Vector([1, 0])

    proj = project(a, b)
    residual = a - proj

    print("a =", a)
    print("b =", b)
    print("projection of a onto b =", proj)
    print("residual =", residual)
    print("residual · b =", residual.dot(b))

    print()
    print("=== Gram-Schmidt Demo ===")

    v1 = Vector([1, 0, 0])
    v2 = Vector([1, 1, 0])
    v3 = Vector([1, 1, 1])

    basis = gram_schmidt([v1, v2, v3])

    for i, u in enumerate(basis):
        print(f"u{i + 1} =", u)
        print(f"|u{i + 1}| =", round(u.magnitude(), 6))

    print("u1 · u2 =", round(basis[0].dot(basis[1]), 6))
    print("u1 · u3 =", round(basis[0].dot(basis[2]), 6))
    print("u2 · u3 =", round(basis[1].dot(basis[2]), 6))


if __name__ == "__main__":
    main()
