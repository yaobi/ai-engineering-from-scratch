class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([
            a + b
            for a, b in zip(self.components, other.components)
        ])

    def __sub__(self, other):
        return Vector([
            a - b
            for a, b in zip(self.components, other.components)
        ])

    def dot(self, other):
        return sum(
            a * b
            for a, b in zip(self.components, other.components)
        )

    def magnitude(self):
        return sum(x ** 2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize a zero vector")
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


def main():
    a = Vector([1, 2, 3])
    b = Vector([4, 5, 6])

    print("a =", a)
    print("b =", b)
    print("a + b =", a + b)
    print("a - b =", a - b)
    print("a · b =", a.dot(b))
    print("|a| =", round(a.magnitude(), 4))
    print("normalized a =", a.normalize())
    print("cosine similarity =", round(a.cosine_similarity(b), 4))


if __name__ == "__main__":
    main()
