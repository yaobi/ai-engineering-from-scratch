import math
import numpy as np


def angle_between(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cosine = np.clip(cosine, -1.0, 1.0)

    return math.degrees(math.acos(cosine))


def most_similar_pair(vectors):
    best_pair = None
    best_score = -1

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a = vectors[i]
            b = vectors[j]

            score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            if score > best_score:
                best_score = score
                best_pair = (i, j)

    return best_pair, best_score


def main():
    print("=== Angle Between Vectors ===")

    a = [1, 0]
    b = [0, 1]

    print("angle between [1, 0] and [0, 1]:", angle_between(a, b))

    print()
    print("=== Scaling Matrix ===")

    scaling = np.array([
        [2, 0],
        [0, 3],
    ])

    point = np.array([1, 1])

    print("scaling matrix:")
    print(scaling)
    print("point:", point)
    print("scaled point:", scaling @ point)

    print()
    print("=== Most Similar Pair ===")

    np.random.seed(42)
    vectors = [np.random.randn(50) for _ in range(5)]

    pair, score = most_similar_pair(vectors)

    print("most similar pair:", pair)
    print("cosine similarity:", round(score, 4))

    print()
    print("=== Rank 2 Matrix ===")

    A = np.array([
        [1, 2, 3],
        [2, 4, 6],
        [1, 0, 1],
    ])

    print("A:")
    print(A)
    print("rank(A):", np.linalg.matrix_rank(A))
    print("The columns span a 2-dimensional plane in 3D space.")

    print()
    print("=== Projection in 3D ===")

    x = np.array([1, 2, 3], dtype=float)
    direction = np.array([1, 1, 1], dtype=float)

    projection = (np.dot(x, direction) / np.dot(direction, direction)) * direction

    print("x:", x)
    print("direction:", direction)
    print("projection:", projection)


if __name__ == "__main__":
    main()
