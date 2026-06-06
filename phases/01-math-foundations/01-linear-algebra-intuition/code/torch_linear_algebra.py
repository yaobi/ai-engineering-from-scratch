import torch


def main():
    print("=== PyTorch Tensor Basics ===")

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])

    print("a =", a)
    print("b =", b)
    print("a + b =", a + b)
    print("a · b =", torch.dot(a, b))
    print("|a| =", torch.linalg.norm(a))

    cosine = torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b))
    print("cosine similarity =", cosine)

    print()
    print("=== Matrix Multiplication ===")

    W = torch.tensor([
        [0.1, -0.2, 0.3],
        [0.4, 0.5, -0.1],
    ])

    x = torch.tensor([1.0, 0.5, -0.3])

    print("W =", W)
    print("x =", x)
    print("W @ x =", W @ x)

    print()
    print("=== Autograd Demo ===")

    x = torch.randn(3, requires_grad=True)
    y = torch.tensor([1.0, 0.0, 0.0])

    similarity = torch.dot(x, y)
    similarity.backward()

    print("x =", x.data)
    print("y =", y)
    print("dot product =", similarity.item())
    print("d(dot)/dx =", x.grad)
    print("The gradient of dot(x, y) with respect to x is y.")


if __name__ == "__main__":
    main()
