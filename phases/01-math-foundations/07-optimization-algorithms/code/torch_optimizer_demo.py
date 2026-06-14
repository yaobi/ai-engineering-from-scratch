import torch


def train_model(optimizer_name="adam", epochs=200):
    torch.manual_seed(42)

    # Simple data: y = 2x + 1
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]])

    model = torch.nn.Linear(1, 1)
    loss_fn = torch.nn.MSELoss()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
        )
    elif optimizer_name == "momentum":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.01,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.01,
            weight_decay=0.01,
        )
    else:
        raise ValueError("Unknown optimizer")

    for epoch in range(epochs):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    weight = model.weight.item()
    bias = model.bias.item()
    final_loss = loss_fn(model(x), y).item()

    return weight, bias, final_loss


def main():
    print("=== PyTorch Optimizer Demo ===")
    print("Target function: y = 2x + 1")
    print()

    optimizers = ["sgd", "momentum", "adam", "adamw"]

    print(f"{'optimizer':>10} {'w':>10} {'b':>10} {'loss':>14}")
    print("-" * 50)

    for name in optimizers:
        w, b, loss = train_model(name, epochs=200)

        print(
            f"{name:>10} "
            f"{w:10.4f} "
            f"{b:10.4f} "
            f"{loss:14.8f}"
        )

    print()
    print("Meaning:")
    print("All optimizers try to learn w=2 and b=1.")
    print("SGD uses basic gradient descent.")
    print("Momentum adds velocity.")
    print("Adam adapts the step size automatically.")
    print("AdamW is Adam with decoupled weight decay.")


if __name__ == "__main__":
    main()
