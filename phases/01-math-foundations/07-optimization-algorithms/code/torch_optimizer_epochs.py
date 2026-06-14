from torch_optimizer_demo import train_model


def main():
    print("=== Optimizer Comparison with Different Epochs ===")
    print("Target function: y = 2x + 1")
    print()

    optimizers = ["sgd", "momentum", "adam", "adamw"]
    epoch_list = [50, 200, 1000, 3000]

    for epochs in epoch_list:
        print(f"Epochs = {epochs}")
        print(f"{'optimizer':>10} {'w':>10} {'b':>10} {'loss':>14}")
        print("-" * 50)

        for name in optimizers:
            w, b, loss = train_model(name, epochs=epochs)

            print(
                f"{name:>10} "
                f"{w:10.4f} "
                f"{b:10.4f} "
                f"{loss:14.8f}"
            )

        print()


if __name__ == "__main__":
    main()
