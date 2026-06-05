import logging
import time
import tracemalloc

import torch
import torch.nn as nn


def debug_print(name, tensor):
    print(
        f"{name}: "
        f"shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, "
        f"device={tensor.device}, "
        f"min={tensor.min().item():.4f}, "
        f"max={tensor.max().item():.4f}, "
        f"mean={tensor.float().mean().item():.4f}, "
        f"has_nan={torch.isnan(tensor).any().item()}"
    )


class Timer:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"[{self.name}] {elapsed:.6f}s")


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)


def check_shapes(model, sample_input):
    print("\n=== Shape Check ===")
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            in_shape = tuple(inp[0].shape)
            out_shape = tuple(out.shape) if hasattr(out, "shape") else type(out)
            print(f"{name}: {in_shape} -> {out_shape}")

        return hook

    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(sample_input)

    for h in hooks:
        h.remove()


def detect_nan(loss, step):
    if torch.isnan(loss):
        print(f"NaN loss detected at step {step}")
        return True
    return False


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("outputs/debug_training.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def memory_demo():
    print("\n=== Memory Profiling Demo ===")
    tracemalloc.start()

    data = [i for i in range(100000)]

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    for stat in top_stats[:3]:
        print(stat)

    del data
    tracemalloc.stop()


def main():
    logger = setup_logger()

    print("=== Debugging Toolkit Demo ===")

    model = TinyModel()
    x = torch.randn(5, 4)
    y = torch.tensor([0, 1, 0, 1, 1])

    debug_print("input x", x)

    check_shapes(model, x)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print("\n=== Training Step Demo ===")

    with Timer("forward pass"):
        outputs = model(x)

    debug_print("outputs", outputs)

    with Timer("loss calculation"):
        loss = criterion(outputs, y)

    print(f"loss: {loss.item():.4f}")
    logger.info("Training step loss=%.4f", loss.item())

    if detect_nan(loss, step=1):
        logger.error("NaN loss detected")

    with Timer("backward pass"):
        loss.backward()
        optimizer.step()

    print("\n=== Gradient Check ===")
    for name, param in model.named_parameters():
        if param.grad is not None:
            debug_print(f"grad {name}", param.grad)

    memory_demo()

    print("\nDone. Log saved to outputs/debug_training.log")


if __name__ == "__main__":
    main()
