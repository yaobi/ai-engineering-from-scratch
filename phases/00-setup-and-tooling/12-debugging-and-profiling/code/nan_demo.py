import torch

def detect_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    print(f"No NaN in {name}")
    return False

x = torch.tensor([1.0, 2.0, 0.0])

print("Original tensor:")
print(x)

y = x / x

print("After division:")
print(y)

detect_nan(y, "y")
