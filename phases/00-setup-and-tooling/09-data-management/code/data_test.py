from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb", split="train[:5]")

print(dataset)
print()
print("First example:")
print(dataset[0])
