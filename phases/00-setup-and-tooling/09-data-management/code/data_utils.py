from datasets import load_dataset
from huggingface_hub import hf_hub_download


def main():
    print("Loading IMDB sample dataset...")

    dataset = load_dataset("stanfordnlp/imdb", split="train[:100]")

    print(dataset)
    print()
    print("First example:")
    print(dataset[0])
    print()

    print("Creating train / validation / test splits...")

    split = dataset.train_test_split(test_size=0.3, seed=42)
    val_test = split["test"].train_test_split(test_size=0.5, seed=42)

    train_ds = split["train"]
    val_ds = val_test["train"]
    test_ds = val_test["test"]

    print(f"Train: {len(train_ds)}")
    print(f"Validation: {len(val_ds)}")
    print(f"Test: {len(test_ds)}")
    print()

    print("Saving small sample files...")

    dataset.to_csv("data/imdb_sample.csv")
    dataset.to_parquet("data/imdb_sample.parquet")

    print("Saved:")
    print("data/imdb_sample.csv")
    print("data/imdb_sample.parquet")
    print()

    print("Downloading a small model config file from Hugging Face cache...")

    config_path = hf_hub_download(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        filename="config.json",
    )

    print(f"Cached model config at: {config_path}")


if __name__ == "__main__":
    main()
