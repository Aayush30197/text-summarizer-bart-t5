from __future__ import annotations

"""
Dataset loading, fixed-subset creation, and tokenization for CNN/DailyMail.

Key responsibilities:
  - Load dataset from Hugging Face and verify split sizes.
  - Create deterministic train/val/test subsets using a fixed seed.
  - Save and load subset indices to data/ so experiments are reproducible.
  - Provide tokenization functions compatible with both BART and T5.

Usage:
    python src/data.py                 # full subset creation
    python src/data.py --smoke         # tiny 50-example test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

# Allow running as `python src/data.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import set_seed, load_config, Timer


# ---------------------------------------------------------------------------
# Expected split sizes from the spec (verified at runtime)
# ---------------------------------------------------------------------------
EXPECTED_SPLITS = {
    "train": 287_113,
    "validation": 13_368,
    "test": 11_490,
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_cnn_dailymail(dataset_name: str = "abisee/cnn_dailymail",
                       dataset_config: str = "3.0.0"):
    """Load the CNN/DailyMail dataset and verify split sizes.

    Tries the spec's dataset name first; falls back to 'cnn_dailymail'
    if the namespaced version isn't found.

    Returns:
        The loaded DatasetDict.
    """
    print(f"[data] Loading dataset: {dataset_name} (config={dataset_config})")
    try:
        ds = load_dataset(dataset_name, dataset_config)
    except Exception as e:
        print(f"[data] Failed with '{dataset_name}': {e}")
        fallback = "cnn_dailymail"
        print(f"[data] Trying fallback name: {fallback}")
        ds = load_dataset(fallback, dataset_config)

    # Verify split sizes
    print("[data] Split sizes:")
    all_match = True
    for split_name, expected_size in EXPECTED_SPLITS.items():
        actual_size = len(ds[split_name])
        match = "✓" if actual_size == expected_size else "✗ MISMATCH"
        if actual_size != expected_size:
            all_match = False
        print(f"  {split_name}: {actual_size:,} (expected {expected_size:,}) {match}")

    if not all_match:
        print("[data] WARNING: Split sizes do not match the spec. "
              "Document this in the report.")

    return ds


# ---------------------------------------------------------------------------
# Fixed-subset index creation
# ---------------------------------------------------------------------------

def create_subset_indices(
    dataset_sizes: dict[str, int],
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Create fixed random subset indices for train, val, and test splits.

    Uses a separate RNG (seeded deterministically) so results are independent
    of any other random state.  No overlap between splits is possible because
    each split draws from its own original split.

    Args:
        dataset_sizes: Dict with keys 'train', 'validation', 'test' and
            their full sizes as values.
        train_size: Number of training examples to sample.
        val_size: Number of validation examples to sample.
        test_size: Number of test examples to sample.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys 'train', 'validation', 'test' mapping to sorted
        lists of integer indices.
    """
    rng = np.random.RandomState(seed)

    indices = {}
    for split_name, subset_size in [
        ("train", train_size),
        ("validation", val_size),
        ("test", test_size),
    ]:
        full_size = dataset_sizes[split_name]
        if subset_size > full_size:
            print(f"[data] WARNING: Requested {subset_size} for {split_name} "
                  f"but only {full_size} available. Using all.")
            subset_size = full_size
        chosen = rng.choice(full_size, size=subset_size, replace=False)
        indices[split_name] = sorted(chosen.tolist())
        print(f"[data] {split_name}: sampled {len(indices[split_name]):,} indices "
              f"from {full_size:,}")

    return indices


def save_indices(indices: dict[str, list[int]], data_dir: str = "data") -> None:
    """Save subset indices to individual JSON files in data_dir."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    for split_name, idx_list in indices.items():
        filename = f"indices_{split_name}.json"
        out_path = data_path / filename
        with open(out_path, "w") as f:
            json.dump(idx_list, f)
        print(f"[data] Saved {len(idx_list):,} indices to {out_path}")


def load_indices(data_dir: str = "data") -> dict[str, list[int]]:
    """Load previously saved subset indices from data_dir."""
    data_path = Path(data_dir)
    indices = {}
    for split_name in ["train", "validation", "test"]:
        filename = f"indices_{split_name}.json"
        file_path = data_path / filename
        with open(file_path, "r") as f:
            indices[split_name] = json.load(f)
        print(f"[data] Loaded {len(indices[split_name]):,} indices from {file_path}")
    return indices


def get_subsets(ds, indices: dict[str, list[int]]):
    """Select fixed subsets from the dataset using pre-saved indices.

    Returns:
        Dict mapping split names to Dataset objects.
    """
    subsets = {}
    for split_name, idx_list in indices.items():
        subsets[split_name] = ds[split_name].select(idx_list)
        print(f"[data] {split_name} subset: {len(subsets[split_name]):,} examples")
    return subsets


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def build_tokenize_function(tokenizer, max_source_length: int, max_target_length: int,
                            source_prefix: str = ""):
    """Return a tokenization function compatible with dataset.map().

    Handles:
      - Optional source prefix (e.g. 'summarize: ' for T5).
      - Padding labels with -100 so the loss ignores pad tokens.

    Args:
        tokenizer: A HuggingFace tokenizer.
        max_source_length: Maximum number of tokens for the source article.
        max_target_length: Maximum number of tokens for the target summary.
        source_prefix: String prepended to each article (empty for BART).

    Returns:
        A function suitable for Dataset.map(batched=True).
    """
    def tokenize_fn(examples):
        # CNN/DailyMail columns: 'article' and 'highlights'
        inputs = [source_prefix + article for article in examples["article"]]
        targets = examples["highlights"]

        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
            padding="max_length",
        )

        # Tokenize targets
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )

        # Replace pad token ids in labels with -100 so they're ignored by the loss
        label_ids = labels["input_ids"]
        label_ids = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
            for label in label_ids
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    return tokenize_fn


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Load CNN/DailyMail, create fixed subsets, save indices."
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use tiny subsets (~50 examples) for a quick pipeline test."
    )
    parser.add_argument(
        "--config", nargs="+",
        default=["configs/base.yaml"],
        help="Config YAML file(s) to load (later files override earlier ones)."
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(*args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # Determine subset sizes
    if args.smoke:
        smoke_size = cfg.get("smoke_subset_size", 50)
        train_size = smoke_size
        val_size = smoke_size
        test_size = smoke_size
        print(f"[data] SMOKE MODE: using {smoke_size} examples per split")
    else:
        train_size = cfg.get("train_subset_size", 12000)
        val_size = cfg.get("val_subset_size", 1000)
        test_size = cfg.get("test_subset_size", 1000)

    data_dir = cfg.get("data_dir", "data")

    # Load dataset and verify splits
    with Timer("Dataset loading"):
        ds = load_cnn_dailymail(
            dataset_name=cfg.get("dataset_name", "abisee/cnn_dailymail"),
            dataset_config=cfg.get("dataset_config", "3.0.0"),
        )

    # Create and save subset indices
    dataset_sizes = {split: len(ds[split]) for split in EXPECTED_SPLITS}
    indices = create_subset_indices(
        dataset_sizes=dataset_sizes,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )
    save_indices(indices, data_dir=data_dir)

    # Verify subsets load correctly
    subsets = get_subsets(ds, indices)

    # Print a sample article for sanity checking
    print("\n" + "=" * 60)
    print("SAMPLE ARTICLE (first from train subset)")
    print("=" * 60)
    sample = subsets["train"][0]
    article_preview = sample["article"][:300]
    highlights_preview = sample["highlights"][:200]
    print(f"Article (first 300 chars):\n{article_preview}...")
    print(f"\nHighlights (first 200 chars):\n{highlights_preview}...")
    print("=" * 60)

    print("\n[data] Done. Subset indices saved to data/.")
    print("[data] Next step: inspect data statistics and tokenize.")


if __name__ == "__main__":
    main()
