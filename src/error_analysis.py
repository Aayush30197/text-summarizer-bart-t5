"""
Error analysis: builds a blinded rating sheet and (later) de-anonymizes results.

Steps:
  1. Sample 25 test articles (fixed seed).
  2. Generate summaries from both fine-tuned models.
  3. Create a blinded CSV with shuffled 'Summary A' / 'Summary B' columns.
  4. Create a separate key file mapping A/B to model names.
  5. After manual ratings, run --deblind to produce summary tables.

Usage:
    python src/error_analysis.py --generate          # generate blinded sheet
    python src/error_analysis.py --generate --smoke  # quick 5-article test
    python src/error_analysis.py --deblind           # after ratings are filled in
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_cnn_dailymail, load_indices, get_subsets
from src.utils import (
    get_device,
    is_mac_mps,
    load_config,
    load_decoding_config,
    set_seed,
    Timer,
)


def generate_summaries_simple(
    model,
    tokenizer,
    articles: list[str],
    decoding_cfg: dict,
    source_prefix: str,
    max_source_length: int,
    device: torch.device,
) -> list[str]:
    """Generate one summary at a time (small sample, no batching needed)."""
    model.eval()
    summaries = []
    for article in tqdm(articles, desc="Generating"):
        text = source_prefix + article
        inputs = tokenizer(
            text,
            max_length=max_source_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                num_beams=decoding_cfg["num_beams"],
                max_length=decoding_cfg["max_length"],
                min_length=decoding_cfg["min_length"],
                length_penalty=decoding_cfg["length_penalty"],
                no_repeat_ngram_size=decoding_cfg["no_repeat_ngram_size"],
                early_stopping=decoding_cfg["early_stopping"],
            )
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        summaries.append(decoded)
    return summaries


def generate_blinded_sheet(args):
    """Generate the blinded rating CSV and key file."""
    # Configs
    cfg = load_config("configs/base.yaml")
    bart_cfg = load_config("configs/base.yaml", "configs/bart.yaml")
    t5_cfg = load_config("configs/base.yaml", "configs/t5.yaml")
    decoding_cfg = load_decoding_config()
    seed = cfg.get("seed", 42)
    set_seed(seed)

    sample_size = 5 if args.smoke else 25
    results_dir = Path(cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    on_mps = is_mac_mps(device)

    # Load test data
    with Timer("Dataset loading"):
        ds = load_cnn_dailymail()
    indices = load_indices(cfg.get("data_dir", "data"))
    subsets = get_subsets(ds, indices)

    # Fixed-seed sample of test articles
    rng = random.Random(seed)
    test_indices = list(range(len(subsets["test"])))
    sampled_indices = sorted(rng.sample(test_indices, min(sample_size, len(test_indices))))
    sample_data = subsets["test"].select(sampled_indices)

    articles = sample_data["article"]
    references = sample_data["highlights"]
    print(f"[error_analysis] Sampled {len(articles)} articles for error analysis")

    # Check if precomputed evaluation outputs exist (e.g. from Kaggle run)
    bart_eval_path = results_dir / "eval_bart_full_outputs.json"
    t5_eval_path = results_dir / "eval_t5_full_outputs.json"

    if bart_eval_path.exists() and t5_eval_path.exists():
        print(f"[error_analysis] Found precomputed outputs in {results_dir}! Using exact test predictions.")
        with open(bart_eval_path) as f:
            bart_all = json.load(f)
        with open(t5_eval_path) as f:
            t5_all = json.load(f)

        bart_summaries = [bart_all[idx]["prediction"] for idx in sampled_indices]
        t5_summaries = [t5_all[idx]["prediction"] for idx in sampled_indices]
    else:
        # Load and generate with BART
        bart_checkpoint = args.bart_checkpoint or bart_cfg["output_dir"]
        print(f"[error_analysis] Loading BART from {bart_checkpoint}")
        bart_tokenizer = AutoTokenizer.from_pretrained(bart_checkpoint)
        bart_model = AutoModelForSeq2SeqLM.from_pretrained(bart_checkpoint).to(device)
        if bart_cfg.get("fp16", False) and not on_mps:
            bart_model.half()

        print("[error_analysis] Generating BART summaries...")
        bart_summaries = generate_summaries_simple(
            bart_model, bart_tokenizer, articles, decoding_cfg,
            bart_cfg.get("source_prefix", ""), cfg["max_source_length"], device,
        )
        del bart_model, bart_tokenizer
        torch.cuda.empty_cache() if device.type == "cuda" else None

        # Load and generate with T5
        t5_checkpoint = args.t5_checkpoint or t5_cfg["output_dir"]
        print(f"[error_analysis] Loading T5 from {t5_checkpoint}")
        t5_tokenizer = AutoTokenizer.from_pretrained(t5_checkpoint)
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_checkpoint).to(device)

        print("[error_analysis] Generating T5 summaries...")
        t5_summaries = generate_summaries_simple(
            t5_model, t5_tokenizer, articles, decoding_cfg,
            t5_cfg.get("source_prefix", ""), cfg["max_source_length"], device,
        )
        del t5_model, t5_tokenizer

    # Build blinded CSV: randomly assign BART/T5 to Summary A/B per article
    rng2 = random.Random(seed + 1)  # different seed for assignment shuffle
    key_data = []  # mapping from article to which model is A/B

    csv_rows = []
    for i, (art, ref, bart_sum, t5_sum) in enumerate(
        zip(articles, references, bart_summaries, t5_summaries)
    ):
        # Randomly swap A and B
        if rng2.random() < 0.5:
            summary_a, summary_b = bart_sum, t5_sum
            model_a, model_b = "BART", "T5"
        else:
            summary_a, summary_b = t5_sum, bart_sum
            model_a, model_b = "T5", "BART"

        csv_rows.append({
            "article_id": i,
            "article_preview": art[:500],
            "reference": ref,
            "summary_a": summary_a,
            "summary_b": summary_b,
            # Rating columns (to be filled manually)
            "coverage_a": "",
            "coverage_b": "",
            "factual_consistency_a": "",
            "factual_consistency_b": "",
            "readability_a": "",
            "readability_b": "",
            "redundancy_a": "",
            "redundancy_b": "",
            "error_tags_a": "",
            "error_tags_b": "",
        })

        key_data.append({
            "article_id": i,
            "summary_a_model": model_a,
            "summary_b_model": model_b,
        })

    # Save blinded rating sheet
    rating_path = results_dir / "error_analysis_rating_sheet.csv"
    fieldnames = list(csv_rows[0].keys())
    with open(rating_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[error_analysis] Blinded rating sheet saved to {rating_path}")

    # Save key file (hidden — do not look at during rating)
    key_path = results_dir / "error_analysis_key.json"
    with open(key_path, "w") as f:
        json.dump(key_data, f, indent=2)
    print(f"[error_analysis] Key file saved to {key_path}")
    print("[error_analysis] DO NOT look at the key file until ratings are complete.")


def deblind_results(args):
    """De-anonymize ratings and build summary tables."""
    results_dir = Path("results")

    # Load key
    key_path = results_dir / "error_analysis_key.json"
    with open(key_path) as f:
        key_data = json.load(f)
    key_map = {item["article_id"]: item for item in key_data}

    # Load rated CSV
    rating_path = results_dir / "error_analysis_rating_sheet.csv"
    with open(rating_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Aggregate scores per model
    metrics = ["coverage", "factual_consistency", "readability", "redundancy"]
    model_scores = {"BART": {m: [] for m in metrics}, "T5": {m: [] for m in metrics}}
    model_errors = {"BART": [], "T5": []}

    for row in rows:
        aid = int(row["article_id"])
        key = key_map[aid]

        for suffix, model_name in [("a", key["summary_a_model"]), ("b", key["summary_b_model"])]:
            for m in metrics:
                val = row.get(f"{m}_{suffix}", "").strip()
                if val:
                    try:
                        model_scores[model_name][m].append(float(val))
                    except ValueError:
                        pass

            error_tags = row.get(f"error_tags_{suffix}", "").strip()
            if error_tags:
                tags = [t.strip() for t in error_tags.split(",") if t.strip()]
                model_errors[model_name].extend(tags)

    # Print summary
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS RESULTS (De-anonymized)")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'BART (mean)':<15} {'T5 (mean)':<15}")
    print("-" * 55)
    for m in metrics:
        bart_vals = model_scores["BART"][m]
        t5_vals = model_scores["T5"][m]
        bart_mean = f"{sum(bart_vals)/len(bart_vals):.2f}" if bart_vals else "N/A"
        t5_mean = f"{sum(t5_vals)/len(t5_vals):.2f}" if t5_vals else "N/A"
        print(f"{m:<25} {bart_mean:<15} {t5_mean:<15}")

    print(f"\n{'Error Tag':<30} {'BART (count)':<15} {'T5 (count)':<15}")
    print("-" * 60)
    all_tags = set(model_errors["BART"] + model_errors["T5"])
    for tag in sorted(all_tags):
        bart_count = model_errors["BART"].count(tag)
        t5_count = model_errors["T5"].count(tag)
        print(f"{tag:<30} {bart_count:<15} {t5_count:<15}")

    # Save summary
    summary = {
        "mean_scores": {
            model: {m: (sum(vals)/len(vals) if vals else None) for m, vals in scores.items()}
            for model, scores in model_scores.items()
        },
        "error_counts": {
            model: {tag: errors.count(tag) for tag in set(errors)}
            for model, errors in model_errors.items()
        },
        "num_articles_rated": len(rows),
        "note": "Single rater, small sample. See report for caveats.",
    }
    summary_path = results_dir / "error_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[error_analysis] Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Error analysis: blinded rating sheet generation and de-anonymization."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true", help="Generate blinded sheet.")
    group.add_argument("--deblind", action="store_true", help="De-anonymize and summarize.")
    parser.add_argument("--smoke", action="store_true", help="Use 5 articles instead of 25.")
    parser.add_argument("--bart-checkpoint", type=str, default=None)
    parser.add_argument("--t5-checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.generate:
        generate_blinded_sheet(args)
    else:
        deblind_results(args)


if __name__ == "__main__":
    main()
