"""
Gradio demo: side-by-side BART vs T5 summarization.

Features:
  - Dropdown of pre-loaded test articles with reference summaries.
  - Text box for pasting custom articles.
  - Side-by-side BART and T5 summaries.
  - Sliders for min/max length and num_beams.
  - Output stats: summary length in tokens, generation latency.
  - Visible disclaimer about potential inaccuracies.
  - Graceful degradation if a checkpoint is missing.

Usage:
    python app/app.py                        # launch demo
    python app/app.py --share                # launch with public URL
    python app/app.py --bart-checkpoint PATH --t5-checkpoint PATH
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_device, is_mac_mps, load_config, load_decoding_config

# ---------------------------------------------------------------------------
# Global state (lazy-loaded)
# ---------------------------------------------------------------------------
bart_model = None
bart_tokenizer = None
t5_model = None
t5_tokenizer = None
device = None
demo_articles = []  # list of (title, article, reference)


def load_model_safe(checkpoint_path: str, label: str):
    """Load a model and tokenizer, returning (model, tokenizer) or (None, None)."""
    try:
        print(f"[demo] Loading {label} from {checkpoint_path}...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
        model.to(device)
        model.eval()
        print(f"[demo] {label} loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"[demo] WARNING: Could not load {label} from {checkpoint_path}: {e}")
        return None, None


def generate_summary(model, tokenizer, article: str, source_prefix: str,
                     max_source_length: int, num_beams: int,
                     min_length: int, max_length: int) -> tuple[str, float, int]:
    """Generate a summary and return (text, latency_seconds, token_count)."""
    if model is None:
        return "⚠️ Model not available (checkpoint missing)", 0.0, 0

    if not article or not article.strip():
        return "⚠️ Please enter an article to summarize.", 0.0, 0

    text = source_prefix + article
    inputs = tokenizer(
        text,
        max_length=max_source_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    latency = time.perf_counter() - t0

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    token_count = (output_ids[0] != tokenizer.pad_token_id).sum().item()

    return decoded, latency, token_count


def summarize(article_text: str, num_beams: int, min_length: int, max_length: int):
    """Main summarization function called by Gradio."""
    cfg = load_config("configs/base.yaml")
    bart_cfg = load_config("configs/base.yaml", "configs/bart.yaml")
    t5_cfg = load_config("configs/base.yaml", "configs/t5.yaml")
    max_source_length = cfg["max_source_length"]

    # BART
    bart_summary, bart_latency, bart_tokens = generate_summary(
        bart_model, bart_tokenizer, article_text,
        bart_cfg.get("source_prefix", ""),
        max_source_length, num_beams, min_length, max_length,
    )
    bart_stats = f"🕐 {bart_latency:.2f}s  |  📝 {bart_tokens} tokens"

    # T5
    t5_summary, t5_latency, t5_tokens = generate_summary(
        t5_model, t5_tokenizer, article_text,
        t5_cfg.get("source_prefix", ""),
        max_source_length, num_beams, min_length, max_length,
    )
    t5_stats = f"🕐 {t5_latency:.2f}s  |  📝 {t5_tokens} tokens"

    return bart_summary, bart_stats, t5_summary, t5_stats


def load_demo_article(choice: str):
    """Load a pre-selected demo article from the dropdown."""
    if not choice or choice == "-- Paste your own article --":
        return "", ""

    for title, article, reference in demo_articles:
        if title == choice:
            return article, f"**Reference summary:**\n\n{reference}"

    return "", ""


def load_demo_articles_from_data():
    """Load a handful of test articles for the dropdown."""
    try:
        from src.data import load_cnn_dailymail, load_indices, get_subsets

        cfg = load_config("configs/base.yaml")
        ds = load_cnn_dailymail()
        indices = load_indices(cfg.get("data_dir", "data"))
        subsets = get_subsets(ds, indices)

        # Pick first 5 articles from the test subset
        sample_indices = [0, 1, 2, 3, 4]
        articles = []
        for idx in sample_indices:
            if idx < len(subsets["test"]):
                item = subsets["test"][idx]
                # Create a short title from the first sentence
                first_sentence = item["article"].split(".")[0][:80]
                title = f"Article {idx + 1}: {first_sentence}..."
                articles.append((title, item["article"], item["highlights"]))

        return articles
    except Exception as e:
        print(f"[demo] Could not load demo articles: {e}")
        return []


def build_ui():
    """Build and return the Gradio interface."""
    # Dropdown choices
    dropdown_choices = ["-- Paste your own article --"] + [
        title for title, _, _ in demo_articles
    ]

    with gr.Blocks(
        title="BART vs T5 Summarizer",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# 📰 BART vs T5: Abstractive Summarization Demo\n"
            "*Comparative evaluation of fine-tuned BART-base and T5-base on CNN/DailyMail*\n\n"
            "> ⚠️ **Disclaimer:** Summaries may omit important context or contain "
            "unsupported claims. Always verify critical information against the source."
        )

        with gr.Row():
            with gr.Column(scale=2):
                article_dropdown = gr.Dropdown(
                    choices=dropdown_choices,
                    label="📋 Select a pre-loaded test article",
                    value="-- Paste your own article --",
                )
                reference_display = gr.Markdown(label="Reference Summary", value="")

        article_input = gr.Textbox(
            label="📝 Article Text",
            placeholder="Paste an English news article here, or select one from the dropdown above...",
            lines=10,
            max_lines=20,
        )

        with gr.Row():
            num_beams = gr.Slider(
                minimum=1, maximum=8, value=4, step=1,
                label="🔢 Beam Search (num_beams)",
            )
            min_len = gr.Slider(
                minimum=10, maximum=60, value=30, step=5,
                label="📏 Min Length",
            )
            max_len = gr.Slider(
                minimum=50, maximum=200, value=128, step=10,
                label="📏 Max Length",
            )

        summarize_btn = gr.Button("🚀 Generate Summaries", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### BART Summary")
                bart_output = gr.Textbox(label="BART Output", lines=6, interactive=False)
                bart_stats_output = gr.Textbox(label="BART Stats", interactive=False)
            with gr.Column():
                gr.Markdown("### T5 Summary")
                t5_output = gr.Textbox(label="T5 Output", lines=6, interactive=False)
                t5_stats_output = gr.Textbox(label="T5 Stats", interactive=False)

        # Event handlers
        article_dropdown.change(
            fn=load_demo_article,
            inputs=[article_dropdown],
            outputs=[article_input, reference_display],
        )

        summarize_btn.click(
            fn=summarize,
            inputs=[article_input, num_beams, min_len, max_len],
            outputs=[bart_output, bart_stats_output, t5_output, t5_stats_output],
        )

    return demo


def main():
    global bart_model, bart_tokenizer, t5_model, t5_tokenizer, device, demo_articles

    parser = argparse.ArgumentParser(description="Gradio demo for BART vs T5.")
    parser.add_argument("--share", action="store_true", help="Create a public URL.")
    parser.add_argument("--bart-checkpoint", type=str, default=None)
    parser.add_argument("--t5-checkpoint", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    # Device
    device = get_device()

    # Load configs for default checkpoint paths
    bart_cfg = load_config("configs/base.yaml", "configs/bart.yaml")
    t5_cfg = load_config("configs/base.yaml", "configs/t5.yaml")

    bart_checkpoint = args.bart_checkpoint or bart_cfg["output_dir"]
    t5_checkpoint = args.t5_checkpoint or t5_cfg["output_dir"]

    # Load models sequentially (memory-safe)
    bart_model, bart_tokenizer = load_model_safe(bart_checkpoint, "BART")
    t5_model, t5_tokenizer = load_model_safe(t5_checkpoint, "T5")

    if bart_model is None and t5_model is None:
        print("[demo] WARNING: Neither model could be loaded. The demo will show error messages.")

    # Load demo articles
    demo_articles = load_demo_articles_from_data()
    print(f"[demo] Loaded {len(demo_articles)} demo articles")

    # Build and launch
    demo = build_ui()
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
