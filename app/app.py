"""
Gradio demo: side-by-side BART vs T5 summarization.

Features:
  - Curated dropdown of presentation spotlight articles with pre-computed fine-tuned outputs.
  - Side-by-side BART and T5 summaries with latency and token counters.
  - Interactive sliders for beam search (num_beams), min_length, and max_length.
  - Live model generation when local checkpoints are provided.
  - Instant pre-loaded comparison for live presentation reliability (zero network/RAM lag).
  - Visible disclaimer and academic error analysis notes for class discussion.

Usage:
    python app/app.py                        # launch demo
    python app/app.py --share                # launch with public URL
    python app/app.py --bart-checkpoint PATH --t5-checkpoint PATH
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_device, load_config, load_decoding_config

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
bart_model = None
bart_tokenizer = None
t5_model = None
t5_tokenizer = None
device = None
demo_articles_data = []


def load_demo_articles() -> list[dict]:
    """Load curated spotlight articles from data/demo_articles.json."""
    demo_path = Path("data/demo_articles.json")
    if demo_path.exists():
        try:
            with open(demo_path) as f:
                articles = json.load(f)
            print(f"[demo] Loaded {len(articles)} curated spotlight articles from {demo_path}")
            return articles
        except Exception as e:
            print(f"[demo] Could not read {demo_path}: {e}")

    # Fallback minimal spotlight if file is missing
    return [
        {
            "id": "spotlight_1",
            "title": "Spotlight 1: Cape Town ISIS Recruiter (T5 Wins — BART Hallucination)",
            "analysis": "💡 **Presentation Spotlight (T5 Wins):** BART suffers from extrinsic hallucination by retrieving a spurious headline (*\"Virginia teen accused of being ISIS recruiter\"*). T5 stays strictly grounded.",
            "article": "(CNN)Authorities detained a 15-year-old girl from Cape Town, South Africa, at the city's airport after receiving information she was leaving the country to join ISIS, State Security Minister David Mahlobo said...",
            "reference": "Official: Girl, 15, \"expressed an interest in joining a terrorist group called ISIS\"\nAuthorities detain girl at Cape Town airport, release her into family's care, he says .",
            "bart_summary": "Virginia teen accused of being ISIS recruiter .\nPolice and airport security located the 15-year-old girl at the city's airport after receiving information she was leaving the country to join ISIS, State Security Minister David Mahlobo says .\nOfficials debriefed the girl's family and released her into the family's care .",
            "bart_stats": "🕐 0.31s  |  📝 58 tokens  |  Evaluation: Kaggle T4 GPU",
            "t5_summary": "a 15-year-old girl from Cape Town, South Africa, was detained at Cape Town's airport . Police and airport security located the girl on a Johannesburg-bound flight . It's not clear how the girl was recruited or how the airfare was arranged .",
            "t5_stats": "🕐 0.57s  |  📝 54 tokens  |  Evaluation: Kaggle T4 GPU",
        }
    ]


def load_model_safe(checkpoint_path: str, label: str):
    """Load a model and tokenizer, returning (model, tokenizer) or (None, None)."""
    if not checkpoint_path or not Path(checkpoint_path).exists():
        print(f"[demo] Checkpoint path '{checkpoint_path}' not found on local disk.")
        return None, None

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
        return "⚠️ Model not loaded locally (see note below).", 0.0, 0

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


def load_preset_article(choice: str):
    """Load a pre-selected demo article and its benchmark outputs immediately."""
    if not choice or choice == "-- Paste your own article --":
        return "", "", "", "", "", "", ""

    for item in demo_articles_data:
        if item["title"] == choice:
            ref_md = f"### 📌 Human Reference Summary (Ground Truth)\n{item['reference']}"
            analysis_md = item.get("analysis", "")
            return (
                item["article"],
                ref_md,
                analysis_md,
                item["bart_summary"],
                item["bart_stats"],
                item["t5_summary"],
                item["t5_stats"],
            )

    return "", "", "", "", "", "", ""


def summarize(article_text: str, num_beams: int, min_length: int, max_length: int, selected_choice: str):
    """Main summarization function called when the user clicks Generate."""
    # Check if models are loaded in memory
    models_available = (bart_model is not None and t5_model is not None)

    if models_available:
        cfg = load_config("configs/base.yaml")
        bart_cfg = load_config("configs/base.yaml", "configs/bart.yaml")
        t5_cfg = load_config("configs/base.yaml", "configs/t5.yaml")
        max_source_length = cfg.get("max_source_length", 512)

        # BART
        bart_summary, bart_latency, bart_tokens = generate_summary(
            bart_model, bart_tokenizer, article_text,
            bart_cfg.get("source_prefix", ""),
            max_source_length, num_beams, min_length, max_length,
        )
        bart_stats = f"🕐 {bart_latency:.2f}s (Live on {device})  |  📝 {bart_tokens} tokens"

        # T5
        t5_summary, t5_latency, t5_tokens = generate_summary(
            t5_model, t5_tokenizer, article_text,
            t5_cfg.get("source_prefix", ""),
            max_source_length, num_beams, min_length, max_length,
        )
        t5_stats = f"🕐 {t5_latency:.2f}s (Live on {device})  |  📝 {t5_tokens} tokens"

        return bart_summary, bart_stats, t5_summary, t5_stats

    # If models are not loaded locally, check if user is viewing a preset spotlight
    for item in demo_articles_data:
        if item["title"] == selected_choice and article_text.strip() == item["article"].strip():
            # Return pre-computed benchmark outputs
            return (
                item["bart_summary"],
                f"{item['bart_stats']} (Pre-computed Benchmark)",
                item["t5_summary"],
                f"{item['t5_stats']} (Pre-computed Benchmark)",
            )

    # Fallback for custom text when checkpoints are not loaded
    msg = (
        "ℹ️ Model weights not loaded locally on this machine.\n"
        "To run live inference on custom text, launch with:\n"
        "python app/app.py --bart-checkpoint <path> --t5-checkpoint <path>\n\n"
        "👉 You can select any of the curated spotlight articles from the dropdown above "
        "to view verified fine-tuned model outputs instantly!"
    )
    return msg, "N/A", msg, "N/A"


def build_ui():
    """Build and return the Gradio interface."""
    dropdown_choices = [item["title"] for item in demo_articles_data] + ["-- Paste your own article --"]

    with gr.Blocks(
        title="BART vs T5 Summarizer | CSCI 390",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# 📰 Comparative Evaluation: BART vs. T5 for Abstractive Summarization\n"
            "**CSCI 390 Final Project Demo** | *Fisk University* | Author: **Aayush Khanal**\n\n"
            "> ⚠️ **Disclaimer:** Machine-generated summaries may omit critical context or contain unsupported claims. "
            "Always verify essential facts against the original source text."
        )

        with gr.Row():
            with gr.Column(scale=3):
                article_dropdown = gr.Dropdown(
                    choices=dropdown_choices,
                    label="🎯 Select a Curated Presentation Spotlight Article (or paste custom text below)",
                    value=dropdown_choices[0] if dropdown_choices else "-- Paste your own article --",
                )
                analysis_display = gr.Markdown(
                    value=demo_articles_data[0].get("analysis", "") if demo_articles_data else ""
                )

        with gr.Row():
            with gr.Column(scale=2):
                article_input = gr.Textbox(
                    label="📄 Source Article Text (CNN/DailyMail)",
                    value=demo_articles_data[0]["article"] if demo_articles_data else "",
                    placeholder="Paste an English news article here, or select a spotlight from the dropdown...",
                    lines=8,
                    max_lines=15,
                )
            with gr.Column(scale=1):
                reference_display = gr.Markdown(
                    value=f"### 📌 Human Reference Summary (Ground Truth)\n{demo_articles_data[0]['reference']}" if demo_articles_data else ""
                )

        with gr.Accordion("⚙️ Decoding Hyperparameters (Shared Beam Search Configuration)", open=False):
            with gr.Row():
                num_beams = gr.Slider(
                    minimum=1, maximum=8, value=4, step=1,
                    label="🔢 Beam Search Width (num_beams)",
                    info="Default: 4. Set to 1 for greedy decoding.",
                )
                min_len = gr.Slider(
                    minimum=10, maximum=60, value=30, step=5,
                    label="📏 Minimum Length (tokens)",
                    info="Default: 30 tokens.",
                )
                max_len = gr.Slider(
                    minimum=50, maximum=200, value=128, step=10,
                    label="📏 Maximum Length (tokens)",
                    info="Default: 128 tokens.",
                )

        summarize_btn = gr.Button("🚀 Generate / Refresh Summaries", variant="primary", size="lg")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔵 Fine-Tuned BART-base (~140M Parameters)")
                bart_output = gr.Textbox(
                    label="BART Summary Output",
                    value=demo_articles_data[0]["bart_summary"] if demo_articles_data else "",
                    lines=6,
                    interactive=False,
                )
                bart_stats_output = gr.Textbox(
                    label="BART Telemetry & Latency",
                    value=demo_articles_data[0]["bart_stats"] if demo_articles_data else "",
                    interactive=False,
                )

            with gr.Column():
                gr.Markdown("### 🟢 Fine-Tuned T5-base (~220M Parameters)")
                t5_output = gr.Textbox(
                    label="T5 Summary Output",
                    value=demo_articles_data[0]["t5_summary"] if demo_articles_data else "",
                    lines=6,
                    interactive=False,
                )
                t5_stats_output = gr.Textbox(
                    label="T5 Telemetry & Latency",
                    value=demo_articles_data[0]["t5_stats"] if demo_articles_data else "",
                    interactive=False,
                )

        gr.Markdown(
            "---\n"
            "🔬 **Evaluation Summary (1,000 Test Articles):** BART-base (ROUGE-1: 40.23, Latency: 0.31s) | "
            "T5-base (ROUGE-1: 42.88, Latency: 0.57s) | [GitHub Repository](https://github.com/Aayush30197/text-summarizer-bart-t5)"
        )

        # Event: Dropdown change
        article_dropdown.change(
            fn=load_preset_article,
            inputs=[article_dropdown],
            outputs=[
                article_input,
                reference_display,
                analysis_display,
                bart_output,
                bart_stats_output,
                t5_output,
                t5_stats_output,
            ],
        )

        # Event: Summarize button click
        summarize_btn.click(
            fn=summarize,
            inputs=[article_input, num_beams, min_len, max_len, article_dropdown],
            outputs=[bart_output, bart_stats_output, t5_output, t5_stats_output],
        )

    return demo


def main():
    global bart_model, bart_tokenizer, t5_model, t5_tokenizer, device, demo_articles_data

    parser = argparse.ArgumentParser(description="Gradio demo for BART vs T5.")
    parser.add_argument("--share", action="store_true", help="Create a public URL.")
    parser.add_argument("--bart-checkpoint", type=str, default=None)
    parser.add_argument("--t5-checkpoint", type=str, default=None)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    device = get_device()

    # Load default configs
    bart_cfg = load_config("configs/base.yaml", "configs/bart.yaml")
    t5_cfg = load_config("configs/base.yaml", "configs/t5.yaml")

    bart_checkpoint = args.bart_checkpoint or bart_cfg.get("output_dir")
    t5_checkpoint = args.t5_checkpoint or t5_cfg.get("output_dir")

    # Load models if checkpoints exist
    bart_model, bart_tokenizer = load_model_safe(bart_checkpoint, "BART")
    t5_model, t5_tokenizer = load_model_safe(t5_checkpoint, "T5")

    if bart_model is None and t5_model is None:
        print("[demo] Notice: Running in presentation preset mode with verified benchmark outputs.")

    # Load spotlight articles
    demo_articles_data = load_demo_articles()

    # Build and launch
    demo = build_ui()
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
