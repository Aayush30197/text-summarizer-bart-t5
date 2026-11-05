# Comparative Evaluation of BART and T5 for Abstractive Text Summarization

**Course:** CSCI 390, Fisk University  
**Instructor:** Dr. Qi Li  
**Dataset:** CNN/DailyMail (non-anonymized `3.0.0`)  

---

## Overview

This project implements a reproducible NLP pipeline to:
1. Measure zero-shot pretrained baselines for `facebook/bart-base` (~140M) and `t5-base` (~220M).
2. Fine-tune both models under strictly matched conditions on a fixed random subset of CNN/DailyMail.
3. Evaluate models quantitatively (ROUGE-1, ROUGE-2, ROUGE-L) and qualitatively (blinded 25-article error analysis).
4. Benchmark inference efficiency (latency and peak memory).
5. Serve an interactive side-by-side Gradio comparison demo.

---

## Repository Structure

```
├── PROJECT_SPEC.md        # Single source of truth for experiment scope & constraints
├── README.md              # Project overview and reproduction guide
├── requirements.txt       # Pinned library dependencies
├── configs/               # Declarative hyperparameters
│   ├── base.yaml          # Global pipeline settings
│   ├── bart.yaml          # BART fine-tuning config
│   ├── t5.yaml            # T5 fine-tuning config
│   └── decoding.yaml      # Locked beam search decoding settings
├── data/                  # Fixed, non-overlapping subset indices (train/val/test)
├── src/                   # Pipeline source code
│   ├── utils.py           # Device detection, seeding, run logging
│   ├── data.py            # CNN/DailyMail loader and tokenizer
│   ├── baseline.py        # Zero-shot baseline evaluation
│   ├── train.py           # Resumable Seq2SeqTrainer fine-tuning
│   ├── eval_model.py      # Checkpoint evaluation & efficiency metrics
│   └── error_analysis.py  # Double-blind human evaluation generator
├── app/
│   └── app.py             # Gradio side-by-side interactive application
├── results/               # Metrics JSON records and generated summaries
└── report/                # Research paper drafts and figures
```

---

## Setup & Reproduction

### 1. Environment Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Baselines (Smoke Test)
```bash
python src/baseline.py --model bart --smoke
python src/baseline.py --model t5 --smoke
```

### 3. Fine-Tuning
```bash
# Smoke test on local hardware:
python src/train.py --model bart --smoke
python src/train.py --model t5 --smoke

# Full training (Kaggle / Colab GPU recommended):
python src/train.py --model bart
python src/train.py --model t5
```

### 4. Interactive Demo
```bash
python app/app.py
```

---

## Key Results (1,000 Test Articles)

| Model | Parameters | Precision | ROUGE-1 | ROUGE-2 | ROUGE-L | Latency (s/art) | Human Coverage (1–5) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BART-base** | ~140M | fp16 | 40.23% | 17.71% | 27.06% | **0.309s** | **4.24** |
| **T5-base** | ~220M | fp32 | **42.88%** | **20.24%** | **30.12%** | 0.569s | 3.88 |

* **Interactive Viability:** BART runs **1.84× faster** during beam search exploration, making it optimal for real-time web deployment.
* **Factual Grounding:** Both models exhibited high factual consistency (>4.6/5.0); dominant error mode was critical fact omission caused by generation length limits.
* **Full Report:** See [`report/report.md`](report/report.md) for the complete academic analysis and [`report/PRESENTATION_SCRIPT.md`](report/PRESENTATION_SCRIPT.md) for the presentation guide.
