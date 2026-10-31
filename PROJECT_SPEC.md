# PROJECT_SPEC: BART vs. T5 for Abstractive Summarization

This file is the single source of truth for the project. If anything in a chat message conflicts with this file, ask before deviating. If a decision here needs to change, update this file first.

## 1. Project Overview

- **Title:** Comparative Evaluation of BART and T5 for Abstractive Text Summarization
- **Course:** CSCI 390, Dr. Qi Li, Fisk University
- **Dataset:** CNN/DailyMail, non-anonymized (Hugging Face `abisee/cnn_dailymail`, config `3.0.0`)
- **Expected split sizes** (verify in code; report any mismatch): train 287,113 / validation 13,368 / test 11,490
- **Goal:** a reproducible pipeline that (1) measures pretrained baselines, (2) fine-tunes BART and T5 under comparable conditions, (3) evaluates them with ROUGE, efficiency measurements, and error analysis, and (4) serves a Gradio demo comparing both models side by side.

### Research questions
1. How do fine-tuned BART and T5 compare in summary quality on CNN/DailyMail?
2. How well do generated summaries preserve important information?
3. How do summary length and decoding parameters affect informativeness, redundancy, and readability?
4. How do the models compare in inference time, memory, and suitability for an interactive app?

Do not assume either model wins in advance.

## 2. Constraints

- **Team:** one person.
- **Dates:** report due **Nov 7, 2026**; 10-minute live demo presentation **Nov 9, 2026**. Project started Oct 26, 2026.
- **Hardware:** MacBook M4 (8-core GPU, 16 GB RAM, PyTorch `mps`); free Kaggle GPU (T4-class, 16 GB, 30 GPU hrs/week quota, none used yet); Colab with Google AI Pro as backup (check remaining compute units and available GPUs in Colab). No other paid compute.
- **Compute is not the bottleneck; the calendar is.** Avoid wasting days on failed runs: always smoke-test on the Mac first.
- **Mac is for:** development, tiny smoke tests, data inspection, error analysis, running the Gradio demo, latency measurement.
- **Kaggle/Colab are for:** baselines, real fine-tuning, bulk generation and evaluation.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` on the Mac. No fp16 on the Mac.
- All scripts must auto-detect device (cuda, then mps, then cpu) and run both as CLI scripts and in a Kaggle notebook.
- Free sessions disconnect: save checkpoints every ~200-300 steps and make training resumable.

## 3. Scope Decisions (already made)

| Item | Decision |
|---|---|
| Models | `facebook/bart-base` (~140M) and `t5-base` (~220M) by default, to be confirmed by the Oct 29 smoke test (closer in size than t5-small, so a fairer comparison). Fall back to `t5-small` (~60M) if t5-base is too slow or runs out of memory in fp32; if so, the report must state the size mismatch. The T5 checkpoint is a config option. |
| Max source length | 512 tokens, both models (documented deviation from the proposal's 1,024 due to compute) |
| Max target length | 128 tokens, both models |
| Training data | Fixed random subset of 10k-15k articles (final size set after the smoke test), 1 epoch |
| Validation data | Fixed subset of 1,000 articles |
| Test data | Fixed subset of 1,000 articles (500 if generation is too slow) |
| Subset handling | Save indices to `data/indices_{train,val,test}.json`; subsets never change; no overlap between train and val/test |
| Effective batch size | 32 for both models (e.g. per-device 4 x grad accumulation 8; adjust per-device size to fit memory) |
| Optimizer | AdamW, weight decay 0.01, linear warmup then linear decay |
| Warmup | About 10% of total steps (about 375 total steps at 12k examples and batch 32, so the proposal's 500-step warmup is not usable) |
| Learning rate | BART 5e-5. T5 starts at 1e-4 (T5 tolerates up to ~3e-4). If time permits, run a tiny LR check and document it. Do not silently favor one model. |
| Precision | BART: fp16 OK on Kaggle. T5: **fp32 only** (fp16 gives NaN losses; T4 lacks good bf16). No fp16 on Mac. |
| Input format | T5: prefix `summarize: `. BART: plain text with its own tokenizer. |
| Seed | One seed (42), recorded. Do not claim ROUGE differences below ~0.5 points are meaningful. |
| **Dropped unless requested** | BERTScore, multiple seeds, full 11,490-example test set, second human rater, training from scratch |

## 4. Fixed Decoding Configuration

Shared by both models, stored in `configs/decoding.yaml`, never changed mid-project:

```yaml
num_beams: 4
max_length: 128
min_length: 30
length_penalty: 2.0
no_repeat_ngram_size: 3
early_stopping: true
```

These are adapted from the common 142/56 values because they conflict with the 128-token target limit. Document this in the report.

**Decoding ablation (stretch goal):** 3 settings on 300 test articles, changing one thing at a time (e.g. beams 1 vs 4; short vs default vs long max length).

## 5. Experiments

### 5.1 Baselines (same 500 test articles for both models)
- Pretrained `facebook/bart-base`, no fine-tuning. Expect weak output; that is fine and shows what fine-tuning adds.
- Pretrained T5 (`t5-base` by default, same checkpoint that will be fine-tuned), zero-shot with the `summarize: ` prefix.
- **Known asymmetry:** original T5 checkpoints were multi-task pretrained with CNN/DailyMail included, so T5 will look strong at baseline. Explain this in the report. The comparison that matters most is after both are fine-tuned the same way.
- **Do NOT use `facebook/bart-large-cnn` as a baseline** (already fine-tuned on this dataset). It may appear only as a clearly labeled optional "reference ceiling" if time allows.

### 5.2 Fine-tuned models
Fine-tune BART and T5 with `Seq2SeqTrainer`, comparable settings, validation each ~100-200 steps, best checkpoint kept by validation ROUGE-L or loss (pick one, document it).

## 6. Evaluation

1. **ROUGE-1/2/L:** Hugging Face `evaluate` + `rouge_score`, same implementation and settings for all models (F-measure, stemming on). Save per-example outputs and aggregate scores to `results/` as CSV/JSON.
2. **Efficiency:**
   - Mean inference latency per article over 100 articles, after warm-up
   - Approximate peak memory
   - Average generated length in tokens
   - Training time
   - Measure latency for both models on the SAME machine (the Mac) and record hardware and generation settings.
3. **Error analysis:**
   - Fixed-seed sample of 25 test articles
   - Generate a CSV with shuffled, anonymized columns "Summary A" / "Summary B", plus a separate hidden key file mapping A/B to models
   - Rubric (1-5): coverage, factual consistency, readability, redundancy
   - Error tags: wrong entity, wrong number/date, hallucinated claim, omission of key fact, repetition
   - A notebook that, after ratings are filled, de-anonymizes and builds summary tables (mean scores and error counts per model)
   - Report must note: one rater, small sample
4. **Data statistics:** percentage of articles exceeding 512 tokens (truncation limitation); article and summary length histograms.

## 7. Gradio Demo

- Runs locally on the Mac using the fine-tuned checkpoints. Load models sequentially or lazily to avoid memory problems.
- Features:
  - Dropdown of 3-5 pre-loaded test articles with their reference summaries (safer for a live talk)
  - Text box for pasting a custom English article
  - Side-by-side BART vs T5 summaries
  - Sliders for min/max length and num_beams (within supported limits)
  - Output stats: summary length in tokens, generation latency
  - Visible disclaimer: summaries may omit context or contain unsupported claims
- Must degrade gracefully (clear message, no crash) if a checkpoint is missing or input is empty.
- Designed for a 10-minute talk: show an example, compare outputs, adjust one parameter, discuss one omission / redundancy / factual-error example.
- Choose demo articles in advance: ideally one where BART wins, one where T5 wins, one with a clear factual error.
- Prepare a backup screen recording in case the live demo fails.

## 8. Repo Layout

```
project/
  PROJECT_SPEC.md        # this file
  README.md              # setup + exact commands to reproduce everything
  requirements.txt       # pinned versions
  configs/               # base.yaml, bart.yaml, t5.yaml, decoding.yaml
  data/                  # indices_train.json, indices_val.json, indices_test.json
  src/
    utils.py             # device detection, seeding, config loading
    data.py              # load dataset, create/save fixed subsets, tokenization
    baseline.py          # zero-shot generation + ROUGE for pretrained models
    train.py             # Seq2SeqTrainer fine-tuning (resumable, checkpoints)
    evaluate.py          # generation with fixed decoding config + ROUGE + efficiency
    error_analysis.py    # builds blinded rating sheet + summary tables
  notebooks/             # data_inspection.ipynb, error_analysis.ipynb, kaggle_run.ipynb
  results/               # metrics JSON/CSV, sample outputs, plots
  app/app.py             # Gradio demo
  report/
```

## 9. Coding Requirements

- Python, PyTorch, Hugging Face Transformers / Datasets / Evaluate, Gradio. Pin versions in `requirements.txt`.
- All hyperparameters live in YAML configs; no magic numbers in scripts.
- Log seed, library versions, checkpoint names, hardware, and training duration to a JSON run record for every run.
- Every script supports a `--smoke` flag (about 50 examples) so the pipeline can be tested on the Mac before spending GPU time.
- Handle `-100` label padding correctly; use `DataCollatorForSeq2Seq`; use `predict_with_generate`.
- Clear comments and docstrings; keep functions short.
- After each phase, run the relevant smoke test and show the result before moving on.
- Never invent results or numbers. Only report what scripts actually produced.

## 10. Schedule

| Date | Goal | Machine |
|---|---|---|
| Mon Oct 26 | Setup, repo, configs, dataset loads, verify split sizes | Mac |
| Tue Oct 27 | Data inspection, tokenization, save fixed subset indices | Mac |
| Wed Oct 28 | Write evaluation script; run baselines on 500 examples | Kaggle/Colab |
| Thu Oct 29 | Smoke fine-tune both models on 1,000 examples; record time per 1,000 steps and peak memory; size the real training set; confirm t5-base or fall back to t5-small | Kaggle |
| Fri Oct 30 | Full BART fine-tuning ("Save & Run All"); build Gradio skeleton; draft report intro/method | Kaggle + Mac |
| Sat Oct 31 | Full T5 fine-tuning (fp32); draft dataset/setup sections | Kaggle + Mac |
| Sun Nov 1 | Buffer: rerun failures | any |
| Mon Nov 2 | Evaluate both fine-tuned models on test subset; efficiency timing | Kaggle + Mac |
| Tue Nov 3 | Decoding ablation (if on schedule) | Kaggle |
| Wed Nov 4 | Error analysis; choose 3 demo articles | Mac |
| Thu Nov 5 | Finish and test demo end to end; record backup screen recording | Mac |
| Fri Nov 6 | Write results, error analysis, limitations, conclusion | Mac |
| Sat Nov 7 | Proofread, verify README reproduces results, **SUBMIT** | Mac |
| Sun Nov 8 | Write and rehearse the 10-minute script twice | - |
| Mon Nov 9 | **Present** | - |

### Cut order if behind
1. Drop decoding ablation
2. Reduce error analysis to 20 articles
3. Shrink test sample to 500

**Never cut:** baselines, both fine-tuned models, ROUGE table, demo.

### If ahead of schedule (spend spare time in this order)
1. Larger training subset (e.g. 20k-30k articles)
2. Quick learning-rate check for T5
3. Decoding ablation
4. Optional `bart-large-cnn` "reference ceiling" row (clearly labeled)

## 11. Report Support

Draft in Markdown: introduction, dataset and preprocessing, experimental design, training configuration, results tables, error analysis, efficiency, challenges and limitations, conclusion.

Be honest about limitations:
- Small models (base/small, not large)
- Training on a subset, 1 epoch
- 512-token truncation
- Single seed
- Single rater, small sample
- ROUGE does not measure factual correctness
- Results limited to English CNN/DailyMail news
- T5 baseline advantage from multi-task pretraining
- Size mismatch between the two models (only if the t5-small fallback is used)

## 12. Open Questions (ask the user before deciding)

- Mac RAM (fill in `[X]` in section 2; find it with `sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}'`)?
- Colab (Google AI Pro): which GPUs are available and how many compute units remain?
- After the Oct 29 smoke test: final training subset size, and confirm t5-base or fall back to t5-small?

Resolved: Kaggle quota is 30 GPU hrs/week with none used; Mac is an M4 with an 8-core GPU.
