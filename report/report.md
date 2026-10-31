# Comparative Evaluation of BART and T5 for Abstractive Text Summarization

**Course:** CSCI 390 (Senior Capstone / Advanced NLP)  
**Institution:** Fisk University  
**Instructor:** Dr. Qi Li  
**Author:** Aayush Khanal  
**Date:** Fall 2026  

---

## Abstract

Abstractive text summarization requires condensing long documents into coherent, fluent, and factually accurate summaries while generating novel phrasing not present in the source. This project investigates and compares two prominent transformer architectures—Bidirectional and Auto-Regressive Transformers (**BART-base**, ~140M parameters) and the Text-to-Text Transfer Transformer (**T5-base**, ~220M parameters)—on the non-anonymized CNN/DailyMail benchmark. We design a rigorous, reproducible experimental pipeline that evaluates zero-shot pre-trained baselines, performs fine-tuning under strictly identical hyperparameters and fixed data subsets (12,000 training, 1,000 validation, and 1,000 test articles), and benchmarks both models using automated ROUGE metrics, inference latency, peak memory consumption, and a double-blind qualitative error analysis on 25 test samples. Finally, we implement an interactive side-by-side Gradio application designed for live demonstration and latency profiling.

---

## 1. Introduction & Research Questions

Modern abstractive summarization benchmarks are dominated by encoder-decoder transformer models pre-trained on massive text corpora via self-supervised denoising objectives. However, architectural paradigms differ significantly in their pre-training objectives, tokenization schemes, and handling of task conditioning:
* **BART** combines a bidirectional encoder (similar to BERT) with an autoregressive sequence-to-sequence decoder (similar to GPT), pre-trained to reconstruct corrupted text using token masking, deletion, and document rotation.
* **T5** treats every NLP task as a string-to-string transformation, pre-trained on C4 with masked span prediction, and conditions generation using explicit natural language task prefixes (e.g., `summarize: `).

### Research Questions
1. **RQ1 (Summarization Quality):** How do fine-tuned BART-base and T5-base compare in abstractive summary quality when evaluated via standardized ROUGE-1, ROUGE-2, and ROUGE-L metrics under strictly matched training regimes?
2. **RQ2 (Information Preservation & Factual Consistency):** How faithfully do the generated summaries preserve core source facts, and what error patterns (e.g., entity hallucination, date distortion, or omissions) emerge most frequently?
3. **RQ3 (Decoding Sensitivity):** How do beam search constraints (beam size, length penalties, and n-gram blocking) influence summary informativeness, redundancy, and repetition?
4. **RQ4 (Computational Efficiency & Interactive Viability):** How do the models compare in terms of inference latency, peak memory footprint, and feasibility for real-time web deployment?

---

## 2. Dataset & Preprocessing Pipeline

### 2.1 Benchmark Selection: CNN/DailyMail 3.0.0
We evaluate on the standard non-anonymized Hugging Face release of `abisee/cnn_dailymail` (version `3.0.0`). The canonical split distribution is:
* **Full Train:** 287,113 document-highlight pairs
* **Full Validation:** 13,368 document-highlight pairs
* **Full Test:** 11,490 document-highlight pairs

### 2.2 Controlled Subsetting & Reproducibility
Given computational resource constraints (MacBook M4 for local development and Kaggle T4 cloud GPU for full training), training on the entire 287k corpus is computationally prohibitive within our calendar schedule. To eliminate selection bias and prevent data leakage:
1. Subsets were randomly sampled once using seed `42` and serialized to immutable index files:
   * `data/indices_train.json` ($N = 12,000$)
   * `data/indices_validation.json` ($N = 1,000$)
   * `data/indices_test.json` ($N = 1,000$)
2. Validation and test sets are strictly disjoint from the training partition.
3. All baseline, fine-tuning, and evaluation scripts consume these exact index sets.

### 2.3 Tokenization and Truncation Trade-offs
* **Max Source Length:** 512 tokens.
* **Max Target Length:** 128 tokens.
* **Padding & Masking:** Padded labels are assigned `-100` so that pad tokens are ignored by the cross-entropy loss function via `DataCollatorForSeq2Seq`.
* **Truncation Analysis:** In news summarization, news articles often exceed 700–1,000 words. Truncating at 512 tokens forces both models to summarize predominantly from the lead paragraphs (the "inverted pyramid" style typical of journalistic writing).

---

## 3. Experimental Setup & Training Configuration

### 3.1 Model Specifications
* **BART-base (`facebook/bart-base`):** 6 encoder layers, 6 decoder layers, 12 attention heads, hidden dimension 768 (~140M parameters). Byte-Pair Encoding (BPE) vocabulary of 50,265 tokens.
* **T5-base (`t5-base`):** 12 encoder layers, 12 decoder layers, 12 attention heads, hidden dimension 768 (~220M parameters). SentencePiece vocabulary of 32,128 tokens. Requires input prefix `summarize: `.

### 3.2 Hyperparameter Parity
To guarantee a fair comparison, both models share identical training mechanics where possible:

| Hyperparameter | BART-base | T5-base | Rationale |
|---|---|---|---|
| **Training Articles** | 12,000 | 12,000 | Fixed subset from `indices_train.json` |
| **Epochs** | 1.0 (~375 steps) | 1.0 (~375 steps) | Avoids overfitting on small subsets |
| **Effective Batch Size** | 32 (4 per device × 8 grad accum) | 32 (4 per device × 8 grad accum) | Matched gradient dynamics |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$) | AdamW ($\beta_1=0.9, \beta_2=0.999$) | Standard transformer optimizer |
| **Weight Decay** | 0.01 | 0.01 | L2 regularization |
| **Learning Rate** | $5 \times 10^{-5}$ | $1 \times 10^{-4}$ | Recommended baseline ranges for BART/T5 |
| **Warmup Ratio** | ~10% (38 steps) | ~10% (38 steps) | Linear warmup followed by linear decay |
| **Precision** | fp16 (on CUDA) | **fp32** | T5 exhibits numerical instability (NaN loss) in fp16 on T4 GPUs |
| **Evaluation / Checkpointing** | Every 100 steps | Every 100 steps | Best checkpoint selected via validation loss |

### 3.3 Locked Decoding Configuration
During inference and evaluation, decoding parameters are locked in `configs/decoding.yaml`:
* **Beam search:** `num_beams = 4`
* **Length bounds:** `min_length = 30`, `max_length = 128`
* **Length penalty:** `length_penalty = 2.0`
* **Repetition control:** `no_repeat_ngram_size = 3`
* **Early stopping:** `early_stopping = True`

---

## 4. Empirical Results & Findings

### 4.1 Zero-Shot Pre-Trained Baselines
Before fine-tuning, both pre-trained models were evaluated in zero-shot mode on the first 50 test samples using Apple Silicon MPS acceleration to measure baseline generation capacity:

| Model | Checkpoint Status | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Peak Memory |
|---|---|---|---|---|---|---|
| **BART-base** | Zero-shot (Untrained) | 27.87% | 9.95% | 17.70% | 21.27% | 1,916 MB |
| **T5-base** | Zero-shot (`summarize: `) | **30.43%** | **10.40%** | **21.55%** | **24.57%** | **1,415 MB** |

> **Critical Empirical Note on Baseline Asymmetry:**  
> T5-base demonstrates a noticeable advantage in zero-shot ROUGE scores prior to fine-tuning. This is an expected artifact of T5's pre-training regimen: the original C4 pre-training corpus for T5 included multi-task supervised datasets, including CNN/DailyMail. Conversely, `facebook/bart-base` is a pure denoising autoencoder with zero task-specific supervised training. The decisive comparison occurs after both are fine-tuned on the identical 12,000-article partition.

### 4.2 Fine-Tuned Model Performance (Test Partition $N=1,000$)
Both models were fine-tuned for 1.0 epoch (188 optimizer steps at effective batch size 32) on the identical 12,000-article training partition and evaluated on the 1,000-article test set under locked beam decoding (`num_beams: 4`, `min_length: 30`, `max_length: 128`):

| Model | Fine-Tuning Steps | Train Loss | Val Loss | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Mean Latency (s/art) | Mean Tokens | Peak Memory |
|---|---|---|---|---|---|---|---|---|---|---|
| **BART-base** | 188 steps (1 epoch) | 2.4891 | 1.8937 | **40.23%** | **17.71%** | **27.06%** | **37.01%** | **0.309s** (±0.060s) | 83.1 | 1,878 MB |
| **T5-base** | 188 steps (1 epoch) | 1.6706 | 1.5300 | **42.88%** | **20.24%** | **30.12%** | **36.67%** | **0.569s** (±0.140s) | 100.3 | 1,898 MB |

### 4.3 Baseline vs. Fine-Tuning Gain Analysis
* **BART-base:** Increased ROUGE-1 from 27.87% to 40.23% (**+12.36% absolute gain**), ROUGE-2 from 9.95% to 17.71% (**+7.76% gain**), and ROUGE-L from 17.70% to 27.06% (**+9.36% gain**).
* **T5-base:** Increased ROUGE-1 from 30.43% to 42.88% (**+12.45% absolute gain**), ROUGE-2 from 10.40% to 20.24% (**+9.84% gain**), and ROUGE-L from 21.55% to 30.12% (**+8.57% gain**).
* **Synthesis:** Both models exhibit remarkably comparable fine-tuning leaps (~12.4% gain in ROUGE-1). T5-base maintains a slight lead of ~2.6 ROUGE points in overall overlap, largely inherited from its initial pre-training advantage.

---

## 5. Computational Efficiency & Hardware Profiling

To answer **RQ4**, efficiency was benchmarked on NVIDIA Tesla T4 (16 GB VRAM, CUDA 13.0):
1. **Training Throughput & Convergence:**
   * **BART-base** completed full training in **11 minutes 30 seconds** (17.38 samples/sec) using fp16 mixed precision.
   * **T5-base** completed training in **21 minutes 52 seconds** (9.14 samples/sec) due to fp32 precision requirements.
2. **Inference Latency & Interactivity:**
   * Over 100 consecutive test evaluations, **BART-base achieved an average latency of 0.309s per article**, running **1.84× faster** than T5-base (**0.569s per article**).
   * BART's shallower 6-layer decoder and BPE tokenization yield substantially faster token generation during beam exploration, making BART the superior architecture for interactive, latency-sensitive web deployment.
3. **Generated Summary Length:**
   * BART generated an average of **83.1 tokens** per summary, producing punchy, concise journalistic summaries.
   * T5 generated an average of **100.3 tokens** per summary, favoring complete compound clauses.

---

## 6. Qualitative Error Analysis

Automated ROUGE scores capture lexical n-gram overlap but fail to evaluate factual faithfulness, hallucination, or discourse coherence. To address **RQ2**, we implemented a double-blind human rating protocol (`src/error_analysis.py`).

### 6.1 Evaluation Methodology
* A fixed random sample of 25 test articles was extracted (`seed=42`).
* Summaries from fine-tuned BART and T5 were shuffled into anonymized columns (`Summary A` vs. `Summary B`), with model identities stored in a secure separate key file.
* Each summary was evaluated on a 1–5 Likert scale across four dimensions:
  1. **Coverage:** Does the summary capture the main news event?
  2. **Factual Consistency:** Is all information directly verifiable from the source article?
  3. **Readability & Fluency:** Is the text grammatical, natural, and well-structured?
  4. **Conciseness / Redundancy:** Is the summary free of repetitive phrasing and filler content?

### 6.2 Error Taxonomy
Each sample was tagged for specific failure modes:
* **Wrong Entity (WE):** Attributing actions to the wrong person, organization, or location.
* **Numeric / Date Distortion (ND):** Misreporting quantities, ages, financial figures, or dates.
* **Extrinsic Hallucination (EH):** Introducing facts not grounded anywhere in the source text.
* **Critical Fact Omission (OM):** Omitting the defining resolution or consequence of the news event.
* **Repetition Loop (REP):** Cycling across duplicate phrases despite n-gram blocking.

---

## 7. Interactive Demonstration System

To satisfy course presentation requirements and test interactive viability, we developed an end-to-end Gradio web application (`app/app.py`).

### 7.1 Architecture & Features
* **Curated Test Presets:** Dropdown containing 3–5 representative CNN/DailyMail articles alongside ground-truth human reference highlights.
* **Live Side-by-Side Comparison:** Generates and displays BART and T5 summaries simultaneously in formatted response cards.
* **Dynamic Parameter Sliders:** Real-time adjustment of `num_beams` (1 to 8), `min_length` (10 to 80), and `max_length` (60 to 200).
* **Latency & Token Telemetry:** Displays instantaneous generation time (in seconds) and generated token length for each model.
* **Graceful Degradation:** Detects missing local checkpoints and prompts the user with informative guidance rather than crashing.

---

## 8. Limitations & Threats to Validity

In adherence to scientific rigor, several limitations must be noted:
1. **Model Scale:** Due to memory constraints, experiments were conducted on base architectures (~140M–220M parameters) rather than large counterparts (~400M–770M parameters).
2. **512-Token Truncation:** Long articles were truncated to 512 input tokens. Summarization performance is therefore biased toward the opening sections of articles.
3. **Subset Training (1 Epoch):** Training on 12,000 examples for 1 epoch provides strong fine-tuning signal while fitting within compute budgets, but does not represent asymptotic model convergence.
4. **Single Evaluator for Error Analysis:** Qualitative error analysis was conducted by a single human annotator on 25 articles. While double-blinded, inter-annotator agreement (e.g., Cohen's Kappa) could not be calculated.
5. **Metric Constraints:** ROUGE scores reward lexical recall against a single human reference, occasionally penalizing valid abstractive paraphrasing.

---

## 9. Conclusion

This capstone project establishes a complete, reproducible empirical pipeline comparing BART and T5 for abstractive summarization. By holding dataset partitions, evaluation implementations, and decoding strategies constant, our results isolate genuine architectural and objective differences between the two models. The accompanying Gradio web application bridges experimental results with interactive demonstration.

---

## References

1. Lewis, M., et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.* ACL 2020.
2. Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR 2020.
3. Hermann, K. M., et al. (2015). *Teaching Machines to Read and Comprehend.* NeurIPS 2015. (CNN/DailyMail Dataset).
4. Lin, C. Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL Workshop.
5. Wolf, T., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing.* EMNLP 2020.
