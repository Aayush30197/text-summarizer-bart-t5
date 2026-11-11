# Comparative Evaluation of BART and T5 for Abstractive Text Summarization

**Course:** CSCI 390 (Senior Capstone / Advanced NLP)  
**Institution:** Fisk University  
**Instructor:** Dr. Qi Li  
**Author:** Aayush Khanal  
**Date:** Fall 2026  

---

## Abstract

Text summarization is the process of distilling a long document into a concise version that preserves the most critical information. While traditional *extractive* summarization functions like a highlighter—simply copying existing sentences from the original article—modern natural language processing focuses on *abstractive* summarization. Much like a human taking study notes, an abstractive model reads the full document, comprehends the narrative, and synthesizes key points into brand-new, fluent sentences. 

This project investigates and compares two prominent transformer architectures—Bidirectional and Auto-Regressive Transformers (**BART-base**, ~140M parameters) and the Text-to-Text Transfer Transformer (**T5-base**, ~220M parameters)—on the non-anonymized CNN/DailyMail benchmark. We design a rigorous, reproducible experimental pipeline that evaluates zero-shot pre-trained baselines, performs fine-tuning under strictly identical hyperparameters and fixed data subsets (12,000 training, 1,000 validation, and 1,000 test articles), and benchmarks both models using automated ROUGE metrics, inference latency, peak memory consumption, and a double-blind qualitative error analysis on 25 test samples. Finally, we implement an interactive side-by-side Gradio application designed for live demonstration and latency profiling.

---

## 1. Introduction & Research Questions

In everyday information consumption, humans face a deluge of long-form articles, research papers, and news updates. Automating the synthesis of these documents is a foundational objective of Natural Language Processing (NLP). 

### 1.1 The Two Paradigms: Extractive vs. Abstractive Summarization
1. **Extractive Summarization:** Functions like a yellow highlighter in a textbook. It scores sentences within the original text and extracts top-ranked sentences verbatim. While grammatically guaranteed to be correct and computationally cheap, the resulting summaries often read disjointedly and cannot combine facts or synthesize novel phrasing.
2. **Abstractive Summarization:** Functions like a human reader synthesizing notes. The model parses the source text into high-dimensional semantic representations and generates novel sentences token-by-token. This approach produces coherent, natural summaries, but introduces serious technical risks: the model may "hallucinate" untrue facts, misattribute actions, or omit the defining climax of a story.

### 1.2 Architectural Paradigms: BART vs. T5
Modern abstractive summarization benchmarks are dominated by encoder-decoder transformer models pre-trained on massive text corpora via self-supervised objectives:
* **BART (Bidirectional and Auto-Regressive Transformer):** Developed by Meta / Facebook, BART combines a bidirectional encoder (which reads sentences from left-to-right and right-to-left simultaneously, like BERT) with an autoregressive sequence-to-sequence decoder (which generates words one by one, like GPT). BART is pre-trained as a *denoising autoencoder*: researchers intentionally corrupted clean text—erasing words with blanks, deleting tokens, and scrambling sentences—and trained BART to reconstruct the original, clean text. This pre-training makes BART inherently adept at editing, reorganizing, and condensing text.
* **T5 (Text-to-Text Transfer Transformer):** Developed by Google, T5 reframes every natural language processing task—whether translation, sentiment analysis, question answering, or summarization—into a unified text-to-text format. Rather than having a task-specific classification head, T5 is conditioned using explicit natural language prefixes. To trigger news summarization, the model is fed the prompt: `summarize: [article text]`.

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

### 4.3 Head-to-Head Comparison: Before vs. After Fine-Tuning

To evaluate how much each model benefited from fine-tuning on the 12,000 CNN/DailyMail training articles, Table 3 directly pairs the zero-shot baseline metrics against the final fine-tuned checkpoints:

**Table 3: Comprehensive Model Comparison Before and After Fine-Tuning**

| Model Architecture | Metric | Before Fine-Tuning (Pre-Trained Zero-Shot) | After Fine-Tuning (12,000 Articles, 1 Epoch) | Net Absolute Improvement ($\Delta$) | Relative Percentage Gain |
|---|---|:---:|:---:|:---:|:---:|
| **BART-base** (~140M) | **ROUGE-1** | 27.87% | **40.23%** | **+12.36%** | **+44.3%** |
| | **ROUGE-2** | 9.95% | **17.71%** | **+7.76%** | **+78.0%** |
| | **ROUGE-L** | 17.70% | **27.06%** | **+9.36%** | **+52.9%** |
| | **ROUGE-Lsum** | 21.27% | **37.01%** | **+15.74%** | **+74.0%** |
| | **Inference Latency** | 0.352s / article | **0.309s / article** | −0.043s | **1.84× Faster than T5** |
| **T5-base** (~220M) | **ROUGE-1** | 30.43% | **42.88%** | **+12.45%** | **+40.9%** |
| | **ROUGE-2** | 10.40% | **20.24%** | **+9.84%** | **+94.6%** |
| | **ROUGE-L** | 21.55% | **30.12%** | **+8.57%** | **+39.8%** |
| | **ROUGE-Lsum** | 24.57% | **36.67%** | **+12.10%** | **+49.2%** |
| | **Inference Latency** | 0.612s / article | **0.569s / article** | −0.043s | Baseline Latency |

### 4.4 Multi-Epoch Cloud Scaling Study (Google Colab & Kaggle)

To evaluate model convergence beyond the 1-epoch baseline, we scaled fine-tuning across cloud GPU platforms (Google Colab and Kaggle with NVIDIA Tesla T4 16GB GPUs):

1. **BART-base Multi-Epoch Scaling (Google Colab):**
   * **Setup:** 3 full epochs (1,125 optimizer steps at effective batch size 32, fp16 mixed precision).
   * **Empirical Test Results ($N=1,000$ unseen articles):**
     * **ROUGE-1:** **40.86%** (+0.63% gain over 1-epoch baseline of 40.23%)
     * **ROUGE-2:** **18.44%** (+0.73% gain over 1-epoch baseline of 17.71%)
     * **ROUGE-L:** **27.90%** (+0.84% gain over 1-epoch baseline of 27.06%)
     * **ROUGE-Lsum:** **37.95%** (+0.94% gain over 1-epoch baseline of 37.01%)
     * **Inference Latency:** **0.296s** (±0.060s per article, maintaining high throughput)
     * **Peak VRAM:** 1,748.85 MB
   * **Key Observation:** Scaling BART to 3 epochs yields consistent incremental gains across every automated metric without validation degradation, while further extending BART's architectural lead on discourse structure (ROUGE-Lsum reaching 37.95% vs T5's 36.67%).

2. **T5-base Multi-Epoch Scaling (Kaggle):**
   * **Setup:** 3 full epochs (564 optimizer steps, effective batch size 32, fp32 precision).
   * **Convergence Metrics:** Validation loss steadily improved from 3.072 down to **3.0425** at runtime 1:10:32, demonstrating stable multi-epoch learning without numerical instability when fp32 is maintained.

### 4.5 Synthesis of Comparative Findings
1. **Identical Fine-Tuning Trajectory:** Both architectures extracted virtually identical learning gains from the 12,000-article dataset: BART improved by **+12.36 points** in ROUGE-1, while T5 improved by **+12.45 points**. This demonstrates that under identical hyperparameters, learning rates, and batch dynamics, neither model architecture is inherently "faster at learning" summarization; both adapt at nearly identical rates.
2. **Pre-Training Heritage vs. Fine-Tuning Lead:** T5-base's final advantage in ROUGE-1 (42.88% vs 40.23%, a 2.65-point margin) directly tracks its starting zero-shot advantage (30.43% vs 27.87%, a 2.56-point margin). Because T5's pre-training on C4 already exposed the model to multi-task sentence generation, it began the experiment with an inherent head start.
3. **Efficiency Divergence:** While T5 achieves higher lexical overlap, BART-base decisively wins in computational throughput: generating summaries in **0.309s per article (1.84× faster)** and training in **11.5 minutes vs. 21.9 minutes** for T5.
4. **Multi-Epoch Scalability:** The cloud scaling study confirms that BART continues to scale cleanly to **40.86% ROUGE-1** and **37.95% ROUGE-Lsum** with sub-0.30s latency, solidifying BART's viability for high-throughput production environments.

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

### 6.3 Qualitative Findings & Score Breakdown

Following completion of the double-blind human rating protocol across all 25 test articles, the annotations were de-anonymized via `python src/error_analysis.py --deblind`. Table 4 summarizes the average 1–5 ratings, and Table 5 presents the observed failure mode frequencies.

**Table 4: Double-Blind Human Evaluation Results (1–5 Likert Scale, N=25 articles)**
| Metric | BART-base (Mean) | T5-base (Mean) | Δ (T5 − BART) | Qualitative Interpretation |
|---|:---:|:---:|:---:|---|
| **Coverage** | **4.24** | 3.88 | −0.36 | BART more reliably captures lead journalistic event facts. |
| **Factual Consistency** | **4.88** | 4.68 | −0.20 | Both models exhibit high grounding; T5 had slightly more factual confusions. |
| **Readability & Fluency** | 4.32 | **4.40** | +0.08 | T5 produces more syntactically finished, natural sentence structures. |
| **Conciseness (Redundancy)** | 4.00 | **4.16** | +0.16 | T5 summaries exhibited slightly less n-gram re-phrasing. |

**Table 5: Distribution of Qualitative Error Modes across 25 Evaluated Samples**
| Error Category | Code | BART-base (Count) | T5-base (Count) | Primary Root Cause |
|---|:---:|:---:|:---:|---|
| **Critical Fact Omission** | `OM` | 5 | 8 | Truncation of later article paragraphs or conservative generation stopping. |
| **Extrinsic Hallucination** | `EH` | 1 | 3 | Prior language model associations leaking into specialized domains. |
| **Wrong Entity Attribution** | `WE` | 1 | 1 | Co-referent swapping between secondary actors and focal subjects. |
| **Repetition Loop** | `REP` | 0 | 0 | Effectively eliminated in both models via `no_repeat_ngram_size=3`. |

### 6.4 Error Analysis Case Studies

1. **Hallucination of Extrinsic Context (BART, Article 0):**
   * *Article Context:* A 15-year-old girl in Cape Town, South Africa, detained at the airport en route to join ISIS.
   * *BART Generation:* `"Virginia teen accused of being ISIS recruiter . Police and airport security located the 15-year-old girl..."`
   * *Analysis:* BART prepended an ungrounded snippet from another famous news story (a Virginia teen recruiter), demonstrating catastrophic pretraining retrieval interference before continuing with the factually accurate South African story.
2. **Entity & Venue Confusion (T5, Article 7):**
   * *Article Context:* Crystal Palace hosting Manchester City in a Premier League match at Selhurst Park.
   * *T5 Generation:* `"Alan Pardew has refused to rule Manchester City out of the Premier League title race . Crystal Palace host City at the emirates on Monday night ."`
   * *Analysis:* T5 accurately extracted the manager quotes and stakes, but hallucinated `"the emirates"` (Arsenal's stadium) as the host venue. This underscores the classic hazard of seq2seq models relying on high-frequency semantic collocates instead of grounded context.
3. **Sentence Truncation and Discourse Boundaries:**
   * In Article 6 (dog saving hunter) and Article 19 (car fire rescue), BART hit the 128-token limit mid-clause (`"...where he was listed in guarded"`), impacting its readability score (4.32 vs 4.40). T5 generally adhered more strictly to full-stop boundaries before exhausting length penalties.

---

## 7. Interactive Demonstration System

To satisfy course presentation requirements and test interactive viability, we developed an end-to-end Gradio web application (`app/app.py`).

### 7.1 Architecture & Features
* **Curated Test Spotlights:** Dropdown containing 4 representative case studies (`data/demo_articles.json`) loaded directly with verified fine-tuned predictions, ground-truth human references, and analytical commentary notes.
* **Instant Offline Reliability:** Guarantees zero-latency comparison during live presentation without requiring local multi-gigabyte checkpoints or internet access.
* **Live Side-by-Side Comparison:** Evaluates custom text inputs when local model checkpoints are supplied (`--bart-checkpoint`, `--t5-checkpoint`) with Apple Silicon MPS acceleration.
* **Dynamic Parameter Sliders:** Real-time adjustment of `num_beams` (1 to 8), `min_length` (10 to 60), and `max_length` (50 to 200).
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

## 9. Conclusion & Synthesis of Research Questions

This capstone project establishes a complete, reproducible empirical pipeline comparing BART and T5 for abstractive summarization on CNN/DailyMail. By holding dataset partitions, evaluation implementations, and decoding strategies constant, our experiments yield decisive answers to the core research questions:

1. **RQ1 (Summarization Quality):** Both models achieve substantial and comparable improvements of **+12.4% absolute gain in ROUGE-1** over their untrained baselines. T5-base maintains a slight lead in lexical overlap (42.88% vs. 40.23% ROUGE-1, 30.12% vs. 27.06% ROUGE-L), primarily reflecting its multi-task pre-training corpus.
2. **RQ2 (Information Preservation & Factuality):** In double-blind human evaluation, **BART-base achieved higher coverage (4.24 vs. 3.88)** and **higher factual consistency (4.88 vs. 4.68)**, demonstrating superior retention of core journalistic leads. The primary failure mode across both models was **critical fact omission (`OM`)** caused by length caps, while explicit hallucination (`EH`) was rare (1 occurrence in BART, 3 in T5).
3. **RQ3 (Decoding Dynamics):** Frozen beam search with `num_beams=4` and `no_repeat_ngram_size=3` successfully eliminated degenerate repetition loops (`REP = 0`) across both architectures, with T5 generating longer, more compound sentences (100.3 vs. 83.1 tokens).
4. **RQ4 (Computational Efficiency & Deployment Viability):** **BART-base is decisively superior for interactive web deployment**, delivering **1.84× faster inference latency (0.309s vs. 0.569s per article)**, half the training duration (11.5 min vs. 21.9 min), and stable fp16 execution on consumer/cloud GPUs without numerical instability.

The accompanying Gradio web application bridges rigorous academic benchmarking with an intuitive, interactive demonstration tool suitable for real-time presentation and user exploration.

---

## References

1. Lewis, M., et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.* ACL 2020.
2. Raffel, C., et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR 2020.
3. Hermann, K. M., et al. (2015). *Teaching Machines to Read and Comprehend.* NeurIPS 2015. (CNN/DailyMail Dataset).
4. Lin, C. Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL Workshop.
5. Wolf, T., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing.* EMNLP 2020.
