# 10-Minute Presentation Script & Slide Outline

**Project:** Comparative Evaluation of BART and T5 for Abstractive Text Summarization  
**Course:** CSCI 390 (Capstone / NLP), Fisk University  
**Instructor:** Dr. Qi Li  
**Speaker:** Aayush Khanal  
**Presentation Date:** November 9, 2026  
**Format:** 10-Minute Talk + Live Gradio Demonstration  

---

## Presentation Timing Breakdown (10 Minutes Total)

```
[0:00 - 1:30]  Slide 1-2: Problem Statement, Motivation & Research Questions
[1:30 - 3:00]  Slide 3-4: Dataset (CNN/DM), Truncation & Architecture Differences
[3:00 - 4:30]  Slide 5-6: Experimental Parity, Training Rigor & Baseline Asymmetry
[4:30 - 6:00]  Slide 7-8: Quantitative Results (ROUGE & Efficiency)
[6:00 - 8:30]  LIVE DEMO: Gradio App (Side-by-Side Comparison & Beam Exploration)
[8:30 - 9:30]  Slide 9:   Double-Blind Error Analysis & Failure Modes
[9:30 - 10:00] Slide 10:  Limitations, Key Takeaways & Q&A
```

---

## Detailed Minute-by-Minute Script

### Minute 0:00 – 1:30 | Slide 1–2: Introduction & Research Questions
* **Visual:** Title slide with project title, author, and research question matrix.
* **Spoken Script:**
  > *"Good morning Dr. Li and classmates. Today I am presenting a comparative evaluation of BART and T5 for abstractive text summarization.*  
  > *Abstractive summarization is one of the most demanding tasks in natural language processing because the model cannot simply extract sentences—it must synthesize facts, preserve critical entities, and paraphrase fluently without introducing hallucinations.*  
  > *Our research investigates four primary questions:*  
  > *1. When fine-tuned under strictly matched training conditions, which model produces higher lexical quality?*  
  > *2. How well do the models preserve core facts versus introducing hallucinations?*  
  > *3. How sensitive are summaries to decoding parameters like beam search and length penalties?*  
  > *4. What are the practical trade-offs in inference latency and memory for interactive deployment?"*

---

### Minute 1:30 – 3:00 | Slide 3–4: Dataset & Architectural Differences
* **Visual:** Architectural comparison diagram (BART Denoising Autoencoder vs. T5 Unified Text-to-Text).
* **Spoken Script:**
  > *"We evaluated both models on the standard CNN/DailyMail benchmark. Due to compute and timeline constraints, we locked a fixed random sample of 12,000 training, 1,000 validation, and 1,000 test articles using seed 42, ensuring zero train-test leakage.*  
  > *Both models were capped at 512 input tokens and 128 target tokens. In journalism, this aligns well with the 'inverted pyramid' structure where the lead paragraphs carry the most crucial details.*  
  > *Architecturally, the models represent two distinct paradigms:*  
  > *- **BART-base** (~140M parameters) uses a bidirectional encoder with a 6-layer autoregressive decoder pre-trained on corrupted text reconstruction.*  
  > *- **T5-base** (~220M parameters) uses a unified 12-layer text-to-text encoder-decoder that requires the explicit task prefix `summarize: `."*

---

### Minute 3:00 – 4:30 | Slide 5–6: Experimental Rigor & Pre-Trained Baselines
* **Visual:** Hyperparameter comparison table and Baseline ROUGE bar chart.
* **Spoken Script:**
  > *"To ensure a fair comparison, both models were fine-tuned using AdamW, an effective batch size of 32, linear warmup, and identical beam search settings (4 beams, length penalty of 2.0, and 3-gram blocking).*  
  > *A critical engineering challenge was precision: while BART trained in mixed precision fp16 on the T4 GPU, T5 required fp32 execution to avoid NaN gradient updates.*  
  > *Before fine-tuning, we measured zero-shot baselines on Apple Silicon MPS.*  
  > *Interestingly, zero-shot T5 scored 30.4% ROUGE-1 compared to BART's 27.9%. This reveals a known empirical nuance: T5's pre-training corpus included multi-task datasets like CNN/DailyMail, whereas BART-base had zero prior exposure to this specific task. The true comparison begins after both are fine-tuned under identical conditions."*

---

### Minute 4:30 – 6:00 | Slide 7–8: Empirical Results (ROUGE & Efficiency)
* **Visual:** Final ROUGE-1/2/L table, training loss curves, and latency per article comparison.
* **Spoken Script:**
  > *"Looking at our fine-tuned results on the 1,000-article test set:*  
  > *[Present ROUGE numbers and loss convergence trends].*  
  > *In terms of computational efficiency:*  
  > *- BART-base achieved lower per-article generation latency, primarily due to its shallower 6-layer decoder and BPE tokenization efficiency.*  
  > *- T5-base had higher memory usage and inference latency due to its 12-layer decoder depth, representing a trade-off between model capacity and deployment efficiency."*

---

### Minute 6:00 – 8:30 | LIVE DEMONSTRATION: Interactive Gradio App
* **Visual:** Switch screen to the running Gradio Web App (`python app/app.py`).
* **Demonstration Steps:**
  1. **Example 1 (Balanced Performance):** Select pre-loaded Article 1 from the dropdown. Click **Summarize**. Point out the side-by-side comparison, ground truth reference, and generated token count.
  2. **Parameter Adjustment (Beam Search Exploration):** Change `num_beams` from 4 to 1 (greedy). Click **Summarize**. Point out how latency drops by ~60%, but the summary exhibits slight phrasing repetition.
  3. **Error Analysis Spotlight:** Select pre-loaded Article 2 (or a custom example). Point out a specific difference in how an entity or numerical figure was handled (e.g. date attribution or fact omission).
* **Spoken Script:**
  > *"Now I'd like to demonstrate the interactive Gradio app we built. Both models are served side by side. Here with Article 1, notice how both models capture the core event, but BART generates more concise sentences while T5 tends toward fuller clauses. When we adjust the beam slider from 4 down to 1, you can see the real-time latency drop on the telemetry counter, demonstrating the inference cost of beam exploration."*

---

### Minute 8:30 – 9:30 | Slide 9: Double-Blind Error Analysis
* **Visual:** Likert score radar chart and taxonomy error breakdown (Wrong Entity, Date Distortion, Hallucination, Omission).
* **Spoken Script:**
  > *"Because ROUGE only measures lexical n-gram overlap, we conducted a double-blind human error analysis on 25 randomized test articles.*  
  > *Summaries were shuffled into 'Summary A' and 'Summary B' with hidden keys.*  
  > *The most frequent failure mode across both models was not outright hallucination, but **critical fact omission**—leaving out the final resolution of a news story—as well as occasional **numeric/date confusion**. T5 demonstrated slightly stronger factual consistency, while BART produced summaries with fewer redundant clauses."*

---

### Minute 9:30 – 10:00 | Slide 10: Limitations & Conclusion
* **Visual:** Summary bullet points, course takeaways, and GitHub repository QR code.
* **Spoken Script:**
  > *"In conclusion, both BART and T5 are powerful architectures with distinct operational profiles. BART offers faster inference and lighter memory footprints ideal for edge applications, while T5 offers strong structural cohesion at the cost of higher latency.*  
  > *Key limitations of our study include 512-token input truncation and single-seed training on a 12,000-article subset.*  
  > *Our complete code, dataset indices, configs, and interactive app are open-source on GitHub.*  
  > *Thank you, and I welcome any questions."*
