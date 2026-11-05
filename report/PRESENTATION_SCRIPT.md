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
  > *- Both models saw an impressive +12.4% leap in ROUGE-1 over their baselines.*  
  > *- T5-base achieved 42.88% ROUGE-1, 20.24% ROUGE-2, and 30.12% ROUGE-L.*  
  > *- BART-base achieved 40.23% ROUGE-1, 17.71% ROUGE-2, and 27.06% ROUGE-L, while achieving a higher ROUGE-Lsum of 37.01%.*  
  > *However, when we look at computational efficiency, BART-base was the clear winner:*  
  > *- BART generated summaries in just 0.309 seconds per article—nearly twice as fast as T5 at 0.569 seconds.*  
  > *- BART completed full training in just 11.5 minutes compared to 22 minutes for T5.*  
  > *- BART produced concise 83-token summaries, whereas T5 produced longer 100-token summaries."*

---

### Minute 6:00 – 8:30 | LIVE DEMONSTRATION: Interactive Gradio App
* **Visual:** Switch screen to the running Gradio Web App (`python app/app.py`).
* **Demonstration Steps:**
  1. **Spotlight 1 (T5 Outperforms BART):** Select *"Spotlight 1: Cape Town ISIS Recruiter"* from the dropdown. Highlight the analytical callout: BART hallucinates a spurious headline (*"Virginia teen accused of being ISIS recruiter"*), while T5 remains completely grounded and faithful.
  2. **Spotlight 2 (BART Outperforms T5):** Select *"Spotlight 2: Premier League Title Race"*. Point out that T5 hallucinates the stadium as *"the Emirates"* (Arsenal's ground) for Crystal Palace, whereas BART preserves accurate quotes and venue fidelity.
  3. **Spotlight 3 (Coverage Contrast):** Select *"Spotlight 3: Reserve Deputy Shooting"*. Show how BART captures the critical event resolution (second-degree manslaughter charge in the death of Eric Harris), while T5 only reports training paperwork.
* **Spoken Script:**
  > *"Now I'd like to demonstrate our interactive Gradio application. In this interface, both architectures are presented side by side alongside the human ground-truth reference summary and real-time telemetry.*  
  > *First, let's examine Spotlight 1. Notice how BART hallucinates an external headline about a 'Virginia teen recruiter', whereas T5 remains strictly grounded in the Cape Town context.*  
  > *Next, in Spotlight 2, we see the inverse: T5 hallucinates 'the Emirates stadium' for Crystal Palace, while BART maintains perfect venue fidelity.*  
  > *Finally, our telemetry counters show BART running at 0.31s per article versus 0.57s for T5—demonstrating BART's 1.84x latency advantage for real-time applications."*

---

### Minute 8:30 – 9:30 | Slide 9: Double-Blind Error Analysis
* **Visual:** Likert score breakdown (Coverage: BART 4.24 vs T5 3.88; Factual Consistency: BART 4.88 vs T5 4.68; Fluency: T5 4.40 vs BART 4.32) and taxonomy error counts (OM: 5 vs 8, EH: 1 vs 3, WE: 1 vs 1).
* **Spoken Script:**
  > *"Because ROUGE only measures lexical n-gram overlap, we conducted a double-blind human error analysis on 25 randomized test articles using a 1-to-5 Likert rubric and a 5-class error taxonomy.*  
  > *Summaries were shuffled into anonymized 'Summary A' and 'Summary B' columns with model identities hidden in a key file.*  
  > *The empirical results were revealing: BART scored higher in coverage—4.24 versus 3.88—and factual consistency—4.88 versus 4.68. The dominant error mode across both models was not hallucination, but **critical fact omission** due to the 128-token generation limit. However, we did discover rare, fascinating failure modes: for example, T5 hallucinated 'the Emirates stadium' instead of Selhurst Park for a Crystal Palace match, while BART once prepended a headline from a Virginia news story onto a South African article.*  
  > *T5 achieved slightly higher readability (4.40 vs 4.32), primarily because BART occasionally truncated mid-sentence when hitting token limits."*

---

### Minute 9:30 – 10:00 | Slide 10: Limitations & Conclusion
* **Visual:** Summary bullet points, course takeaways, and GitHub repository QR code.
* **Spoken Script:**
  > *"In conclusion, both BART and T5 are powerful architectures with distinct operational profiles. BART offers faster inference and lighter memory footprints ideal for edge applications, while T5 offers strong structural cohesion at the cost of higher latency.*  
  > *Key limitations of our study include 512-token input truncation and single-seed training on a 12,000-article subset.*  
  > *Our complete code, dataset indices, configs, and interactive app are open-source on GitHub.*  
  > *Thank you, and I welcome any questions."*
