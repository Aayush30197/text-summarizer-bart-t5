# Instructions for Instructor & Evaluator (Dr. Qi Li)

**Course:** CSCI 390: Natural Language Processing and Applications — Fall 2026  
**Institution:** Fisk University  
**Project:** Comparative Evaluation of BART and T5 for Abstractive Text Summarization  
**Author:** Aayush Khanal  
**GitHub Repository:** [https://github.com/Aayush30197/text-summarizer-bart-t5](https://github.com/Aayush30197/text-summarizer-bart-t5)  

---

## ⚡ 1-Minute Quickstart (Fast Evaluation)

If you simply want to test the interactive web demo, slides, and technical report immediately, run:

```bash
# 1. Clone repository
git clone https://github.com/Aayush30197/text-summarizer-bart-t5.git
cd text-summarizer-bart-t5

# 2. Setup Python environment (Python 3.9+)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Launch the Interactive Gradio Demo (Deliverable 3.4)
python app/app.py
```
👉 Open your browser to **`http://127.0.0.1:7860`** (or `http://127.0.0.1:8080`).

---

## 📋 Course Deliverables & Grading Criteria Mapping

| Deliverable | Syllabus Requirement | Location in Repository | Points |
|---|---|---|:---:|
| **1. Code Repository** | Reproducible code, clean structure, dependencies, and instructions. | Modular pipeline in `src/`, declarative configs in `configs/`, index files in `data/`, and [`README.md`](README.md). | **30%** |
| **2. Technical Report** | 2–4 page formal report with problem formulation, architecture, ROUGE evaluation, and reflections on errors. | Markdown: [`report/report.md`](report/report.md)<br>Printable HTML: [`report/report.html`](report/report.html) (print to PDF via <kbd>Cmd</kbd>+<kbd>P</kbd>). | **25%** |
| **3. Interactive Presentation** | 30-min talk (5 min intro, 5 min implementation, 5 min errors, 10 min demo, 5 min Q&A). | Interactive slide deck: [`report/slides.html`](report/slides.html)<br>Speaker notes & script: [`report/PRESENTATION_SCRIPT.md`](report/PRESENTATION_SCRIPT.md). | **25%** |
| **4. Interactive Demo** | Web-based system allowing students to input articles and generate summaries live. | Gradio web application in [`app/app.py`](app/app.py) with 4 spotlight error cases and live input capabilities. | **15%** |
| **5. Innovation & Extension** | Going beyond basic baseline replication. | • Direct parity comparison of **TWO architectures** (BART vs. T5)<br>• **Double-blind human evaluation** with 5-class error taxonomy (`src/error_analysis.py`)<br>• Discovery of T5 fp16 NaN loss instability on T4 GPUs and mitigation in fp32. | **5%** |

---

## 🖥️ How to Run & Inspect Each Component

### A. Run the Interactive Gradio Demo (`app/app.py`)
```bash
python app/app.py
```
* **Features to Test:**
  1. **Spotlight 1 (Cape Town Airport):** Select from the dropdown to observe BART's extrinsic hallucination (*"Virginia teen accused of being recruiter"*) contrasted with T5's faithful summary.
  2. **Spotlight 2 (Premier League Soccer):** Observe T5 hallucinating *"the Emirates stadium"* for a Crystal Palace fixture while BART maintains 100% stadium accuracy.
  3. **Spotlight 3 (Police Shooting):** Observe event coverage differences (manslaughter charge vs. training paperwork).
  4. **Custom Text Input:** Paste any English news article into the source text box and click **Generate / Refresh Summaries**.
  5. **Decoding Sliders:** Test adjusting `num_beams` (1 to 8), `min_length`, and `max_length`.

---

### B. View the Presentation Slide Deck (`report/slides.html`)
```bash
# On macOS:
open report/slides.html

# Or open report/slides.html in Chrome, Safari, or Firefox
```
* **Slide Controls:**
  * <kbd>→</kbd> / <kbd>Space</kbd>: Next slide
  * <kbd>←</kbd>: Previous slide
  * <kbd>N</kbd>: Toggle the **Speaker Notes Drawer** to view the verbatim speaking script.
  * **Built-in Presentation Stopwatch:** Click **Start** in the header to time the 10/30 minute talk.

---

### C. View the Technical Report (`report/report.html`)
```bash
# On macOS:
open report/report.html
```
* Built with an academic serif layout, formatted ROUGE tables, error distribution statistics, case studies, and answers to RQ1–RQ4.
* Click the blue **"📄 Save as Canvas PDF (Print)"** button at the top to print/save a publication-ready PDF.

---

### D. Reproduce the Evaluation & Error Analysis Pipelines

#### 1. Zero-Shot Baselines (Smoke Test)
```bash
python src/baseline.py --model bart --smoke
python src/baseline.py --model t5 --smoke
```

#### 2. Fine-Tuning Pipeline
```bash
# Local smoke test (5 samples):
python src/train.py --model bart --smoke
python src/train.py --model t5 --smoke

# Full training (12,000 articles, 1 epoch):
python src/train.py --model bart
python src/train.py --model t5
```

#### 3. Full 1,000-Article Evaluation
```bash
python src/evaluate.py --model bart --checkpoint checkpoints/bart-base
python src/evaluate.py --model t5 --checkpoint checkpoints/t5-base
```

#### 4. Double-Blind Human Error Analysis Tool
```bash
# Generate randomized, blinded evaluation sheets:
python src/error_analysis.py --generate

# De-blind and compute Likert ratings and taxonomy stats:
python src/error_analysis.py --deblind
```

---

## 📊 Summary of Benchmark Findings

| Model | Parameters | Training Time | Test Latency | ROUGE-1 | ROUGE-2 | ROUGE-L | Human Coverage | Human Factuality |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BART-base** | ~140M | **11.5 min** (fp16) | **0.309s / art** | 40.23% | 17.71% | 27.06% | **4.24 / 5.0** | **4.88 / 5.0** |
| **T5-base** | ~220M | 21.9 min (fp32) | 0.569s / art | **42.88%** | **20.24%** | **30.12%** | 3.88 / 5.0 | 4.68 / 5.0 |

* **Speed Winner:** BART is **1.84× faster** during beam search inference.
* **Recall Winner:** T5 achieves **+2.65% higher ROUGE-1** due to multi-task pre-training and longer sentences.
* **Factuality Winner:** Human study revealed BART retains lead journalistic facts more reliably (4.24 vs 3.88 coverage) with fewer hallucinations (1 vs. 3).

---
*For questions or grading inquiries, please contact Aayush Khanal at Fisk University.*
