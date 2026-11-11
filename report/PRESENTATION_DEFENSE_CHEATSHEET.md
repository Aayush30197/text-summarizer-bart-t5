# CSCI 390 Capstone Defense Cheatsheet & Study Guide

**Author:** Aayush Khanal  
**Project:** Comparative Evaluation of BART and T5 for Abstractive Summarization  
**Instructor:** Dr. Qi Li • Fisk University  
**Date of Defense:** Fall 2026  

---

## 1. The Numbers to Have on the Tip of Your Tongue

If Dr. Li or a classmate asks you for a specific number during the presentation, here is the exact cheat sheet:

| Metric / Dimension | BART-base (~140M) | T5-base (~220M) | Key Relationship |
|---|:---:|:---:|---|
| **Zero-Shot Baseline (Untrained)** | 27.87% ROUGE-1 | 30.43% ROUGE-1 | T5 started **+2.56% higher** due to C4 pre-training |
| **Fine-Tuned Score (1,000 Test Articles)** | **40.23%** ROUGE-1 | **42.88%** ROUGE-1 | T5 leads by **+2.65%** in lexical word recall |
| **ROUGE-2 (Two-word phrases)** | 17.71% | 20.24% | T5 leads by +2.53% |
| **ROUGE-L (Sentence flow)** | 27.06% | 30.12% | T5 leads by +3.06% |
| **ROUGE-Lsum (Sentence-level LCS)** | **37.01%** | 36.67% | **BART leads by +0.34%** |
| **Net Fine-Tuning Gain ($\Delta$)** | **+12.36%** | **+12.45%** | **Virtually identical learning rate (+12.4%)** |
| **Inference Latency** | **0.309s / article** | 0.569s / article | **BART is 1.84× faster (near instant)** |
| **Training Duration (1.0 Epoch)** | **11.5 minutes** (fp16) | 21.9 minutes (fp32) | BART trained in **half the time** |
| **Average Summary Length** | 83.1 tokens | 100.3 tokens | BART is punchy; T5 writes compound sentences |
| **Human Event Coverage (1–5)** | **4.24 / 5.0** | 3.88 / 5.0 | **BART wins (+0.36)**: better lead news retention |
| **Human Factuality / Truth (1–5)** | **4.88 / 5.0** | 4.68 / 5.0 | **BART wins (+0.20)**: fewer hallucinations |
| **Human Fluency & Grammar (1–5)** | 4.32 / 5.0 | **4.40 / 5.0** | T5 wins (+0.08): finishes clauses before length caps |
| **Hallucination Count (N=25)** | **1 occurrence** | 3 occurrences | T5 hallucinated 3× more often |
| **Dataset Splits** | 12,000 train | 1,000 val / 1,000 test | Sampled with locked seed 42 (zero leakage) |
| **Input / Output Token Limits** | 512 input tokens | 128 output tokens | Leverages news "Inverted Pyramid" structure |
| **Decoding Configuration** | `num_beams=4` | `min_length=30`, `max_length=128` | `no_repeat_ngram_size=3`, `length_penalty=2.0` |

---

## 2. Technical Concepts Explained in Simple Terms

### A. What is Seq2Seq (Encoder-Decoder)?
* **Encoder:** Reads the source article and compresses its meaning into high-dimensional vectors (dense numerical representations).
* **Decoder:** Unpacks those vectors token-by-token into a new English summary, using cross-attention to look back at the original article at each step.

### B. BART vs. T5 Architecture
* **BART (Bidirectional and Auto-Regressive Transformer):**
  * *Encoder:* Bidirectional (looks left and right simultaneously, like BERT).
  * *Decoder:* Autoregressive (generates words left-to-right, like GPT).
  * *Pre-training:* **Denoising Autoencoder.** Meta corrupted text (deleted words, masked tokens, scrambled sentences) and trained BART to reconstruct the original clean text.
* **T5 (Text-to-Text Transfer Transformer):**
  * Built by Google. Formulates every NLP problem as text-in, text-out.
  * Conditions tasks using explicit natural language prefixes (e.g., `summarize: `).
  * Pre-trained on C4 (Colossal Clean Crawled Corpus) with masked span corruption.

### C. What is Beam Search?
* **Greedy Search (`num_beams=1`):** Picks the single most probable word at every step. Fast, but gets stuck in repetitive or dead-end sentences.
* **Beam Search (`num_beams=4`):** Maintains the top 4 most probable partial sentences simultaneously at each token step, pruning low-scoring paths. Produces much higher quality, fluent sentences.
* **3-Gram Blocking (`no_repeat_ngram_size=3`):** Forces the probability of any 3-word phrase to zero if it was already generated earlier in the summary. This completely eliminates repetitive stuttering loops.

### D. What is ROUGE?
* **ROUGE-1:** Overlap of single words (unigrams) between AI and human.
* **ROUGE-2:** Overlap of 2-word consecutive phrases (bigrams).
* **ROUGE-L:** Longest Common Subsequence of words appearing in sentence order.
* **ROUGE-Lsum:** Calculates ROUGE-L on split sentences, standard for multi-sentence news summaries.
* **The Blindspot:** ROUGE only counts word overlap. It cannot detect if an AI says "guilty" instead of "not guilty"!

---

## 3. Top 15 Toughest Presentation Questions & Word-for-Word Answers

### Q1: "Why did T5 score higher on ROUGE, but lower on human factuality and coverage?"
> **Your Answer:**  
> *"That was the most fascinating empirical finding of our study! ROUGE measures lexical n-gram recall against a single human reference. Because T5 generates significantly longer summaries (100.3 tokens average vs. 83.1 for BART), it naturally captures more word overlap by virtue of length.*  
> *However, our double-blind human evaluation on 25 randomized articles showed that BART focuses more strictly on the defining lead event facts (scoring 4.24 vs. 3.88 in coverage) and exhibited fewer factual hallucinations (1 for BART vs. 3 for T5). This demonstrates why relying solely on automated metrics like ROUGE in NLP can be misleading."*

---

### Q2: "What was the technical cause of the T5 fp16 NaN loss bug, and how did you fix it?"
> **Your Answer:**  
> *"When fine-tuning T5 on NVIDIA Tesla T4 GPUs in 16-bit mixed precision (fp16), the loss crashed to NaN around step 42.*  
> *The root cause is structural: unlike BART, T5 uses Root Mean Square Layer Normalization (RMSNorm) and no additive biases. On T4 architectures, accumulating layer activations causes gradient values to exceed the maximum range of IEEE 754 half-precision (65,504), resulting in numerical overflow and NaN weights.*  
> *We resolved this by enforcing full 32-bit floating point precision (fp32) for T5. While fp32 doubled T5's training time to 21.9 minutes, it guaranteed mathematical convergence. Meanwhile, BART's post-layer norm design remained 100% stable in fp16, training in just 11.5 minutes."*

---

### Q3: "Why did you truncate articles to 512 tokens instead of feeding the full text?"
> **Your Answer:**  
> *"We made this decision based on computational constraints and domain theory:*  
> *First, standard self-attention has quadratic memory complexity $O(N^2)$. On free cloud GPU quotas, processing 1,500+ tokens causes Out-Of-Memory errors during backpropagation.*  
> *Second, in journalism, articles strictly follow the 'inverted pyramid' principle: reporters place the critical 5 W's—who, what, where, when, why—in the lead three paragraphs. Academic literature on CNN/DailyMail confirms that the first 512 tokens contain over 90% of the information needed to generate the summary highlights."*

---

### Q4: "Why did you train for only 1 epoch on 12,000 articles?"
> **Your Answer:**  
> *"Our project operated under fixed compute constraints (Kaggle's 30 GPU hour/week quota on a single GPU). Training the full 287,000 CNN/DailyMail dataset for multiple epochs would require several days of compute.*  
> *By locking a representative random sample of 12,000 articles (seed 42), both models received sufficient gradient signal to achieve massive +12.4% leaps in ROUGE-1, reaching competitive benchmark performance (>40% ROUGE-1) while finishing training in under 25 minutes."*

---

### Q5: "How did you guarantee there was zero data leakage between train and test sets?"
> **Your Answer:**  
> *"We sampled the dataset once using a fixed random seed (42) and serialized the exact sample indices to immutable JSON files: `indices_train.json` ($N=12,000$), `indices_val.json` ($N=1,000$), and `indices_test.json` ($N=1,000$).*  
> *Our dataset loader in `src/data.py` asserts that the set intersection between train, validation, and test indices is strictly empty ($\emptyset$). Furthermore, all baseline, evaluation, and human error scripts load from these exact frozen index files."*

---

### Q6: "Why did you pick BART-base (~140M) and T5-base (~220M) instead of their Large counterparts?"
> **Your Answer:**  
> *"We selected base architectures for three scientific reasons:*  
> *1. **Fair Size Parity:** BART-base (140M) and T5-base (220M) are in the same parameter order of magnitude. In contrast, T5-small is only 60M parameters, which would have put T5 at an unfair architectural disadvantage.*  
> *2. **Compute Feasibility:** Large variants (BART-large 406M, T5-large 770M) require multi-GPU distributed clusters or 40GB+ A100 VRAM for fine-tuning without extreme batch size degradation.*  
> *3. **Interactive Latency:** Base models allow our local Gradio application to run inference in 0.3 to 0.5 seconds on a laptop without cloud latency."*

---

### Q7: "What hyperparameters did you lock for decoding, and why?"
> **Your Answer:**  
> *"In `configs/decoding.yaml`, we locked:*  
> *- `num_beams = 4`: Balances diverse hypothesis exploration with latency.*  
> *- `min_length = 30` & `max_length = 128`: Ensures summaries are neither empty snippets nor run-on essays.*  
> *- `length_penalty = 2.0`: Encourages models to produce complete, informative sentences.*  
> *- `no_repeat_ngram_size = 3`: Blocks any 3-gram from repeating. This completely eliminated degenerate repetition loops (`REP = 0`) across all 1,000 test articles."*

---

### Q8: "In Spotlight 1, why did BART hallucinate 'Virginia teen accused of being recruiter'?"
> **Your Answer:**  
> *"This is a textbook example of extrinsic hallucination caused by pre-training memory retrieval!*  
> *BART was pre-trained on billions of words of web text, including high-profile news stories about an American teenager in Virginia who was recruited by ISIS. When the fine-tuned BART saw prompt tokens like 'teen', 'detained', and 'ISIS', its language modeling weights activated high-frequency associations from that famous Virginia case, prepending that headline before returning to the factually accurate Cape Town details.*  
> *T5 avoided this error because its unified span-corruption objective and strict task prefix (`summarize: `) kept it tightly bound to the input context."*

---

### Q9: "In Spotlight 2, why did T5 hallucinate 'the Emirates stadium' instead of Selhurst Park?"
> **Your Answer:**  
> *"This is an entity-collocate hallucination. The article discussed Crystal Palace hosting Manchester City. Crystal Palace's home ground is Selhurst Park, which appeared only once late in the article.*  
> *Because Premier League soccer coverage on the web heavily features Arsenal's famous stadium 'the Emirates', T5's language model decoder relied on semantic collocation probabilities rather than strict source grounding. BART, by contrast, preserved the manager's exact quotes and stadium fidelity."*

---

### Q10: "What is the difference between ROUGE-L and ROUGE-Lsum?"
> **Your Answer:**  
> *"ROUGE-L computes the Longest Common Subsequence treating the entire summary as a single unbroken string of text.*  
> *ROUGE-Lsum, introduced specifically for multi-sentence summarization, splits the summary into sentences using newline breaks and calculates the union LCS score across all sentence pairs. ROUGE-Lsum is widely preferred for news benchmarks because news summaries consist of 3 to 4 distinct bullet points. On ROUGE-Lsum, BART achieved 37.01% versus 36.67% for T5."*

---

### Q11: "How was your human error analysis double-blinded?"
> **Your Answer:**  
> *"To eliminate cognitive bias, `src/error_analysis.py` sampled 25 randomized test articles and shuffled the fine-tuned BART and T5 summaries into anonymized columns labeled 'Summary A' and 'Summary B'.*  
> *The true mapping—which model produced Summary A or B for each sample—was saved to a separate, encrypted key file that was not opened until after all 25 articles were scored on a 1-to-5 Likert scale across Coverage, Factuality, Fluency, and Conciseness. Only then did we run `python src/error_analysis.py --deblind` to calculate the final aggregate ratings."*

---

### Q12: "Why didn't you use BERTScore or GPT-4 evaluation?"
> **Your Answer:**  
> *"In our initial project specification, we scoped our resources to adhere to course standards and avoid paid commercial APIs:*  
> *1. Standard ROUGE metrics are the canonical benchmark required by the literature and our course rubric.*  
> *2. While BERTScore measures semantic embedding similarity, it still does not assess factual grounding or catch entity swaps.*  
> *3. Rather than relying on an unverified automated LLM judge, we conducted an empirical, double-blind human rating study with a formal 5-class error taxonomy, which provides far more reliable qualitative insights."*

---

### Q13: "If you were the Lead ML Engineer at a news startup, which model would you deploy today?"
> **Your Answer:**  
> *"I would deploy **BART-base** without hesitation!*  
> *Here is the business case:*  
> *- **User Experience:** BART generates summaries in 0.31 seconds (1.84× faster than T5). In a consumer application, sub-second latency is the difference between an app feeling instant versus lagging.*  
> *- **Cloud Infrastructure Costs:** BART completed training in 11.5 minutes in fp16, cutting GPU compute bills in half compared to T5's 22-minute fp32 requirement.*  
> *- **Factual Reliability:** In our blinded human study, BART achieved higher factual consistency (4.88 vs 4.68) and better lead news coverage (4.24 vs 3.88). For news readers, truthful facts matter far more than a 2% difference in word overlap."*

---

### Q14: "How do the tokenizers of BART and T5 differ, and why does that matter?"
> **Your Answer:**  
> *"BART uses Byte-Pair Encoding (BPE) with a vocabulary of 50,265 tokens, which treats text at the byte level and avoids out-of-vocabulary tokens.*  
> *T5 uses SentencePiece unigram tokenization with a smaller vocabulary of 32,128 tokens, treating whitespace as a special character (`_`).*  
> *Because BART's vocabulary is larger, it often encodes common English words into fewer total tokens, contributing to its faster decoder generation speed."*

---

### Q15: "What would you do next if you had more time and compute?"
> **Your Answer:**  
> *"Three logical next steps:*  
> *1. **Multi-epoch Training:** Fine-tune for 3 to 5 epochs with early stopping to explore asymptotic convergence.*  
> *2. **Long-Context Transformers:** Evaluate architectures like Longformer or LED (Longformer-Encoder-Decoder) that can ingest 4,096+ tokens without 512-token truncation.*  
> *3. **RLHF / Factuality-Aware Loss:** Incorporate factual consistency penalties (like FactCC or entity-preservation loss) directly into the fine-tuning loss function to penalize hallucinations during training."*

---

## 4. How to Handle Any Question You Don't Know the Answer To

If Dr. Li asks an obscure question or something outside your project scope, **never make up an answer**. Use the classic academic 3-step pivot:

1. **Acknowledge the premise:** *"That is a great question, Dr. Li."*
2. **State what your data showed:** *"In our specific experiments on the 1,000-article CNN/DailyMail test split, we focused primarily on measuring lexical ROUGE overlap and human factuality under locked beam decoding..."*
3. **Hypothesize and offer next steps:** *"...while we didn't test that specific condition in our 1-epoch runs, based on the transformer literature, I would hypothesize that [give your intuition], and that would be a fantastic ablation study to test in future work."*
