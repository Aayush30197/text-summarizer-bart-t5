# 10-Minute Plain-English Presentation Script

**Project:** Comparative Evaluation of BART and T5 for Abstractive Text Summarization  
**Course:** CSCI 390 (Senior Capstone / NLP), Fisk University  
**Instructor:** Dr. Qi Li  
**Speaker:** Aayush Khanal  
**Audience:** Classmates & Instructor (Explaining concepts from first principles with intuitive analogies)  
**Format:** 10-Minute Talk + Live Gradio Demonstration  

---

## 10-Minute Timing Breakdown

```
[0:00 - 1:30]  Slide 1: What is Abstractive Summarization? (Highlighter vs. Note-Taking)
[1:30 - 3:00]  Slide 2-3: Why is this Hard? & Meet the Contenders (BART vs. T5)
[3:00 - 4:30]  Slide 4-5: Training on 12k News Articles & What is ROUGE?
[4:30 - 6:00]  Slide 6: Quantitative Results (Who Won the Numbers Game?)
[6:00 - 8:30]  LIVE DEMO: Gradio App (Side-by-Side Comparison & Real AI Mistakes)
[8:30 - 9:30]  Slide 7: Why Numbers Lie (Double-Blind Human Error Analysis)
[9:30 - 10:00] Slide 9-10: Limitations, When to Choose Which Model & Q&A
```

---

## Word-for-Word Speaking Script

### Minute 0:00 – 1:30 | Slide 1: What is Abstractive Summarization?
* **On Screen:** Slide 1 with Extractive (Highlighter) vs. Abstractive (Note-Taking) comparison.
* **What to Say:**
  > *"Good morning Dr. Li and classmates! Today, I’m presenting my capstone project on a problem that impacts all of us: Can we teach computers to read long, complicated news stories and write accurate summaries in their own words?*  
  > *In Natural Language Processing, there are two distinct ways a computer can summarize:*  
  > *- The old way is called **Extractive Summarization**. Think of this like taking a yellow highlighter to a textbook: the computer can only copy and paste full sentences that were already written. It's safe, but it sounds choppy and robotic.*  
  > *- What we are building today is called **Abstractive Summarization**. Think of this like a student reading an entire chapter and writing their own study notes. The AI has to understand what happened, synthesize the core facts, and generate brand-new sentences from scratch.*  
  > *Our goal was to take 1,000-word breaking news articles from CNN and generate concise, accurate 3-sentence summaries in under a second using two leading language models: **BART** and **T5**."*

---

### Minute 1:30 – 3:00 | Slides 2 & 3: Why is this Hard? & The Two Models
* **On Screen:** Slide 2 (Hallucinations, Missing the Point, Loops) → Slide 3 (BART vs. T5).
* **What to Say:**
  > *"Why is abstractive summarization so challenging? Because modern language models are basically giant probability calculators—they predict words, but they don't have an inherent concept of objective truth. This creates three big pitfalls:*  
  > *1. **Hallucinations:** The AI can confidently invent facts out of thin air—like naming a stadium that was never in the story.*  
  > *2. **Missing the main point:** The AI might get distracted describing background paperwork and forget to mention whether the defendant was found guilty.*  
  > *3. **Repetition loops:** Without mathematical penalties, the model can get stuck repeating the same phrase over and over.*  
  > *To test how different AI designs handle these challenges, we selected two distinct transformer models:*  
  > *- **BART** (built by Meta): Think of BART as a 'Sentence Repairman'. It was pre-trained by taking clean text, erasing random words, scrambling sentences, and learning to piece the original article back together. This makes it naturally skilled at editing and condensing.*  
  > *- **T5** (built by Google): Think of T5 as a 'Universal Text Converter'. Google trained it to do every NLP task using plain English prompts. To make T5 summarize, you literally feed it the instruction: `summarize:` followed by the text."*

---

### Minute 3:00 – 4:30 | Slides 4 & 5: Training & What is ROUGE?
* **On Screen:** Slide 4 (12,000 Articles & 512 Cutoff) → Slide 5 (What is ROUGE?).
* **What to Say:**
  > *"To ensure a completely fair comparison, I trained both models under identical conditions:*  
  > *- We used the famous **CNN/DailyMail dataset**, consisting of real news articles paired with bullet-point summaries written by human journalists.*  
  > *- We sampled **12,000 training articles** and **1,000 test articles** using a locked random seed so there was zero cheating or data leakage.*  
  > *- We capped input articles at **512 words**. In journalism, there's a well-known concept called the 'inverted pyramid': reporters put the most critical who, what, when, and where right at the top. So reading the first 512 words captures over 90% of the core facts while saving massive computing time.*  
  > *Now, how do computer scientists grade an AI summary? We use a metric called **ROUGE**.*  
  > *Don't let the acronym intimidate you—ROUGE is simply a word-matching game against a human journalist:*  
  > *- **ROUGE-1** counts how many individual words match.*  
  > *- **ROUGE-2** counts how many 2-word phrases like 'White House' match.*  
  > *- **ROUGE-L** measures the longest sequence of words that appear in the exact same sentence order.*  
  > *In news summarization, a ROUGE score around 40% is considered a strong, human-grade result."*

---

### Minute 4:30 – 6:00 | Slide 6: Benchmark Results (Who Won?)
* **On Screen:** Slide 6 (Results Table: ROUGE scores vs. Speed).
* **What to Say:**
  > *"Here are the empirical results across 1,000 brand-new, unseen news articles:*  
  > *First, training was a massive success: both models leaped by **+12.4 percentage points** in ROUGE-1 after learning on our 12,000 articles.*  
  > *If you look strictly at the ROUGE leaderboard, **T5 won the numbers game**: it reached **42.88% ROUGE-1** compared to BART's **40.23%**.*  
  > *However, look at the right side of this table—computational efficiency tells a completely different story:*  
  > *- **BART generated summaries in 0.31 seconds per article—nearly twice as fast as T5 (0.57 seconds).**  
  > *- During training, BART finished in just **11.5 minutes**, whereas T5 took **22 minutes**.*  
  > *- BART produced punchy, concise summaries averaging 83 words, while T5 wrote wordier, compound sentences averaging 100 words.*  
  > *If you're building a live website or mobile app where a user is waiting for an instant summary, BART's speed advantage is huge."*

---

### Minute 6:00 – 8:30 | LIVE DEMONSTRATION (Gradio Web App)
* **Action:** Click **"🚀 Open Live Gradio Demo"** in the slide header to switch to `http://127.0.0.1:8080`.
* **What to Say & Do:**
  > *"Now, let's look at the models side by side in our interactive Gradio application running live on my laptop.*  
  > *(Select Spotlight 1: Cape Town ISIS Recruiter)*  
  > *In Spotlight 1, look at what BART wrote: It starts with 'Virginia teen accused of being ISIS recruiter'—but this article took place in South Africa! BART suffered from an extrinsic hallucination by retrieving a memory from a different story it saw during pre-training. Meanwhile, T5 stayed completely faithful to the Cape Town context.*  
  > *(Select Spotlight 2: Premier League Soccer)*  
  > *Now look at Spotlight 2, where the tables turn: Crystal Palace played Manchester City. T5 hallucinated that the match was played at 'the Emirates'—which is Arsenal's stadium! BART, on the other hand, got the venue 100% right and preserved the manager's exact quotes.*  
  > *(Select Spotlight 3: Police Shooting)*  
  > *In Spotlight 3, notice event coverage: BART captures the critical headline that the reserve deputy was charged with second-degree manslaughter. T5 got lost describing background training paperwork and missed the actual criminal charge!*  
  > *Finally, our live telemetry shows BART generating summaries in ~0.3 seconds versus ~0.57 seconds for T5."*

---

### Minute 8:30 – 9:30 | Slide 7: Why Numbers Lie (Double-Blind Human Study)
* **On Screen:** Slide 7 (1–5 Star Human Ratings & Error Taxonomy).
* **What to Say:**
  > *"Our demo proves a famous saying in NLP: **High ROUGE scores do not guarantee true facts.** An AI can copy long phrases to get high ROUGE points, but still get the facts completely backward.*  
  > *To test this scientifically, I conducted a **double-blind human evaluation** on 25 randomized articles. I anonymized the summaries into 'Summary A' and 'Summary B' with model names locked in a key file so I couldn't be biased.*  
  > *I graded each summary from 1 to 5 stars on four questions:*  
  > *1. Coverage: Did it capture the main news story?*  
  > *2. Factuality: Were the facts strictly true to the article?*  
  > *3. Fluency: Did it read naturally?*  
  > *4. Conciseness: Was it free of fluff?*  
  > *When we unlocked the key, the surprise was that **BART actually beat T5 in factual accuracy (4.88 vs. 4.68) and main story coverage (4.24 vs. 3.88)**.*  
  > *T5 had 3 factual hallucinations, while BART only had 1. The dominant error for both models was not hallucination, but **omission**—simply running out of space under the 128-word limit before finishing late details."*

---

### Minute 9:30 – 10:00 | Slides 9 & 10: Limitations & The Final Verdict
* **On Screen:** Slide 9 (Limitations) → Slide 10 (Decision Matrix).
* **What to Say:**
  > *"Every good research project has limitations: We trained for 1 epoch on 12,000 articles due to free cloud GPU quotas; long-form investigative pieces get truncated past 512 words; and our human study evaluated 25 articles with a single evaluator.*  
  > *So, which model should you choose if you are building an NLP product?*  
  > *- Pick **BART** if you care about speed, lower cloud bills, and direct factual accuracy. It runs twice as fast, cuts straight to the headline, and trains in half the time.*  
  > *- Pick **T5** if your priority is longer, grammatically complex prose or if you want a single model that can handle translation and question-answering alongside summarization.*  
  > *Thank you Dr. Li, thank you classmates, and I'd be happy to take any questions or run custom articles through our demo!"*

---

## 3 Quick Answers to Questions Your Classmates Might Ask

1. **"Can I paste any news article into your app right now?"**
   * *Answer:* "Yes! The Gradio app allows you to paste any English text into the source box, adjust the beam search sliders, and compare both models live."
2. **"Why did BART confuse Virginia and South Africa in that one article?"**
   * *Answer:* "BART was pre-trained on billions of words of web text. When it saw words like 'teen recruiter', its language model memory retrieved a high-profile Virginia news story and stitched it into the output. It's a classic example of why human oversight is still essential."
3. **"Why did T5 take twice as long to train and run?"**
   * *Answer:* "T5 has 220 million parameters compared to BART's 140 million (about 57% more math operations per token). Furthermore, T5 required full 32-bit floating point precision, whereas BART ran efficiently in 16-bit mixed precision."
