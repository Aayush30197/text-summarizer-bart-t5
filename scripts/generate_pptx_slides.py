#!/usr/bin/env python3
"""
Generate a modern, high-end Canva/PowerPoint (16:9) presentation deck for
Aayush Khanal's CSCI 390 Senior Capstone Defense:
"Comparative Evaluation of BART and T5 for Abstractive Text Summarization"

All content is strictly framed in first-person singular ("I", "my", "me")
and every slide includes complete speaker notes for Dr. Qi Li and presentation submission.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# --- Color Palette (Canva Dark Minimalist / Indie Tech) ---
BG_DARK = RGBColor(12, 16, 23)        # #0c1017 deep film slate
CARD_BG = RGBColor(22, 30, 44)        # #161e2c sleek dark card
CARD_BORDER = RGBColor(45, 59, 82)    # #2d3b52 subtle border
GOLD = RGBColor(245, 158, 11)         # #f59e0b indie amber
CYAN = RGBColor(56, 189, 248)         # #38bdf8 glacier cyan (BART)
EMERALD = RGBColor(52, 211, 153)      # #34d399 emerald (T5 / Success)
CRIMSON = RGBColor(248, 113, 113)     # #f87171 warning / error
PURPLE = RGBColor(168, 85, 247)       # #a855f7 deep insight
TEXT_WHITE = RGBColor(248, 250, 252)  # #f8fafc heading white
TEXT_MUTED = RGBColor(148, 163, 184)  # #94a3b8 soft slate
TEXT_BODY = RGBColor(226, 232, 240)   # #e2e8f0 readable body

FONT_TITLE = "Helvetica"
FONT_BODY = "Arial"

def create_deck(output_paths):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_DARK
        bg.line.fill.background()
        return bg

    def add_header(slide, act_tag, title_text, indicator_text=""):
        # Act / Category Tag
        tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(8), Inches(0.35))
        tf = tag_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = act_tag.upper()
        p.font.name = FONT_BODY
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = GOLD

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(9.5), Inches(0.65))
        tf_t = title_box.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.name = FONT_TITLE
        p_t.font.size = Pt(22)
        p_t.font.bold = True
        p_t.font.color.rgb = TEXT_WHITE

        # Top Right Indicator / Badge
        if indicator_text:
            ind_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(9.8), Inches(0.55), Inches(2.7), Inches(0.45))
            ind_box.fill.solid()
            ind_box.fill.fore_color.rgb = RGBColor(30, 41, 59)
            ind_box.line.color.rgb = CARD_BORDER
            tf_i = ind_box.text_frame
            tf_i.margin_left = tf_i.margin_right = tf_i.margin_top = tf_i.margin_bottom = 0
            p_i = tf_i.paragraphs[0]
            p_i.text = indicator_text
            p_i.alignment = PP_ALIGN.CENTER
            p_i.font.name = FONT_BODY
            p_i.font.size = Pt(10)
            p_i.font.bold = True
            p_i.font.color.rgb = CYAN

    def add_card(slide, left, top, width, height, border_accent=None):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = border_accent if border_accent else CARD_BORDER
        card.line.width = Pt(1.5 if border_accent else 1.0)
        return card

    def add_banner(slide, text, icon="🎯", top=6.25):
        banner = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(top), Inches(11.733), Inches(0.65))
        banner.fill.solid()
        banner.fill.fore_color.rgb = RGBColor(18, 25, 38)
        banner.line.color.rgb = GOLD
        banner.line.width = Pt(1.0)
        tf = banner.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.text = f"{icon}  {text}"
        p.font.name = FONT_BODY
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = TEXT_BODY
        return banner

    def set_notes(slide, notes_text):
        notes_slide = slide.notes_slide
        tf = notes_slide.notes_text_frame
        tf.text = notes_text

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1)

    # Accent decorative bar
    bar = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(0.12), Inches(4.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = GOLD
    bar.line.fill.background()

    # Meta Tags
    meta_box = slide1.shapes.add_textbox(Inches(1.2), Inches(1.2), Inches(10), Inches(0.4))
    tf1 = meta_box.text_frame
    p = tf1.paragraphs[0]
    p.text = "FISK UNIVERSITY  //  DEPARTMENT OF COMPUTER SCIENCE  //  CSCI 390 CAPSTONE"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = GOLD

    # Main Title
    title_box = slide1.shapes.add_textbox(Inches(1.2), Inches(1.7), Inches(11), Inches(1.8))
    tf1_t = title_box.text_frame
    tf1_t.word_wrap = True
    p = tf1_t.paragraphs[0]
    p.text = "Comparative Evaluation of BART and T5 for Abstractive Text Summarization"
    p.font.name = FONT_TITLE
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = TEXT_WHITE

    # Subtitle
    sub_box = slide1.shapes.add_textbox(Inches(1.2), Inches(3.6), Inches(11), Inches(0.7))
    tf1_s = sub_box.text_frame
    tf1_s.word_wrap = True
    p = tf1_s.paragraphs[0]
    p.text = "A Rigorous Empirical Benchmark of Architectural Efficiency, Hallucination Dynamics, and Multi-Epoch Cloud Scaling on CNN/DailyMail"
    p.font.name = FONT_BODY
    p.font.size = Pt(14)
    p.font.color.rgb = TEXT_MUTED

    # Author Card
    auth_card = add_card(slide1, 1.2, 4.6, 5.5, 1.4, border_accent=CYAN)
    tf_a = auth_card.text_frame
    tf_a.margin_left = Inches(0.3)
    tf_a.margin_top = Inches(0.2)
    p = tf_a.paragraphs[0]
    p.text = "AUTHOR & PRESENTER"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = CYAN
    p2 = tf_a.add_paragraph()
    p2.text = "Aayush Khanal"
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_WHITE
    p3 = tf_a.add_paragraph()
    p3.text = "Senior Computer Science Major • Fisk University"
    p3.font.size = Pt(11)
    p3.font.color.rgb = TEXT_MUTED

    # Advisor Card
    adv_card = add_card(slide1, 7.0, 4.6, 5.5, 1.4, border_accent=GOLD)
    tf_adv = adv_card.text_frame
    tf_adv.margin_left = Inches(0.3)
    tf_adv.margin_top = Inches(0.2)
    p = tf_adv.paragraphs[0]
    p.text = "FACULTY ADVISOR & INSTRUCTOR"
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = GOLD
    p2 = tf_adv.add_paragraph()
    p2.text = "Dr. Qi Li"
    p2.font.size = Pt(18)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_WHITE
    p3 = tf_adv.add_paragraph()
    p3.text = "Assistant Professor of Computer Science • Fisk University"
    p3.font.size = Pt(11)
    p3.font.color.rgb = TEXT_MUTED

    add_banner(slide1, "Live Interactive Demonstration: Full Gradio Web Application Running on Localhost:8080", icon="🚀", top=6.3)

    set_notes(slide1, """Good morning Dr. Li and classmates! Today, I am presenting my senior capstone project on a fundamental problem in modern Natural Language Processing: Can a machine read long, complicated news stories and write accurate, natural summaries in its own words?
In this solo project, I conducted an end-to-end comparative study between two heavyweight transformer architectures: Meta's BART and Google's T5.
I designed a controlled experimental pipeline using 12,000 CNN/DailyMail articles, eliminated data leakage with seed 42, benchmarked quantitative ROUGE scores, measured millisecond latency and GPU memory, conducted a double-blind human error analysis, and built a live Gradio demonstration app.""")

    # =========================================================================
    # SLIDE 2: THE TWO PARADIGMS
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2)
    add_header(slide2, "ACT I // SCENE 01: THE FOUNDATIONS", "What is Abstractive Summarization?", "Highlighter vs. Note-Taking")

    card_left = add_card(slide2, 0.8, 1.6, 5.7, 4.4, border_accent=GOLD)
    tf_l = card_left.text_frame
    tf_l.margin_left = tf_l.margin_right = Inches(0.3)
    tf_l.margin_top = Inches(0.3)
    p = tf_l.paragraphs[0]
    p.text = "🖍️ Extractive Summarization (The Old Way)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = GOLD
    p2 = tf_l.add_paragraph()
    p2.text = "Analogy: A student highlighting lines in a textbook with a yellow marker."
    p2.font.size = Pt(11)
    p2.font.italic = True
    p2.font.color.rgb = TEXT_MUTED
    p3 = tf_l.add_paragraph()
    p3.text = "\n• How It Works: Scores existing sentences by keyword frequency and pulls top 3 verbatim."
    p3.font.size = Pt(12)
    p3.font.color.rgb = TEXT_BODY
    p4 = tf_l.add_paragraph()
    p4.text = "• Core Strength: Grammatically guaranteed because professional journalists wrote every word."
    p4.font.size = Pt(12)
    p4.font.color.rgb = TEXT_BODY
    p5 = tf_l.add_paragraph()
    p5.text = "• Fatal Flaw: Choppy and disjointed; completely incapable of combining two related facts into a single novel sentence."
    p5.font.size = Pt(12)
    p5.font.color.rgb = CRIMSON

    card_right = add_card(slide2, 6.8, 1.6, 5.7, 4.4, border_accent=CYAN)
    tf_r = card_right.text_frame
    tf_r.margin_left = tf_r.margin_right = Inches(0.3)
    tf_r.margin_top = Inches(0.3)
    p = tf_r.paragraphs[0]
    p.text = "✍️ Abstractive Summarization (My Project)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN
    p2 = tf_r.add_paragraph()
    p2.text = "Analogy: A student reading a chapter and synthesizing study notes from scratch."
    p2.font.size = Pt(11)
    p2.font.italic = True
    p2.font.color.rgb = TEXT_MUTED
    p3 = tf_r.add_paragraph()
    p3.text = "\n• How It Works: Compresses text into high-dimensional semantic vectors and writes new words token-by-token."
    p3.font.size = Pt(12)
    p3.font.color.rgb = TEXT_BODY
    p4 = tf_r.add_paragraph()
    p4.text = "• Core Strength: Seamless, flowing prose that condenses multiple events across 1,000 words into a punchy paragraph."
    p4.font.size = Pt(12)
    p4.font.color.rgb = EMERALD
    p5 = tf_r.add_paragraph()
    p5.text = "• The Grand Challenge: Hallucination risks! Because the model writes novel sentences, it can invent plausible-sounding falsehoods."
    p5.font.size = Pt(12)
    p5.font.color.rgb = GOLD

    add_banner(slide2, "My Research Goal: Produce accurate, publication-grade 3-sentence summaries in under 1 second using BART & T5.", icon="🎯")

    set_notes(slide2, """In natural language processing, there are two distinct ways a computer can summarize:
The traditional method is Extractive Summarization—just like taking a yellow highlighter to a textbook. It extracts existing sentences verbatim. It is safe, but reads like a robotic Frankenstein monster.
What I investigated in my capstone is Abstractive Summarization. Think of a bright student reading an entire chapter and synthesizing their own study notes. The AI parses the source text, builds high-dimensional semantic representations, and generates brand-new sentences from scratch.
My goal was to take 1,000-word breaking news articles from CNN and generate concise, accurate 3-sentence summaries in under a second.""")

    # =========================================================================
    # SLIDE 3: THREE GRAND PITFALLS
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3)
    add_header(slide3, "ACT I // SCENE 02: THE VULNERABILITIES", "Why is Abstractive Summarization Hard?", "Probability Calculators vs. Objective Truth")

    pitfalls = [
        ("1. Extrinsic Hallucination", "Inventing False Facts", CRIMSON,
         "The AI generates plausible-sounding details never present in the source text.\n\nExample: In my test set, the AI read about Crystal Palace hosting Manchester City at Selhurst Park, but hallucinated that the match was at 'The Emirates' (Arsenal's stadium) because of high-frequency soccer associations in pre-training."),
        ("2. Salience Distortion", "Missing the Core Headline", GOLD,
         "The model fixates on minor background fluff and omits the critical news climax.\n\nExample: An article detailing a 73-year-old reserve deputy shooting a suspect led the AI to spend 80 words describing procedural training paperwork, completely omitting the manslaughter charge!"),
        ("3. Degenerate Repetition", "The Infinite Loop Trap", PURPLE,
         "Left unconstrained, autoregressive decoders get trapped in cyclic probability feedback loops.\n\nExample: Phrases like 'in the city, in the city, in the city' cycling endlessly unless explicit mathematical n-gram blocking is enforced in beam search.")
    ]

    for idx, (title, sub, color, desc) in enumerate(pitfalls):
        c = add_card(slide3, 0.8 + idx * 4.0, 1.6, 3.73, 4.4, border_accent=color)
        tf = c.text_frame
        tf.margin_left = tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.25)
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = color
        p2 = tf.add_paragraph()
        p2.text = sub
        p2.font.size = Pt(10)
        p2.font.italic = True
        p2.font.color.rgb = TEXT_MUTED
        p3 = tf.add_paragraph()
        p3.text = f"\n{desc}"
        p3.font.size = Pt(11)
        p3.font.color.rgb = TEXT_BODY

    add_banner(slide3, "The Central Thesis: Transformers are probability calculators; high word-overlap does NOT guarantee factual truth.", icon="⚠️")

    set_notes(slide3, """Why is abstractive summarization so difficult? Because modern language models are essentially giant probability calculators. They predict the next most likely token, but have no innate concept of objective truth.
This exposes three major pitfalls:
First, Hallucination: The model invents plausible facts from pre-training memory—like placing a soccer match at the wrong stadium.
Second, Salience Distortion: The model gets distracted by secondary details and completely forgets the defining climax of the news event.
Third, Repetition Loops: Without explicit decoding penalties, decoders fall into repetitive stutter loops.
To test how modern architectures conquer these hurdles, I benchmarked two distinct transformer designs: BART and T5.""")

    # =========================================================================
    # SLIDE 4: ARCHITECTURAL HERITAGE: BART VS T5
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4)
    add_header(slide4, "ACT II // SCENE 01: ARCHITECTURAL ROOTS", "Two Different Transformer Philosophies", "BART vs. T5 Pre-Training")

    card_bart = add_card(slide4, 0.8, 1.6, 5.7, 4.4, border_accent=CYAN)
    tf_b = card_bart.text_frame
    tf_b.margin_left = tf_b.margin_right = Inches(0.3)
    tf_b.margin_top = Inches(0.25)
    p = tf_b.paragraphs[0]
    p.text = "🏛️ Meta BART-base (~140M Parameters)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = CYAN
    p2 = tf_b.add_paragraph()
    p2.text = "Denoising Sequence-to-Sequence Autoencoder"
    p2.font.size = Pt(11)
    p2.font.italic = True
    p2.font.color.rgb = TEXT_MUTED
    p3 = tf_b.add_paragraph()
    p3.text = "\n• Architecture: Bidirectional BERT-style encoder (reads left & right) paired with Autogressive GPT-style decoder."
    p3.font.size = Pt(11.5)
    p3.font.color.rgb = TEXT_BODY
    p4 = tf_b.add_paragraph()
    p4.text = "• Pre-Training: Corrupted clean text with mask infilling, token deletion, and sentence shuffling, forcing BART to reconstruct the original clean document."
    p4.font.size = Pt(11.5)
    p4.font.color.rgb = TEXT_BODY
    p5 = tf_b.add_paragraph()
    p5.text = "• Summarization Intuition: Inherently trained to edit, reorder, and restore text. Highly efficient for concise news synthesis."
    p5.font.size = Pt(11.5)
    p5.font.color.rgb = EMERALD

    card_t5 = add_card(slide4, 6.8, 1.6, 5.7, 4.4, border_accent=EMERALD)
    tf_t5 = card_t5.text_frame
    tf_t5.margin_left = tf_t5.margin_right = Inches(0.3)
    tf_t5.margin_top = Inches(0.25)
    p = tf_t5.paragraphs[0]
    p.text = "🌍 Google T5-base (~220M Parameters)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = EMERALD
    p2 = tf_t5.add_paragraph()
    p2.text = "Unified Text-to-Text Transfer Transformer"
    p2.font.size = Pt(11)
    p2.font.italic = True
    p2.font.color.rgb = TEXT_MUTED
    p3 = tf_t5.add_paragraph()
    p3.text = "\n• Architecture: Fully unified encoder-decoder treating every single NLP task as text-in, text-out via natural language prefixes."
    p3.font.size = Pt(11.5)
    p3.font.color.rgb = TEXT_BODY
    p4 = tf_t5.add_paragraph()
    p4.text = "• Task Conditioning: Requires prepending 'summarize: ' to the input. Same model can do translation, QA, or sentiment by changing the prefix."
    p4.font.size = Pt(11.5)
    p4.font.color.rgb = TEXT_BODY
    p5 = tf_t5.add_paragraph()
    p5.text = "• Capacity: 220 million parameters (+57.1% larger than BART), 12 encoder and 12 decoder layers for rich contextual modeling."
    p5.font.size = Pt(11.5)
    p5.font.color.rgb = CYAN

    add_banner(slide4, "Attention Intuition: In 'The boy dropped the glass because he was careless', self-attention links 'he' directly back to 'the boy'!", icon="💡")

    set_notes(slide4, """To address these challenges, I compared two leading transformer philosophies:
Meta's BART combines a bidirectional encoder like BERT with an autoregressive decoder like GPT. BART was pre-trained as a denoising autoencoder: researchers scrambled sentences, masked tokens, and deleted words, training BART to reconstruct clean text. This makes BART an instinctual text editor.
Google's T5 reframes all NLP tasks as text-to-text. By prepending the natural language prefix 'summarize: ', T5 executes abstractive summarization through the same unified architecture it uses for translation and question answering. It is 57% larger with 220 million parameters.""")

    # =========================================================================
    # SLIDE 5: ARCHITECTURAL SPECS & BASELINE ADVANTAGE
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5)
    add_header(slide5, "ACT II // SCENE 02: THE HARDWARE BLUEPRINT", "Architectural Specifications & The Baseline Head Start", "140M vs. 220M Parameters")

    # Table of specs
    rows = [
        ("Parameter Count", "~140 Million", "~220 Million (+57.1%)", "T5 has higher representational capacity but higher compute cost"),
        ("Encoder / Decoder Depth", "6 encoder / 6 decoder", "12 encoder / 12 decoder", "T5 has 2× deeper layer stack for hierarchical abstraction"),
        ("Hidden Dimension / Heads", "768 dim / 12 heads", "768 dim / 12 heads", "Identical representation width per attention block"),
        ("Tokenizer & Vocabulary", "Byte-Pair Encoding (50,265)", "SentencePiece (32,128)", "BART uses byte fallbacks; T5 uses unigram subwords"),
        ("Training Precision", "fp16 Mixed Precision", "fp32 Full Precision", "T5 requires fp32 to prevent mathematical instability / NaN loss"),
        ("Pretrained Zero-Shot R-1", "27.87%", "30.43% (+2.56%)", "T5 started with a pre-training head start due to C4 multi-task mix")
    ]

    t_shape = slide5.shapes.add_table(7, 4, Inches(0.8), Inches(1.6), Inches(11.733), Inches(3.6))
    table = t_shape.table
    table.columns[0].width = Inches(2.5)
    table.columns[1].width = Inches(2.2)
    table.columns[2].width = Inches(2.4)
    table.columns[3].width = Inches(4.633)

    headers = ["Dimension", "Meta BART-base", "Google T5-base", "Experimental Impact in My Study"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(18, 25, 38)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = FONT_BODY
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = GOLD

    for row_idx, r in enumerate(rows):
        for col_idx, val in enumerate(r):
            cell = table.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG if row_idx % 2 == 0 else RGBColor(16, 22, 32)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_BODY
            p.font.size = Pt(10)
            if col_idx == 1:
                p.font.color.rgb = CYAN
                p.font.bold = True
            elif col_idx == 2:
                p.font.color.rgb = EMERALD
                p.font.bold = True
            elif col_idx == 0:
                p.font.color.rgb = TEXT_WHITE
                p.font.bold = True
            else:
                p.font.color.rgb = TEXT_MUTED

    add_banner(slide5, "Why T5 Led at Zero-Shot (30.43% vs 27.87%): Google's pretraining mix included supervised news tasks. BART started from pure denoising!", icon="⭐", top=5.65)

    set_notes(slide5, """Here are the architectural specifications.
Notice the scale difference: T5-base is about 57% larger with 220 million parameters and twice the encoder/decoder depth (12 layers vs 6).
Now, here is a critical conceptual finding:
Before fine-tuning, T5 scored 30.43% ROUGE-1 compared to BART's 27.87%.
Why did T5 have this initial head start?
Because Google's pre-training for T5 included multi-task supervised NLP datasets, which already exposed it to CNN/DailyMail formats. BART had only seen general corrupted text.
T5 started with an advantage, but the true test is how both models adapt after I train them on the exact same 12,000 articles.""")

    # =========================================================================
    # SLIDE 6: EXPERIMENTAL RIGOR & DATA HYGIENE
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6)
    add_header(slide6, "ACT III // SCENE 01: THE EXPERIMENTAL DESIGN", "Training on 12,000 News Articles & Data Hygiene", "CNN/DailyMail Benchmark & Seed 42")

    methods = [
        ("📰 CNN/DailyMail 3.0.0", GOLD,
         "Canonical non-anonymized benchmark of real news stories paired with human journalist bullet points.\n\n• Full splits: 287k train, 13.3k val, 11.5k test.\n• Highly challenging abstractive dataset with multi-sentence ground truths."),
        ("🔒 Zero Leakage Protocol", CYAN,
         "Strict subset isolation to eliminate data leakage and selection bias.\n\n• Sampled exactly 12,000 train / 1,000 val / 1,000 test.\n• Locked with Random Seed 42 serialized to disk (`indices_train.json`).\n• Zero overlap between training and test sets."),
        ("📐 Inverted Pyramid (512 Cap)", PURPLE,
         "Standard journalism convention frontloads the 5 W's in the opening paragraphs.\n\n• Input capped at 512 tokens; target at 128 tokens.\n• Captures >90% of critical facts while preventing GPU OOM crashes from O(N²) self-attention.")
    ]

    for idx, (title, color, text) in enumerate(methods):
        c = add_card(slide6, 0.8 + idx * 4.0, 1.6, 3.73, 4.4, border_accent=color)
        tf = c.text_frame
        tf.margin_left = tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.25)
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = color
        p2 = tf.add_paragraph()
        p2.text = f"\n{text}"
        p2.font.size = Pt(11)
        p2.font.color.rgb = TEXT_BODY

    add_banner(slide6, "Fairness Guarantee: Both models trained under identical effective batch size (32), optimizer (AdamW), and evaluation steps.", icon="⚖️")

    set_notes(slide6, """To ensure a completely fair scientific comparison, I enforced strict experimental hygiene:
First, I utilized the canonical CNN/DailyMail dataset of real journalistic reporting.
Second, I locked random seed 42 and serialized indices to disk, guaranteeing zero data leakage between training and testing.
Third, I capped input articles at 512 tokens. In journalism, the 'Inverted Pyramid' frontloads the critical who, what, when, and where into the lead paragraphs. Truncating at 512 tokens captures over 90% of salient facts while keeping quadratic self-attention within GPU memory limits.""")

    # =========================================================================
    # SLIDE 7: DECODING MECHANICS
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7)
    add_header(slide7, "ACT III // SCENE 02: GENERATION DYNAMICS", "Decoding Strategy: How Decoders Think", "Greedy vs. Beam Search & Repetition Penalties")

    c1 = add_card(slide7, 0.8, 1.6, 5.7, 2.3, border_accent=GOLD)
    tf1 = c1.text_frame
    tf1.margin_left = tf1.margin_right = Inches(0.25)
    tf1.margin_top = Inches(0.2)
    p = tf1.paragraphs[0]
    p.text = "⚡ Greedy Search (num_beams = 1)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = GOLD
    p2 = tf1.add_paragraph()
    p2.text = "Picks the single highest-probability next token at each step.\n• Pro: Lightning fast.\n• Con: Extremely short-sighted; easily gets trapped in repetitive dead-ends."
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = TEXT_BODY

    c2 = add_card(slide7, 6.8, 1.6, 5.7, 2.3, border_accent=CYAN)
    tf2 = c2.text_frame
    tf2.margin_left = tf2.margin_right = Inches(0.25)
    tf2.margin_top = Inches(0.2)
    p = tf2.paragraphs[0]
    p.text = "🔭 Beam Search (num_beams = 4 — MY CHOICE)"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = CYAN
    p2 = tf2.add_paragraph()
    p2.text = "Maintains 4 candidate hypotheses in parallel, pruning weak paths at each step.\n• Pro: Produces cohesive, globally optimal paragraph structures.\n• Trade-off: ~3× more compute operations per token generated."
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = TEXT_BODY

    # Parameter Table
    t_shape = slide7.shapes.add_table(5, 3, Inches(0.8), Inches(4.1), Inches(11.733), Inches(2.0))
    table = t_shape.table
    table.columns[0].width = Inches(2.5)
    table.columns[1].width = Inches(2.2)
    table.columns[2].width = Inches(7.033)

    params = [
        ("Parameter", "Fixed Value", "Engineering Purpose in My Pipeline"),
        ("num_beams", "4", "Explores 4 alternative generation paths simultaneously like a chess engine"),
        ("min_length / max_length", "30 / 128", "Guarantees complete 3–4 sentence news summaries without premature cutoff"),
        ("length_penalty", "2.0", "Mathematically encourages models to complete full thoughts instead of stopping short"),
        ("no_repeat_ngram_size", "3", "Bans repeating any 3-word phrase -> Achieved 0.0% repetition loop rate across 1,000 articles!")
    ]

    for r_idx, r in enumerate(params):
        for c_idx, val in enumerate(r):
            cell = table.cell(r_idx, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(18, 25, 38) if r_idx == 0 else CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_BODY
            p.font.size = Pt(10)
            if r_idx == 0:
                p.font.bold = True
                p.font.color.rgb = GOLD
            elif c_idx == 0:
                p.font.bold = True
                p.font.color.rgb = CYAN
            elif c_idx == 1:
                p.font.bold = True
                p.font.color.rgb = EMERALD
            else:
                p.font.color.rgb = TEXT_BODY

    add_banner(slide7, "Repetition Breakthrough: `no_repeat_ngram_size=3` completely eliminated loops across all 1,000 test samples.", icon="🛡️", top=6.25)

    set_notes(slide7, """How does an AI actually choose which words to write? This is called Decoding.
Greedy Search picks the single most probable word at each step. It is fast, but short-sighted and frequently falls into degenerate loops.
In my pipeline, I locked Beam Search with 4 beams. It acts like a chess player, tracking 4 parallel candidate sentences before committing to the best path.
Crucially, I enforced `no_repeat_ngram_size = 3`. This mathematically forbids the model from repeating any 3-word sequence. Across all 1,000 unseen test articles, this achieved an exact 0.0% repetition loop rate!""")

    # =========================================================================
    # SLIDE 8: EVALUATION METRICS & THE COURTROOM TRAP
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8)
    add_header(slide8, "ACT IV // SCENE 01: THE EVALUATION SYSTEM", "Grading the AI: What is ROUGE? & The Fatal Trap", "Word-Matching vs. Factual Truth")

    metrics = [
        ("ROUGE-1", "Single-Word Overlap", CYAN, "Measures unigram recall. Captures whether the model picked up essential topical words and entity names."),
        ("ROUGE-2", "Two-Word Phrases", EMERALD, "Measures bigram recall (e.g. 'White House'). Captures phrasing fluency and multi-word concepts."),
        ("ROUGE-L", "Longest Subsequence", PURPLE, "Measures sentence-level structure by identifying longest word sequence in exact order."),
        ("ROUGE-Lsum", "Discourse Summary", GOLD, "Calculates ROUGE-L across all sentence splits. The definitive test for multi-sentence news summaries.")
    ]

    for idx, (m, sub, color, desc) in enumerate(metrics):
        c = add_card(slide8, 0.8 + idx * 3.0, 1.6, 2.73, 2.3, border_accent=color)
        tf = c.text_frame
        tf.margin_left = tf.margin_right = Inches(0.2)
        tf.margin_top = Inches(0.2)
        p = tf.paragraphs[0]
        p.text = m
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = color
        p2 = tf.add_paragraph()
        p2.text = sub
        p2.font.size = Pt(9.5)
        p2.font.italic = True
        p2.font.color.rgb = TEXT_MUTED
        p3 = tf.add_paragraph()
        p3.text = f"\n{desc}"
        p3.font.size = Pt(10)
        p3.font.color.rgb = TEXT_BODY

    # Courtroom Trap Card
    court_card = add_card(slide8, 0.8, 4.1, 11.733, 1.9, border_accent=CRIMSON)
    tf_c = court_card.text_frame
    tf_c.margin_left = tf_c.margin_right = Inches(0.3)
    tf_c.margin_top = Inches(0.2)
    p = tf_c.paragraphs[0]
    p.text = "🚨 The 'Courtroom Trap': Why Automated Metrics Alone Are Dangerous"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = CRIMSON
    p2 = tf_c.add_paragraph()
    p2.text = "• Human Journalist Reference:  'The senator voted NO on the historic tax reform bill.'\n• Flawed AI Model Summary:     'The senator voted YES on the historic tax reform bill.'\n\n• The Fatal Blindspot: 6 out of 7 words match perfectly! ROUGE awards an astonishing 85.7% score!\n• The Reality: The summary is 100% factually false and defamatory! This is why I conducted a blind human study."
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = TEXT_BODY

    add_banner(slide8, "Key Capstone Lesson: High ROUGE scores do NOT guarantee factual accuracy. Human evaluation is mandatory.", icon="⚖️")

    set_notes(slide8, """How do computer scientists grade an AI summary? We use ROUGE.
ROUGE is essentially an n-gram word-matching game against a human journalist's reference:
ROUGE-1 measures single words; ROUGE-2 measures pairs of words like 'White House'; and ROUGE-L measures sentence structure.
However, look at the Courtroom Trap on this slide:
If a journalist writes 'The senator voted NO on the bill' and the AI writes 'The senator voted YES on the bill'...
6 out of 7 words match! ROUGE awards an 85.7% score!
Yet the summary is 100% false! This is why automated metrics alone are dangerous, and why I designed a double-blind human evaluation protocol.""")

    # =========================================================================
    # SLIDE 9: QUANTITATIVE RESULTS SCOREBOARD
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9)
    add_header(slide9, "ACT IV // SCENE 02: THE SCOREBOARD", "Quantitative Results: Who Won the Numbers Game?", "1,000 Test Articles Evaluated")

    # Table of Results
    t_shape = slide9.shapes.add_table(7, 6, Inches(0.8), Inches(1.5), Inches(11.733), Inches(3.2))
    table = t_shape.table
    table.columns[0].width = Inches(2.2)
    table.columns[1].width = Inches(1.8)
    table.columns[2].width = Inches(1.8)
    table.columns[3].width = Inches(2.0)
    table.columns[4].width = Inches(1.8)
    table.columns[5].width = Inches(2.133)

    score_headers = ["Model", "Metric", "Zero-Shot Baseline", "After Fine-Tuning (1-Ep)", "Absolute Gain", "Relative Improvement"]
    for i, h in enumerate(score_headers):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(18, 25, 38)
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = GOLD

    score_rows = [
        ("BART-base", "ROUGE-1", "27.87%", "40.23%", "+12.36", "+44.3%"),
        ("BART-base", "ROUGE-2", "9.95%", "17.71%", "+7.76", "+78.0%"),
        ("BART-base", "ROUGE-Lsum", "21.27%", "37.01% (WINNER)", "+15.74", "+74.0%"),
        ("T5-base", "ROUGE-1", "30.43%", "42.88% (WINNER)", "+12.45", "+40.9%"),
        ("T5-base", "ROUGE-2", "10.40%", "20.24% (WINNER)", "+9.84", "+94.6%"),
        ("T5-base", "ROUGE-Lsum", "24.57%", "36.67%", "+12.10", "+49.2%")
    ]

    for r_idx, r in enumerate(score_rows):
        for c_idx, val in enumerate(r):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = CARD_BG if r_idx < 3 else RGBColor(16, 22, 32)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            if "WINNER" in val:
                p.font.bold = True
                p.font.color.rgb = CYAN if "BART" in r[0] else EMERALD
            elif c_idx == 4 or c_idx == 5:
                p.font.color.rgb = EMERALD
                p.font.bold = True
            elif c_idx == 0:
                p.font.bold = True
                p.font.color.rgb = CYAN if "BART" in val else EMERALD
            else:
                p.font.color.rgb = TEXT_BODY

    # Cloud Scaling Callout
    cloud_card = add_card(slide9, 0.8, 4.85, 11.733, 1.25, border_accent=CYAN)
    tf_cl = cloud_card.text_frame
    tf_cl.margin_left = tf_cl.margin_right = Inches(0.25)
    tf_cl.margin_top = Inches(0.15)
    p = tf_cl.paragraphs[0]
    p.text = "☁️ Multi-Epoch Cloud Scaling Study (Google Colab & Kaggle Tesla T4 GPUs)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CYAN
    p2 = tf_cl.add_paragraph()
    p2.text = "When I scaled BART to 3 full epochs on Google Colab, metrics leaped further: ROUGE-1 reached 40.86% (+0.63%), ROUGE-2 reached 18.44% (+0.73%), and ROUGE-Lsum climbed to 37.95% (+0.94%), with latency dropping to 0.296s. On Kaggle, T5 3-epoch validation loss converged cleanly to 3.04."
    p2.font.size = Pt(10)
    p2.font.color.rgb = TEXT_BODY

    add_banner(slide9, "Summary Takeaway: Both models jumped +12.4 points. T5 won raw vocabulary; BART won multi-sentence summary structure.", icon="🏆", top=6.25)

    set_notes(slide9, """Here are the empirical results across 1,000 brand-new, unseen news articles:
First, fine-tuning was an overwhelming success: both models leaped by +12.4 percentage points in ROUGE-1 after learning on the 12,000 articles.
If you look strictly at n-gram recall, T5 won the vocabulary game: reaching 42.88% ROUGE-1 and 20.24% ROUGE-2.
However, look at ROUGE-Lsum: BART beat T5 with 37.01% versus 36.67%!
This proves that BART structures sentence boundaries better for multi-sentence news summaries.
Furthermore, in my extended 3-epoch cloud scaling study on Google Colab, BART climbed further to 40.86% ROUGE-1 and 37.95% ROUGE-Lsum, proving steady scaling gains with ultra-fast 0.296s latency.""")

    # =========================================================================
    # SLIDE 10: HARDWARE TELEMETRY & EFFICIENCY
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10)
    add_header(slide10, "ACT IV // SCENE 03: THE TELEMETRY", "The Hardware Reality: Speed, Throughput & VRAM", "NVIDIA Tesla T4 GPU Benchmarks")

    stats = [
        ("BART Inference Latency", "0.309s", "1.84× Faster than T5", CYAN),
        ("T5 Inference Latency", "0.569s", "57% more parameters", EMERALD),
        ("BART Training Duration", "11m 30s", "17.38 samples/s (fp16)", CYAN),
        ("T5 Training Duration", "21m 52s", "9.14 samples/s (fp32)", EMERALD)
    ]

    for idx, (title, val, sub, color) in enumerate(stats):
        c = add_card(slide10, 0.8 + idx * 3.0, 1.6, 2.73, 2.1, border_accent=color)
        tf = c.text_frame
        tf.margin_left = tf.margin_right = Inches(0.2)
        tf.margin_top = Inches(0.2)
        p = tf.paragraphs[0]
        p.text = title.upper()
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = TEXT_MUTED
        p2 = tf.add_paragraph()
        p2.text = val
        p2.font.size = Pt(24)
        p2.font.bold = True
        p2.font.color.rgb = color
        p3 = tf.add_paragraph()
        p3.text = sub
        p3.font.size = Pt(10)
        p3.font.bold = True
        p3.font.color.rgb = TEXT_WHITE

    # Table of hardware dims
    t_shape = slide10.shapes.add_table(4, 4, Inches(0.8), Inches(3.9), Inches(11.733), Inches(2.1))
    table = t_shape.table
    table.columns[0].width = Inches(2.6)
    table.columns[1].width = Inches(2.3)
    table.columns[2].width = Inches(2.3)
    table.columns[3].width = Inches(4.533)

    hw_rows = [
        ("Operational Dimension", "Meta BART-base", "Google T5-base", "Production Deployment Impact"),
        ("Average Summary Length", "83.1 tokens (~3 sentences)", "100.3 tokens (~4 sentences)", "BART cuts straight to the headline; T5 generates compound clauses"),
        ("Numerical Precision", "fp16 Mixed Precision", "fp32 Full Precision", "BART fully utilizes Tensor Cores for 2× faster training"),
        ("Peak VRAM Footprint", "1,878 MB (1.83 GB)", "1,898 MB (1.85 GB)", "Both models easily fit within free 16 GB cloud GPUs")
    ]

    for r_idx, r in enumerate(hw_rows):
        for c_idx, val in enumerate(r):
            cell = table.cell(r_idx, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(18, 25, 38) if r_idx == 0 else CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            if r_idx == 0:
                p.font.bold = True
                p.font.color.rgb = GOLD
            elif c_idx == 1:
                p.font.color.rgb = CYAN
                p.font.bold = True
            elif c_idx == 2:
                p.font.color.rgb = EMERALD
                p.font.bold = True
            else:
                p.font.color.rgb = TEXT_BODY

    add_banner(slide10, "Production Takeaway: In a live API, BART cuts cloud hosting costs by ~45% while responding almost twice as fast!", icon="⚡", top=6.25)

    set_notes(slide10, """However, computational efficiency tells a completely different story:
BART generated summaries in 0.31 seconds per article—nearly twice as fast as T5 (0.57 seconds).
During training, BART finished in just 11.5 minutes, whereas T5 took 22 minutes.
BART produced punchy, concise summaries averaging 83 words, while T5 wrote wordier, compound sentences averaging 100 words.
If you are deploying a real-time web application where an end user expects instantaneous responses, BART's speed advantage is enormous.""")

    # =========================================================================
    # SLIDE 11: ENGINEERING CHALLENGES (THREE BUGS I SOLVED)
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide11)
    add_header(slide11, "ACT V // SCENE 01: BEHIND THE CURTAIN", "Engineering Challenges & The Three Bugs I Solved", "Debugging Transformers on Real Hardware")

    bugs = [
        ("💥 1. T5 fp16 NaN Loss at Step 42", CRIMSON,
         "• Symptom: Training loss suddenly exploded to NaN at step 42 on the T4 GPU.\n• Root Cause: T5 uses RMSNorm without additive bias. In float16, numbers cap at 65,504. Gradient accumulation caused numerical overflow.\n• My Solution: Enforced `fp16: False` (fp32) for T5, restoring rock-solid numerical stability."),
        ("💾 2. CUDA Out-Of-Memory", PURPLE,
         "• Symptom: Articles exceeding 1,500 words caused GPU allocation crashes.\n• Root Cause: Self-attention scales quadratically: O(N²). Long sequences blow past tensor memory buffers.\n• My Solution: Applied journalistic Inverted Pyramid truncation at 512 tokens, keeping peak memory under 1.9 GB."),
        ("🔁 3. Repetition Loops", GOLD,
         "• Symptom: Early models repeated phrases (e.g. 'in the city, in the city').\n• Root Cause: Autoregressive decoding falling into local probability traps.\n• My Solution: Injected `no_repeat_ngram_size = 3` into beam search, achieving a 0.0% repetition loop rate across all 1,000 articles.")
    ]

    for idx, (title, color, text) in enumerate(bugs):
        c = add_card(slide11, 0.8 + idx * 4.0, 1.6, 3.73, 4.4, border_accent=color)
        tf = c.text_frame
        tf.margin_left = tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.25)
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = color
        p2 = tf.add_paragraph()
        p2.text = f"\n{text}"
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = TEXT_BODY

    add_banner(slide11, "The Engineer's Lesson: You cannot treat transformers as identical black boxes. Normalization layers dictate stability!", icon="🛠️")

    set_notes(slide11, """In real-world machine learning, models rarely train smoothly on the first try. In this capstone, I diagnosed and solved three critical engineering bugs:
Bug 1: T5 fp16 NaN Loss. When I launched T5 with fp16 mixed precision on the T4 GPU, the training loss suddenly exploded to NaN at step 42. T5 uses RMSNorm without additive bias, causing float16 overflow past 65,504. I fixed it by switching T5 to fp32 full precision.
Bug 2: CUDA Out-Of-Memory. Articles over 1,500 words crashed PyTorch due to quadratic O(N squared) attention. Capping at 512 tokens solved it.
Bug 3: Repetition Loops. Early models repeated phrases like 'in the city'. Adding 3-gram blocking to beam search killed loops completely.""")

    # =========================================================================
    # SLIDE 12: DOUBLE-BLIND HUMAN STUDY
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide12)
    add_header(slide12, "ACT V // SCENE 02: THE HUMAN JURY", "Why Numbers Lie: Double-Blind Human Study", "25 Articles Rated Blindly on a 1–5 Scale")

    # Table of ratings
    t_shape = slide12.shapes.add_table(5, 4, Inches(0.8), Inches(1.6), Inches(5.7), Inches(2.6))
    table = t_shape.table
    table.columns[0].width = Inches(2.2)
    table.columns[1].width = Inches(1.1)
    table.columns[2].width = Inches(1.1)
    table.columns[3].width = Inches(1.3)

    h_rows = [
        ("Human Dimension (1–5)", "BART", "T5", "Study Winner"),
        ("Event Coverage", "4.24", "3.88", "🏆 BART (+0.36)"),
        ("Factual Consistency", "4.88", "4.68", "🏆 BART (+0.20)"),
        ("Readability & Fluency", "4.32", "4.40", "🏆 T5 (+0.08)"),
        ("Conciseness / Redundancy", "4.00", "4.16", "🏆 T5 (+0.16)")
    ]

    for r_idx, r in enumerate(h_rows):
        for c_idx, val in enumerate(r):
            cell = table.cell(r_idx, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(18, 25, 38) if r_idx == 0 else CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(10)
            if r_idx == 0:
                p.font.bold = True
                p.font.color.rgb = GOLD
            elif "BART" in val:
                p.font.color.rgb = CYAN
                p.font.bold = True
            elif "T5" in val:
                p.font.color.rgb = EMERALD
                p.font.bold = True
            else:
                p.font.color.rgb = TEXT_BODY

    # Error Taxonomy Breakdown
    c_tax = add_card(slide12, 6.8, 1.6, 5.7, 4.4, border_accent=GOLD)
    tf_tax = c_tax.text_frame
    tf_tax.margin_left = tf_tax.margin_right = Inches(0.3)
    tf_tax.margin_top = Inches(0.2)
    p = tf_tax.paragraphs[0]
    p.text = "🔍 Error Taxonomy Observed Across 25 Samples"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = GOLD
    p2 = tf_tax.add_paragraph()
    p2.text = "\n• Critical Fact Omission (OM): BART had 5; T5 had 8.\n  Root Cause: Running out of space under the 128-token cap or stopping early.\n\n• Extrinsic Hallucination (EH): BART had 1; T5 had 3.\n  Root Cause: Pre-training memories leaking into unfamiliar domains.\n\n• Wrong Entity Attribution (WE): BART had 1; T5 had 1.\n  Root Cause: Swapping names between secondary and primary actors.\n\n• Repetition Loops (REP): 0 occurrences across both models!\n  Root Cause: Ban on 3-grams completely eliminated looping."
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = TEXT_BODY

    # Evaluation Methodology Card
    c_m = add_card(slide12, 0.8, 4.4, 5.7, 1.6, border_accent=PURPLE)
    tf_m = c_m.text_frame
    tf_m.margin_left = tf_m.margin_right = Inches(0.25)
    tf_m.margin_top = Inches(0.15)
    p = tf_m.paragraphs[0]
    p.text = "Double-Blind Protocol Methodology"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = PURPLE
    p2 = tf_m.add_paragraph()
    p2.text = "I anonymized model outputs into 'Summary A' and 'Summary B' with identities stored in a locked JSON key file. After grading 25 randomized articles blind on a 1–5 scale, I executed the deblinding script."
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = TEXT_MUTED

    add_banner(slide12, "Surprise Finding: BART achieved higher factual consistency (4.88 vs. 4.68) despite T5's higher ROUGE-1!", icon="💡")

    set_notes(slide12, """My study proves a famous truth in NLP: High ROUGE scores do not guarantee true facts.
To test this scientifically, I conducted a double-blind human evaluation on 25 randomized articles. I anonymized the summaries into 'Summary A' and 'Summary B' with model names locked in a key file so I couldn't be biased.
I graded each summary from 1 to 5 stars on coverage, factuality, fluency, and conciseness.
When I unlocked the key, the surprise was that BART actually beat T5 in factual accuracy (4.88 vs. 4.68) and main story coverage (4.24 vs. 3.88).
T5 suffered from 3 factual hallucinations, while BART only had 1. The dominant error for both was omission due to length caps.""")

    # =========================================================================
    # SLIDE 13: SPOTLIGHT CASE STUDIES
    # =========================================================================
    slide13 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide13)
    add_header(slide13, "ACT V // SCENE 03: CASE STUDIES", "The Anatomy of AI Mistakes: 3 Real Stories", "Spotlight Case Studies from My Live Gradio App")

    stories = [
        ("Story 1: Cape Town Airport", "T5 WINS (BART Hallucination)", EMERALD,
         "• Context: 15-year-old girl detained at South Africa airport en route to ISIS.\n\n• BART Flaw: Started with 'Virginia teen accused of being recruiter...'\n  (Hallucinated Virginia from another famous pretraining case!)\n\n• T5 Grounded: Accurately stated she was from Cape Town and detained in South Africa."),
        ("Story 2: Premier League Venue", "BART WINS (T5 Entity Confusion)", CYAN,
         "• Context: Crystal Palace hosting Manchester City at Selhurst Park.\n\n• T5 Flaw: Stated Palace would host City at 'the Emirates'!\n  (Emirates is Arsenal's stadium! A classic entity substitution).\n\n• BART Grounded: Got the match, stakes, and manager quotes 100% correct."),
        ("Story 3: Police Reserve Trial", "BART WINS (T5 Omission)", CYAN,
         "• Context: 73-year-old deputy confused gun for taser, killing suspect.\n\n• T5 Flaw: Spent 80 words describing training paperwork, completely omitting the manslaughter charge!\n\n• BART Grounded: Captured the lead headline immediately: 'Robert Bates charged with 2nd-degree manslaughter'.")
    ]

    for idx, (title, sub, color, text) in enumerate(stories):
        c = add_card(slide13, 0.8 + idx * 4.0, 1.6, 3.73, 4.4, border_accent=color)
        tf = c.text_frame
        tf.margin_left = tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.25)
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = color
        p2 = tf.add_paragraph()
        p2.text = sub
        p2.font.size = Pt(9.5)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_WHITE
        p3 = tf.add_paragraph()
        p3.text = f"\n{text}"
        p3.font.size = Pt(10.5)
        p3.font.color.rgb = TEXT_BODY

    add_banner(slide13, "Live Verification: All three case studies are pre-loaded in my Gradio web application for instant side-by-side inspection.", icon="🎬")

    set_notes(slide13, """Now let's examine the three spotlight stories from my live demo that demonstrate these AI mistakes in real life:
In Story 1 (Cape Town Airport): Look at what BART wrote: It starts with 'Virginia teen accused of being ISIS recruiter'—but this article took place in South Africa! BART suffered from an extrinsic hallucination by retrieving a memory from pre-training. T5 stayed completely faithful.
In Story 2 (Premier League): The tables turn! Crystal Palace hosted Manchester City. T5 hallucinated that the match was played at 'the Emirates'—which is Arsenal's stadium! BART got the venue and the manager's quotes 100% right.
In Story 3 (Police Shooting): Look at event coverage. BART captures the critical headline that the deputy was charged with second-degree manslaughter. T5 got lost describing background training paperwork and omitted the manslaughter charge completely!""")

    # =========================================================================
    # SLIDE 14: ORAL DEFENSE PREPARATION (DR. LI QUESTIONS)
    # =========================================================================
    slide14 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide14)
    add_header(slide14, "FINALE // SCENE 01: THE DEFENSE", "Oral Defense Mastery: Hardest Questions from Dr. Li", "Technical Mastery & Research Defense")

    questions = [
        ("Q1: Why did T5 achieve higher ROUGE but lower human factuality?", GOLD,
         "Answer: ROUGE measures n-gram overlap, not semantic truth. T5 generated longer summaries (100 tokens vs. BART's 83), giving it more chances to intersect with reference vocabulary. However, human evaluation showed BART captured events better (4.24 vs. 3.88) with fewer omissions and hallucinations."),
        ("Q2: Why did T5 produce NaN loss in fp16 on the GPU?", CRIMSON,
         "Answer: T5 relies on RMSNorm without additive bias. In IEEE 754 float16, the maximum finite value is 65,504. Backward operations caused overflow, generating NaN gradients at step 42. Setting `fp16: False` (fp32) restored numerical stability."),
        ("Q3: Why did you limit input articles to 512 tokens?", PURPLE,
         "Answer: Self-attention memory scales quadratically: O(N²). 1,024 tokens caused memory exhaustion on the GPU. Furthermore, journalism's Inverted Pyramid frontloads the 5 W's, making 512 tokens an optimal compute-accuracy tradeoff."),
        ("Bonus: What if you are asked something you don't know?", CYAN,
         "What I will say: 'That is a great question, Dr. Li. In my experiment, I evaluated on the 1,000-article test set under my fixed decoding config. I did not test that specific condition, but based on transformer theory, my hypothesis would be [...]. That would be an exciting direction for future work.'")
    ]

    for idx, (title, color, text) in enumerate(questions):
        row = idx // 2
        col = idx % 2
        c = add_card(slide14, 0.8 + col * 6.0, 1.6 + row * 2.2, 5.733, 2.05, border_accent=color)
        tf = c.text_frame
        tf.margin_left = tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.15)
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = color
        p2 = tf.add_paragraph()
        p2.text = f"\n{text}"
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = TEXT_BODY

    add_banner(slide14, "The Defense Rule: Never invent answers. Acknowledge the experimental boundary, state your hypothesis, and propose future work.", icon="🎓")

    set_notes(slide14, """Here is my defense preparation for the hardest questions Dr. Li might ask:
Question 1: Why did T5 get a higher ROUGE score but lower human factuality and coverage?
Answer: ROUGE measures word overlap. T5 wrote longer summaries (100 words vs BART's 83), giving it more opportunities to intersect with the human reference. But human evaluation showed BART captured the actual event better with fewer hallucinations.
Question 2: Why did T5 produce NaN loss in fp16?
Answer: T5 uses RMSNorm without additive bias, overflowing past 65,504 in float16. Switching to fp32 fixed it.
Question 3: Why did you limit to 512 tokens?
Answer: Self-attention scales quadratically O(N squared). 512 tokens captures the Inverted Pyramid's 5 Ws while preventing GPU memory crashes.
And if Dr. Li asks something outside my test, I will acknowledge the boundary and explain my theoretical hypothesis!""")

    # =========================================================================
    # SLIDE 15: THE VERDICT & THE MEMORY VAULT
    # =========================================================================
    slide15 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide15)
    add_header(slide15, "FINALE // SCENE 02: THE VERDICT", "Architectural Decision Matrix & The Memory Vault", "All Key Metrics in One Place")

    # Left: Deploy BART vs Deploy T5
    c_dec = add_card(slide15, 0.8, 1.6, 5.7, 4.4, border_accent=CYAN)
    tf_d = c_dec.text_frame
    tf_d.margin_left = tf_d.margin_right = Inches(0.25)
    tf_d.margin_top = Inches(0.2)
    p = tf_d.paragraphs[0]
    p.text = "🏆 When to Deploy Which Model"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = GOLD
    p2 = tf_d.add_paragraph()
    p2.text = "\nDeploy Meta BART-base If:\n• You need high throughput: 0.309s latency (1.84× faster).\n• You prioritize factual grounding: 4.88 / 5.0 factuality.\n• You want punchy, concise summaries: ~83 tokens.\n• You have limited compute: 11.5m training via fp16 Tensor Cores.\n\nDeploy Google T5-base If:\n• You want maximum n-gram vocabulary: 42.88% ROUGE-1.\n• You need longer, compound prose: 100 tokens, 4.40 fluency.\n• You want a unified multi-task system: translation, QA, and summarization in one model."
    p2.font.size = Pt(10)
    p2.font.color.rgb = TEXT_BODY

    # Right: Memory Vault Table
    t_shape = slide15.shapes.add_table(9, 3, Inches(6.8), Inches(1.6), Inches(5.733), Inches(4.4))
    table = t_shape.table
    table.columns[0].width = Inches(2.3)
    table.columns[1].width = Inches(1.7)
    table.columns[2].width = Inches(1.733)

    vault_rows = [
        ("Key Defense Metric", "BART-base", "T5-base"),
        ("Parameters", "~140 Million", "~220 Million (+57%)"),
        ("Baseline R-1 (Zero-Shot)", "27.87%", "30.43% (Winner)"),
        ("Final R-1 (Fine-Tuned)", "40.23%", "42.88% (Winner)"),
        ("Final R-Lsum (Structure)", "37.01% (Winner)", "36.67%"),
        ("3-Epoch Colab Scaling", "40.86% R-1 / 37.95% R-L", "Val loss: 3.04 (Kaggle)"),
        ("Inference Latency", "0.309s (1.84× faster)", "0.569s"),
        ("Training Duration", "11m 30s (fp16)", "21m 52s (fp32)"),
        ("Human Factuality Rating", "4.88 / 5.0 (Winner)", "4.68 / 5.0")
    ]

    for r_idx, r in enumerate(vault_rows):
        for c_idx, val in enumerate(r):
            cell = table.cell(r_idx, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(18, 25, 38) if r_idx == 0 else CARD_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(9.5)
            if r_idx == 0:
                p.font.bold = True
                p.font.color.rgb = GOLD
            elif "Winner" in val or "faster" in val:
                p.font.bold = True
                p.font.color.rgb = CYAN if c_idx == 1 else EMERALD
            elif c_idx == 1:
                p.font.color.rgb = CYAN
            elif c_idx == 2:
                p.font.color.rgb = EMERALD
            else:
                p.font.bold = True
                p.font.color.rgb = TEXT_WHITE

    add_banner(slide15, "Thank you Dr. Li & Classmates! The live Gradio demonstration app is running and ready for questions.", icon="🎓")

    set_notes(slide15, """This final slide is your quick memory vault. Every single number from my capstone project is summarized here:
- BART: 140 million parameters, fp16, 6 layers, 11m 30s training, 0.309s latency, 40.23% R-1, 37.01% R-Lsum, 4.88 factuality.
- BART 3-Epoch (Colab): 40.86% R-1, 18.44% R-2, 37.95% R-Lsum, 0.296s latency.
- T5: 220 million parameters (+57%), fp32, 12 layers, 21m 52s training, 0.569s latency, 42.88% R-1, 36.67% R-Lsum, 4.68 factuality.
- Dataset: CNN/DailyMail 3.0.0, 12k train / 1k val / 1k test, Seed 42, 512 input tokens, 128 max output tokens, batch size 32, beam size 4, 3-gram blocking.
Thank you Dr. Li and classmates! I would be delighted to take any questions or run custom live news stories through my Gradio demo.""")

    for path in output_paths:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        prs.save(path)
        print(f"Saved PowerPoint presentation to: {path}")

if __name__ == "__main__":
    outputs = [
        "report/comparative_summarization_defense.pptx",
        "presentation.pptx"
    ]
    create_deck(outputs)
