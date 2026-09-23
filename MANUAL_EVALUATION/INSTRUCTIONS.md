# Manual Subjective Evaluation Guide (Mean Opinion Score - MOS)

This folder contains a 10-sentence comparative listening set designed to measure subjective human quality across all four model variants:
- **BASE**: `bodhan-ai/indic-speak` foundation model (unadapted baseline)
- **R1** (`M1.wav`): Run 1 (1,200 samples, Attention-only LoRA)
- **R2** (`M2.wav`): Run 2 (3,500 samples, Attention + SwiGLU MLP LoRA)
- **R3** (`M3.wav`): Run 3 (6,829 samples full corpus, 2 epochs)

---

## 1. Directory Layout

Each of the 10 folders contains the exact same prompt synthesized by each of the four models:

```
MANUAL_EVALUATION/
├── manifest.csv           # Fill your 1–5 scores here
├── INSTRUCTIONS.md        # This guide
├── sentence_01/           # "कदाचित आपण दोन्ही करू शकतो!" (Short, conversational)
│   ├── BASE.wav
│   ├── M1.wav
│   ├── M2.wav
│   └── M3.wav
├── sentence_02/           # "ठीक आहे मग, मला मग नेहमीची अंडीच दे." (Colloquial imperative)
│   ├── BASE.wav, M1.wav, M2.wav, M3.wav
...
└── sentence_10/           # "हो, मी check out करतोय." (Code-mixed English borrowing)
    ├── BASE.wav, M1.wav, M2.wav, M3.wav
```

---

## 2. Scoring Rubrics (1–5 Scale)

Please evaluate each clip on three separate dimensions using a **1 to 5 integer scale**:

### A. Naturalness (Human-likeness & Prosody)
*How natural and expressive does the voice sound?*
- **5 = Very Natural**: Sounds like a native Marathi human speaker; smooth rhythm, natural pitch inflections, appropriate pauses.
- **4 = Natural**: Mostly natural; minor synthetic tone or slight stiffness, but pleasant and engaging.
- **3 = Fair**: Understandable human tone, but clearly synthetic cadence or robotic pacing.
- **2 = Poor**: Monotone, unnatural pitch jumps, or noticeably robotic delivery.
- **1 = Very Robotic**: Severely mechanical, flat, or unnatural voice.

### B. Clarity / Intelligibility (Phonetic Accuracy)
*How easy is it to understand every Marathi word and conjunct?*
- **5 = Completely Clear**: Every vowel, consonant, and conjunct (*जोडाक्षर*) is crisp and effortless to comprehend.
- **4 = Clear**: Easy to understand; minor softening of difficult consonants, but meaning is immediately apparent.
- **3 = Fair**: Understandable with moderate effort; some compound words sound slightly slurred.
- **2 = Difficult**: Multiple words are garbled, indistinct, or mispronounced.
- **1 = Incomprehensible**: Cannot understand what is being said.

### C. Noise & Artifacts (Acoustic Quality)
*Is the audio clean and free of background noise, buzzing, or clicks?*
- **5 = None**: Clean, studio-grade audio; no audible distortion or background noise.
- **4 = Slight**: Minor vocoder hiss or faint phase buzz in high frequencies, but non-distracting.
- **3 = Moderate**: Noticeable vocoder buzz, metallic coloration, or slight muffling.
- **2 = Heavy**: Distracting robotic buzz, clicks, phase warping, or gating flutter.
- **1 = Severe**: Harsh static, loud digital clipping, or severe distortion.

---

## 3. Recommended Listening Procedure

1. **Use Headphones**: Listen with over-ear or in-ear headphones rather than laptop speakers to catch subtle phonetic details.
2. **Fixed Volume**: Keep your playback volume at a comfortable, consistent level across all files.
3. **Listen in Batches**: For each sentence (e.g. `sentence_01`), listen to `BASE.wav`, `M1.wav`, `M2.wav`, and `M3.wav` before writing your scores in `manifest.csv`.
4. **Re-listen**: Feel free to replay any clip a second time if two models sound close.

---

## 4. How to Calculate Final MOS

Once you finish filling `manifest.csv`, compute the mean for each column:

$$\text{MOS}_{\text{Model}} = \frac{1}{10} \sum_{i=1}^{10} \text{Score}_i$$

You can report:
- **Base MOS** vs **R1 MOS** vs **R2 MOS** vs **R3 MOS** for both Naturalness and Clarity in your submission report.
