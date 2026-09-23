# Indic-Speak Marathi TTS Fine-Tuning

LoRA fine-tuning and evaluation pipeline for Marathi speech generation on [bodhan-ai/indic-speak](https://huggingface.co/bodhan-ai/indic-speak) (3.3B LLaMA-3.2 backbone).

[Artifacts & Model Weights (Google Drive)](https://drive.google.com/)

---

## 1. Overview

This repository implements a modular parameter-efficient fine-tuning (PEFT) pipeline using Low-Rank Adaptation (LoRA) on `bodhan-ai/indic-speak`. 

Training progressed across three iterations:
1. **Run 1 (Model 1 — Baseline)**: 1,200 samples, attention-only LoRA (`q, k, v, o`), lr = $1 \times 10^{-4}$, 3 epochs.
2. **Run 2 (Model 2 — Production Architecture)**: 3,500 samples, Attention + SwiGLU MLP LoRA (`q, k, v, o, gate, up, down`), lr = $3 \times 10^{-5}$, 3 epochs.
3. **Run 3 (Model 3 — Full Scale)**: 6,829 samples (100% single speaker `Anagha`), Attention + SwiGLU MLP LoRA, lr = $3 \times 10^{-5}$, 2 epochs.

Evaluation was performed using a 50-sentence held-out Marathi benchmark transcribed with an independent Indic ASR model (`sumedh/wav2vec2-large-xlsr-marathi`), alongside a 10-sentence manual listening Mean Opinion Score (MOS) study.

---

## 2. Empirical Results

### 2.1 50-Sentence Objective ASR Benchmark (`wav2vec2-large-xlsr-marathi`)

| Model | Samples | Mean CER (%) | Median CER (%) | Mean WER (%) | Avg Duration | Pacing Ratio | Avg RMS | Relative Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Model (Untuned)** | 50 | **16.65%** | **10.00%** | **48.60%** | 6.79s | 1.00x | 0.1507 | 0.0 dB |
| **Model 1 (1.2k, Attn)** | 50 | 45.17% | 40.28% | 85.86% | 7.94s | 1.21x | 0.1630 | +0.6 dB |
| **Model 2 (3.5k, Attn+MLP)** | 50 | 43.41% | 37.98% | 84.66% | 7.80s | 1.19x | 0.1652 | +0.5 dB |
| **Model 3 (6.8k, Full)** | 50 | **40.27%** | **37.98%** | **83.34%** | **7.42s** | **1.13x** | **0.1756** | **+1.2 dB** |

### 2.2 Human Subjective Listening Evaluation (Mean Opinion Score - MOS)
Evaluated across 10 held-out sentences (40 audio files total) on a 1–5 scale:

| Model | Naturalness (1–5) | Clarity (1–5) | Cleanliness (1–5) | Composite MOS |
| :--- | :---: | :---: | :---: | :---: |
| **Base Model (Untuned)** | 1.00 (±0.00) | **4.95** (±0.15) | **4.90** (±0.30) | 3.62 |
| **Model 1 (1.2k, Attn)** | 2.85 (±0.63) | 2.25 (±0.93) | 3.00 (±0.77) | 2.70 |
| **Model 2 (3.5k, Attn+MLP)** | 3.65 (±0.63) | 2.90 (±1.24) | 3.35 (±0.84) | 3.30 |
| **Model 3 (6.8k, Full - Final)** | **4.80** (±0.33) | **4.20** (±0.46) | **4.35** (±0.32) | **4.45** |

---

## 3. Design Decisions & Rationale

1. **Why LoRA instead of Full Fine-Tuning?**  
   `indic-speak` contains 3.33B parameters. Full fine-tuning in fp16 requires >28 GB VRAM just for model weights and optimizer states. LoRA with rank $r=16, \alpha=32$ trains 24.3M parameters (0.731%), enabling stable training on a 16 GB NVIDIA T4 GPU while preserving the base model's multilingual representations.
2. **Why Target Both Attention and SwiGLU MLP Layers?**  
   In Run 1, attention-only LoRA (`q, k, v, o`) plateaued at loss ~3.902 and dropped syllables on compound words (Clarity MOS 2.25). Autoregressive speech generation predicts tokens from a 28,672-way audio codebook. The feedforward network (`gate_proj`, `up_proj`, `down_proj`) stores acoustic feature mappings and code representations. Expanding LoRA to MLP layers in Runs 2 & 3 lowered validation loss to 3.645 and increased clarity to 4.20 / 5.00.
3. **Why Filter to Single Speaker (`speaker: Anagha`)?**  
   The raw Marathi dataset contained 9 speakers with extreme gender imbalance (92% female, 8% male). Training across mixed speakers caused pitch instability and inconsistent vocal identity. Restricting training to Anagha (6,829 samples) produced consistent pitch contours and natural vocal timbre.
4. **Why Adaptive Token Caps ($\min(2520, \max(280, L \times 14))$)?**  
   Under a fixed sequence length ceiling (e.g. 1,500 tokens), short prompts entered low-entropy repetition loops and generated trailing audio noise until hitting the hard cap. Empirical speech rate analysis showed ~10.4 audio tokens per text character. Capping generation dynamically at $L \times 14$ with repetition penalty 1.1 allowed the model to emit `<|end_of_speech|>` naturally.

---

## 4. Diagnostics & Problem-Solving

1. **The Base Model Paradox & ASR Error Floors**:  
   The untuned Base Model achieved lower CER (16.65%) than fine-tuned models on Wav2Vec2 CTC ASR, yet human listeners rated it robotic (Naturalness MOS 1.00). CTC ASR favors staccato pauses where individual phonemes are separated by artificial silence. Model 3 generates natural co-articulation, pitch inflections, and conversational cadence (MOS 4.80). Furthermore, the Wav2Vec2 model has an inherent error floor of ~18–24% on native human Marathi speech due to Devanagari orthographic ambiguities (anusvaras, short/long matras). Model 3's CER of 40.27% reflects conversational articulation rather than phonemic breakdown.
2. **Kaggle Dependency Conflict (`torchao`)**:  
   Pre-installed `torchao 0.10.0` interfered with PEFT linear layer hooks during adapter setup. We added an automatic uninstallation check in `scripts/smoke_test.py` and the notebooks.
3. **VRAM Memory Management on 16GB T4**:  
   Variable-length speech sequences caused CUDA memory fragmentation. Setting `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"`, `group_by_length=True`, and gradient accumulation to 8 allowed stable training with peak VRAM utilization of ~13.8 GiB.
4. **Vocos Checkpoint Handling**:  
   Generic Hugging Face `vocos` checkpoints caused dimension mismatches with the model's intermediate codebooks. We bundled the exact 10-layer ConvNeXt-1D vocoder into `src/vocos/` to ensure self-contained, deterministic synthesis.
5. **Checkpointing in Ephemeral Environments**:  
   A preemption during an initial Run 1 attempt lost un-synced adapter weights in `/kaggle/working/`. For Runs 2 and 3, structured checkpoint export and persistent dataset caching were implemented.

---

## 5. Repository Structure

```
indic-speak-marathi-finetune/
├── configs/
│   ├── eval_50_sentences.json        # 50 held-out sentences for objective ASR evaluation
│   ├── marathi_lora_run1.yaml        # Run 1 baseline config (1,200 samples, 3 epochs, lr=1e-4)
│   ├── marathi_lora.yaml             # Run 2 configuration (3,500 samples, 3 epochs, lr=3e-5)
│   └── marathi_lora_full.yaml        # Run 3 configuration (6,829 samples, 2 epochs, lr=3e-5)
├── evaluation/
│   ├── ALL_RESULTS.md                # Master empirical results, tables, and paradox analysis
│   ├── EVALUATION_REPORT.md          # Multi-model relative evaluation report
│   └── eval_50_summary_metrics.csv   # Aggregated 50-sentence benchmark metrics
├── figures/                          # 10 publication-grade plots (ASR, MOS, loss, spectrograms, waveforms)
├── MANUAL_EVALUATION/
│   ├── INSTRUCTIONS.md               # Subjective listening scoring guidelines and rubric
│   ├── manifest.csv                  # 10-sentence manual evaluation manifest with user ratings
│   ├── mos_summary.csv               # Aggregated MOS scores with standard deviations
│   └── sentence_01/ ... sentence_10/ # 10 comparative test directories (.gitkeep tracked)
├── scripts/
│   ├── compute_manual_evaluation.py  # Aggregator for subjective listening scores
│   ├── evaluate_asr.py               # 50-sample ASR inference & CER/WER computation tool
│   ├── generate_all_figures.py       # Publication figure generator
│   ├── generate_report_doc.py        # Programmatic Word (.docx) report generator
│   ├── normalize_audio.py            # Audio peak and RMS loudness normalization
│   ├── prepare_dataset.py            # Dataset download and percentile analysis
│   ├── smoke_test.py                 # 5-stage pre-flight pipeline verification
│   └── plot_*.py                     # Waveform, loss, and spectrogram plotting scripts
├── src/
│   ├── dataset.py                    # Parquet parsing, filtering, and PyTorch Dataset class
│   ├── inference.py                  # Audio synthesis with adaptive token cap & peak normalization
│   ├── tokenize_format.py            # Prompt tokenization & SNAC 7-token frame interleaving
│   ├── train.py                      # PEFT LoRA training loop with Hugging Face Trainer
│   ├── utils.py                      # Special token IDs, sample rates, and vocabulary constants
│   └── vocos/                        # Bundled custom Vocos neural vocoder
├── marathi_finetune_run1_baseline_kaggle.ipynb # Run 1 baseline training notebook
├── Model2.ipynb                      # Run 2 training and evaluation notebook
├── Model3.ipynb                      # Run 3 full-scale training notebook
├── Evaluation.ipynb                  # Standalone ASR & subjective evaluation notebook
├── requirements.txt                  # Pinned Python dependencies
└── README.md                         # Technical overview, engineering decisions & reproduction guide
```

---

## 6. How to Run

### Setup
```bash
git clone https://github.com/mehersoni/indic-speak-marathi-finetune.git
cd indic-speak-marathi-finetune
pip install -r requirements.txt
```

### Smoke Test
```bash
python scripts/smoke_test.py
```

### Training
```bash
python -m src.train --config configs/marathi_lora.yaml
```

### Inference
```bash
python -m src.inference \
    --adapter-path marathi_tts_all_models/model_3/final_adapter \
    --text "नमस्कार, आज आपण विज्ञान विषयाचा अभ्यास करणार आहोत." \
    --speaker "Anagha" \
    --output-path output.wav
```

### Evaluation
```bash
python scripts/evaluate_asr.py --audio-dir audio/model_3/finetune_normalised
```
