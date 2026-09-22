<!--
Project documentation for indic-speak-marathi.
Details model architecture, dataset preparation, LoRA configuration, challenges, and verification status.
-->

# indic-speak-marathi

LoRA fine-tuning pipeline for Marathi speech generation on `bodhan-ai/indic-speak`.

---

## 1. Model Architecture

- **Base Model**: `bodhan-ai/indic-speak`
- **Underlying Architecture**: `LlamaForCausalLM` (Llama-3.2-3B base, 28 hidden layers, hidden dimension 3072, 24 attention heads, 8 KV heads)
- **Total Parameters**: `3,300,928,512` (3.30B parameters)
- **Vocabulary Size**: `156,960`
  - Text BPE range: `0 – 127,999` (128,000 Llama-3 BPE tokens)
  - Stock Llama-3 specials: `128,000 – 128,255`
  - Project control tokens: `128,256 – 128,265` (`<|start_of_speech|>` = 128257, `<|end_of_speech|>` = 128258, `<|start_of_human|>` = 128259, `<|start_of_ai|>` = 128261)
  - Audio tokens (SNAC): `128,266 – 156,937` (28,672 tokens across 7 codebooks × 4,096 codes)
  - Conditioning delimiters: `156,938 – 156,943` (`<|speaker>` = 156938, `<speaker|>` = 156939, `<|style>` = 156940, `<style|>` = 156941)
  - Non-verbal tokens: `156,944 – 156,959` (16 closed-set non-verbals)
- **Audio Codec**: SNAC 24 kHz (3 codebooks at 1:2:4 temporal ratio, flattened into 7 tokens per frame in fixed interleave)
- **Vocoder**: Fine-tuned Vocos 24 kHz neural decoder (`vocos/best.pt`)

---

## 2. Dataset

- **Dataset Source**: `snorbyte/indic-tts-sample-snac-encoded` (`data_stage_1.parquet`)
- **Total Rows in Full Dataset**: `63,281`
- **Languages**: 9 Indic languages (Tamil, Marathi, Hindi, Bengali, Kannada, Gujarati, Punjabi, Telugu, Malayalam)
- **Raw Marathi Rows**: `7,604`
- **Raw Marathi Speakers**: 9 unique user IDs (`44`, `82`, `66`, `55`, `80`, `148`, `147`, `67`, `43`)
  - User 44: 4,511 utterances
  - User 82: 1,406 utterances
  - User 66: 548 utterances
  - User 55: 422 utterances
  - User 80: 393 utterances
  - User 148: 197 utterances
  - User 147: 81 utterances
  - User 67: 41 utterances
  - User 43: 5 utterances

---

## 3. Data Preparation

- **Sequence Framing**:
  - Prompt: `<|start_of_human|><|begin_of_text|><|speaker>{speaker}<speaker|>\n{utterance}<|eot_id|><|end_of_human|><|start_of_ai|>`
  - Target: `<|start_of_speech|>{audio_tokens}<|end_of_speech|><|end_of_ai|>`
  - Loss Mask: `-100` for all prompt tokens; unmasked targets over speech generation.
- **Length Filtering**:
  - Applied threshold: `1,400` total sequence tokens (empirical p99 cutoff).
  - Rows dropped: `73` (0.96%)
  - Rows retained: `7,531` (99.04%)
- **Filtered Distribution**:
  - Min: 91 tokens
  - Max: 1,392 tokens
  - Mean: 695.9 tokens
  - Median / p50: 728.0 tokens
  - p90: 990.0 tokens
  - p95: 1,070.0 tokens
  - p99: 1,246.7 tokens
- **Train / Validation Split**:
  - Train: `1,200` samples
  - Validation: `100` samples

---

## 4. Fine-Tuning Strategy

- **Method**: Parameter-Efficient Fine-Tuning with Low-Rank Adaptation (LoRA)
- **Target Modules**: Linear attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) across all 28 transformer layers
- **LoRA Hyperparameters**:
  - Rank ($r$): `16`
  - Alpha ($\alpha$): `32`
  - Dropout: `0.05`
  - Trainable parameters: `9,175,040` / `3,310,103,552` (0.2772% of base model weights)
- **Compute Configuration (for 16GB NVIDIA T4 GPU)**:
  - Precision: `fp16: true` (`bf16: false` — T4 lacks native bfloat16 hardware support)
  - Batch size per device: `1`
  - Gradient accumulation steps: `8` (effective batch size: 8)
  - Gradient checkpointing: `true` (minimizes activation memory)
  - Learning rate: `1e-4` (AdamW optimizer, weight decay `0.01`, warmup ratio `0.05`)
  - Epochs: `3`
  - Checkpoint saving: Every 50 steps (`save_total_limit: 3`)

---

## 5. Challenges

1. **Tokenizer & Codebook Layout Mismatch**:
   The `snorbyte` parquet dataset was originally encoded for `snorTTS-Indic-v0`, which uses a different token layout than `bodhan-ai/indic-speak`. The training sequence construction, delimiter tokens (`<|speaker>`, `<speaker|>`, `<|start_of_human|>`, `<|start_of_ai|>`), and 7-token frame interleaving offset (`128,266 + pos * 4096`) had to be reverse-engineered and asserted directly against `indic-speak`'s own repository token contract.
2. **Speaker Mapping & Gender Imbalance**:
   `indic-speak` uses a closed library of recognized voice artist names (`Anagha` for female Marathi, `Chinmay` for male Marathi); unseen speaker IDs degrade into unconditioned voice priors. Consequently, the 9 distinct user IDs in the dataset had to be collapsed into 2 voice labels based on demographic gender. Furthermore, the dataset exhibits an extreme gender imbalance: `6,940` female rows (92.15%) versus `591` male rows (7.85%).

---

## 6. Limitations & Verification Status

### Verified Locally:
- Dataset ingestion, filtering, and deterministic train/val splitting.
- Token ID assertions against the live `bodhan-ai/indic-speak` tokenizer.
- Exact $1:2:4$ codebook ratio roundtrip and length checks across real Marathi rows.
- Dynamic module discovery and LoRA parameter attachment.

### Unverified Pending Kaggle GPU Access:
- **`scripts/smoke_test.py`**: Full forward and backward pass on full 3.8B weights using CUDA fp16 precision.
- **`src/train.py`**: Execution of the actual training loop and loss convergence on GPU hardware.
- **`src/inference.py`**: End-to-end waveform generation from the 3.8B language model and Vocos checkpoint.

---

## 7. Results

- **Training Loss**: `<TODO>`
- **Validation Loss**: `<TODO>`
- **Runtime & Throughput**: `<TODO>`
- **GPU Peak Memory**: `<TODO>`
- **Sample Audio Files**:
  - Base model samples (`outputs/examples/base_NN.wav`): `<TODO>`
  - Fine-tuned samples (`outputs/examples/finetuned_NN.wav`): `<TODO>`
