"""
Generate the standalone Kaggle evaluation notebook (Evaluation.ipynb).
Creates a self-contained notebook for acoustic analysis, Indic ASR evaluation, and audio playback.
"""

import json
from pathlib import Path

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Marathi TTS Comprehensive Model Evaluation Suite\n",
            "\n",
            "This notebook runs the complete evaluation and comparison pipeline across all models:\n",
            "- **Base Model**: `bodhan-ai/indic-speak` (3.3B parameter multilingual baseline)\n",
            "- **Model 1**: 1,200 samples, Attention-only LoRA (`q, k, v, o`), lr=1e-4\n",
            "- **Model 2**: 3,500 samples (100% Anagha), Attention + SwiGLU MLP LoRA, lr=3e-5\n",
            "- **Model 3**: 6,829 samples (100% Anagha Full Corpus), Attention + SwiGLU MLP LoRA, 2 epochs\n",
            "\n",
            "### Evaluation Pipeline Overview:\n",
            "1. **Objective Acoustic Profiling**: Duration, Peak Amplitude, and RMS Power\n",
            "2. **Indic ASR Benchmarking**: Automated CER/WER computation using Marathi Wav2Vec2\n",
            "3. **Visual Comparison**: Comparative waveform envelopes and Mel-scale spectrograms\n",
            "4. **Interactive Audio Suite**: Side-by-side listening players directly in Kaggle"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Clone repository and install dependencies\n",
            "import os\n",
            "repo_dir = \"/kaggle/working/indic-speak-marathi-finetune\"\n",
            "if not os.path.exists(repo_dir):\n",
            "    !git clone https://github.com/mehersoni/indic-speak-marathi-finetune.git {repo_dir}\n",
            "%cd {repo_dir}\n",
            "!git pull\n",
            "\n",
            "!pip install -q soundfile transformers scipy matplotlib jiwer tabulate"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Objective Acoustic Metrics Analysis (Duration, Peak, RMS Power)\n",
            "from pathlib import Path\n",
            "import soundfile as sf\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "\n",
            "models = {\n",
            "    \"Base Model\": \"audio/model_2/base\",\n",
            "    \"Model 1 (1.2k)\": \"audio/model_1/finetune_normalised\",\n",
            "    \"Model 2 (3.5k)\": \"audio/model_2/finetune_normalised\",\n",
            "    \"Model 3 (6.8k Full)\": \"audio/model_3/finetune_normalised\"\n",
            "}\n",
            "\n",
            "sentences = [\n",
            "    (\"00\", \"नमस्कार, आज आपण विज्ञान विषयाचा अभ्यास करणार आहोत.\"),\n",
            "    (\"01\", \"मॅडम, काही मदत हवी आहे का?\"),\n",
            "    (\"02\", \"महाराष्ट्र हे भारतातील एक पुरोगामी आणि महत्त्वाचे राज्य आहे.\"),\n",
            "    (\"03\", \"शिक्षण हे मानवी जीवनाचा पाया आहे.\"),\n",
            "]\n",
            "\n",
            "records = []\n",
            "for m_label, m_dir in models.items():\n",
            "    p = Path(m_dir)\n",
            "    for idx, text in sentences:\n",
            "        # Locate file\n",
            "        matches = list(p.glob(f\"*{idx}*.wav\"))\n",
            "        if matches:\n",
            "            wav_file = matches[0]\n",
            "            data, sr = sf.read(str(wav_file))\n",
            "            dur = len(data) / sr\n",
            "            peak = np.max(np.abs(data))\n",
            "            rms = np.sqrt(np.mean(data**2))\n",
            "            dbfs = 20 * np.log10(rms + 1e-9)\n",
            "            records.append({\n",
            "                \"Model\": m_label,\n",
            "                \"Sentence\": idx,\n",
            "                \"Duration (s)\": round(dur, 2),\n",
            "                \"Peak\": round(peak, 3),\n",
            "                \"RMS Power\": round(rms, 4),\n",
            "                \"Level (dBFS)\": round(dbfs, 1)\n",
            "            })\n",
            "\n",
            "df_metrics = pd.DataFrame(records)\n",
            "print(\"=== Summary Acoustic Metrics ===\")\n",
            "display(df_metrics.pivot(index=\"Sentence\", columns=\"Model\", values=[\"Duration (s)\", \"RMS Power\"]))"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Run Automated Indic ASR Evaluation (Devanagari Transcriptions & CER/WER)\n",
            "import re\n",
            "import torch\n",
            "import scipy.signal\n",
            "from transformers import AutoProcessor, AutoModelForCTC\n",
            "\n",
            "DEFAULT_GROUND_TRUTHS = dict(sentences)\n",
            "\n",
            "def lev_distance(s1, s2):\n",
            "    m, n = len(s1), len(s2)\n",
            "    dp = [[0]*(n+1) for _ in range(m+1)]\n",
            "    for i in range(m+1): dp[i][0] = i\n",
            "    for j in range(n+1): dp[0][j] = j\n",
            "    for i in range(1, m+1):\n",
            "        for j in range(1, n+1):\n",
            "            if s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1]\n",
            "            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i-1][j-1], dp[i-1][j-1])\n",
            "    return dp[m][n]\n",
            "\n",
            "# Load Marathi ASR model on GPU\n",
            "asr_model_id = \"sumedh/wav2vec2-large-xlsr-marathi\"\n",
            "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
            "print(f\"Loading Marathi ASR model '{asr_model_id}' on {device}...\")\n",
            "processor = AutoProcessor.from_pretrained(asr_model_id)\n",
            "asr_model = AutoModelForCTC.from_pretrained(asr_model_id).to(device)\n",
            "\n",
            "asr_results = []\n",
            "for m_label, m_dir in [(\"Model 2 (Production)\", \"audio/model_2/finetune_normalised\"), (\"Model 3 (Full Corpus)\", \"audio/model_3/finetune_normalised\")]:\n",
            "    p = Path(m_dir)\n",
            "    for idx, ref_text in sentences:\n",
            "        matches = list(p.glob(f\"*{idx}*.wav\"))\n",
            "        if not matches:\n",
            "            continue\n",
            "        wav, sr = sf.read(str(matches[0]))\n",
            "        if sr != 16000:\n",
            "            wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))\n",
            "        inputs = processor(wav, sampling_rate=16000, return_tensors=\"pt\").to(device)\n",
            "        with torch.no_grad():\n",
            "            logits = asr_model(**inputs).logits\n",
            "            predicted_ids = torch.argmax(logits, dim=-1)\n",
            "        hyp = processor.batch_decode(predicted_ids)[0].strip()\n",
            "        \n",
            "        ref_c = list(re.sub(r\"[^\\w]\", \"\", ref_text))\n",
            "        hyp_c = list(re.sub(r\"[^\\w]\", \"\", hyp))\n",
            "        cer = lev_distance(ref_c, hyp_c) / max(1, len(ref_c))\n",
            "        \n",
            "        ref_w = re.sub(r\"[^\\w\\s]\", \"\", ref_text).split()\n",
            "        hyp_w = re.sub(r\"[^\\w\\s]\", \"\", hyp).split()\n",
            "        wer = lev_distance(ref_w, hyp_w) / max(1, len(ref_w))\n",
            "        \n",
            "        asr_results.append({\n",
            "            \"Model\": m_label,\n",
            "            \"Sentence\": idx,\n",
            "            \"Reference\": ref_text,\n",
            "            \"ASR Hypothesis\": hyp,\n",
            "            \"CER (%)\": f\"{cer*100:.1f}%\",\n",
            "            \"WER (%)\": f\"{wer*100:.1f}%\"\n",
            "        })\n",
            "\n",
            "df_asr = pd.DataFrame(asr_results)\n",
            "display(df_asr)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Generate Comparative Visualizations (Waveforms & Spectrograms)\n",
            "!python scripts/plot_all_models_waveforms.py\n",
            "!python scripts/plot_base_vs_finetuned.py\n",
            "\n",
            "from IPython.display import Image, display\n",
            "print(\"=== 3-Model Waveform Comparison Grid ===\")\n",
            "display(Image(\"figures/waveform_comparison_all_models.png\"))\n",
            "\n",
            "print(\"=== Duration & RMS Energy Comparison ===\")\n",
            "display(Image(\"figures/base_vs_finetuned_comparison.png\"))"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Interactive Side-by-Side Listening Suite\n",
            "from IPython.display import Audio, display, HTML\n",
            "\n",
            "display(HTML(\"<h3>🎧 Interactive Audio Listening Comparison</h3>\"))\n",
            "for idx, text in sentences:\n",
            "    display(HTML(f\"<h4>Sentence {idx}: <i>{text}</i></h4>\"))\n",
            "    \n",
            "    # Base\n",
            "    f_base = Path(f\"audio/model_2/base/base_{idx}.wav\")\n",
            "    if f_base.exists():\n",
            "        print(\"Base Model (Untuned):\")\n",
            "        display(Audio(str(f_base)))\n",
            "        \n",
            "    # Model 1\n",
            "    f_m1 = list(Path(\"audio/model_1/finetune_normalised\").glob(f\"*{idx}*.wav\"))\n",
            "    if f_m1:\n",
            "        print(\"Model 1 (1.2k Samples, Attn-Only):\")\n",
            "        display(Audio(str(f_m1[0])))\n",
            "        \n",
            "    # Model 2\n",
            "    f_m2 = list(Path(\"audio/model_2/finetune_normalised\").glob(f\"*{idx}*.wav\"))\n",
            "    if f_m2:\n",
            "        print(\"Model 2 (3.5k Samples, Attn+MLP):\")\n",
            "        display(Audio(str(f_m2[0])))\n",
            "        \n",
            "    # Model 3\n",
            "    f_m3 = list(Path(\"audio/model_3/finetune_normalised\").glob(f\"*{idx}*.wav\"))\n",
            "    if f_m3:\n",
            "        print(\"Model 3 (6.8k Full Corpus):\")\n",
            "        display(Audio(str(f_m3[0])))\n",
            "    print(\"-\" * 80)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 6. Package and export evaluation artifacts\n",
            "!zip -r evaluation_deliverables.zip figures/ audio/ evaluation_results.csv\n",
            "print(\"Evaluation artifacts packaged into evaluation_deliverables.zip\")"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [],
            "isGpuEnabled": True,
            "isInternetEnabled": True,
            "language": "python"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_file = Path("Evaluation.ipynb")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Successfully generated {out_file}")
