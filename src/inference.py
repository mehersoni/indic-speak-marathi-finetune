"""
Inference script for generating Marathi speech with base or LoRA fine-tuned models.
Converts Marathi text to audio tokens and generates 24 kHz WAV files into outputs/examples/.
"""

import argparse
import os
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from snac import SNAC

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tokenize_format import audio_tokens_to_codes, build_prompt_tokens
from src.utils import END_OF_SPEECH_ID, SAMPLE_RATE

DEFAULT_MARATHI_SENTENCES = [
    "नमस्कार, आज आपण विज्ञान विषयाचा अभ्यास करणार आहोत.",
    "मॅडम, काही मदत हवी आहे का?",
    "महाराष्ट्र हे भारतातील एक पुरोगामी आणि महत्त्वाचे राज्य आहे.",
    "शिक्षण हे मानवी जीवनाचा पाया आहे.",
]


def load_vocos_decoder(model_path: str, device: str = "cpu"):
    """Loads fine-tuned Vocos decoder from local directory or Hugging Face repository."""
    local_pt = Path(model_path) / "vocos" / "best.pt"
    if local_pt.is_file():
        ckpt_path = str(local_pt)
    else:
        from huggingface_hub import hf_hub_download
        repo_id = "bodhan-ai/indic-speak" if os.path.isdir(model_path) else model_path
        ckpt_path = hf_hub_download(repo_id=repo_id, filename="vocos/best.pt", token=os.environ.get("HF_TOKEN"))

    from src.vocos.load import load_vocos
    return load_vocos(ckpt_path, device=device)


def generate_speech(
    model,
    tokenizer,
    snac_model,
    vocos_decoder,
    text: str,
    speaker: str = "Anagha",
    device: str = "cpu",
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 2520,
) -> tuple[np.ndarray, int, int]:
    """Generates 24 kHz mono speech waveform for a Marathi text string. Returns (waveform, num_audio_tokens, num_prompt_tokens)."""
    prompt_ids = build_prompt_tokens(tokenizer, text=text, speaker=speaker)
    input_ids = torch.tensor([prompt_ids], device=device)

    # Heuristic cap: ~30 audio tokens per character is a generous upper bound for Marathi.
    # Prevents runaway generation on short sentences without cutting off long ones.
    adaptive_max = min(max_new_tokens, max(280, len(text) * 30))

    with torch.no_grad():
        gen_tokens = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=adaptive_max,
            eos_token_id=[END_OF_SPEECH_ID, tokenizer.eos_token_id],
            pad_token_id=tokenizer.eos_token_id,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        new_ids = gen_tokens[0].tolist()[len(prompt_ids) :]
        num_audio_tokens = len(new_ids)
        codes = audio_tokens_to_codes(new_ids, device=device)
        z_q = snac_model.quantizer.from_codes(codes)

        if vocos_decoder is not None:
            wav = vocos_decoder(z_q.float())
        else:
            wav = snac_model.decoder(z_q)

        waveform = wav[0, 0].clamp(-1, 1).float().cpu().numpy()
        return waveform, num_audio_tokens, len(prompt_ids)


def run_inference(
    base_model_path: str = "bodhan-ai/indic-speak",
    adapter_path: str | None = None,
    sentences: list[str] | None = None,
    speaker: str = "Anagha",
    output_dir: str = "outputs/examples",
    temperature: float = 0.6,
    seed: int = 42,
):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.float16 if device == "cuda" else torch.float32

    # Check local repository snapshot if available
    local_repo = os.path.join(os.path.expanduser("~"), "indic-speak-repo")
    model_id = local_repo if os.path.isdir(local_repo) else base_model_path
    hf_token = os.environ.get("HF_TOKEN")

    print(f"Loading base model and tokenizer from: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=model_dtype,
        token=hf_token,
        attn_implementation="sdpa",
    ).to(device)

    if adapter_path is not None:
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"Specified adapter path '{adapter_path}' does not exist. "
                "Please run training ('python src/train.py --config configs/marathi_lora.yaml') first to generate the adapter."
            )
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path).to(device)
        name_prefix = "finetuned"
    else:
        name_prefix = "base"

    model.eval()

    print("Loading SNAC 24kHz codec and Vocos decoder...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
    try:
        vocos_decoder = load_vocos_decoder(model_id, device=device)
    except Exception as e:
        print(f"Vocos decoder not available ({e}), falling back to stock SNAC decoder.")
        vocos_decoder = None

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    test_sentences = sentences or DEFAULT_MARATHI_SENTENCES
    print(f"\nGenerating audio for {len(test_sentences)} sentences (Prefix: {name_prefix}_NN.wav)...")

    for idx, sentence in enumerate(test_sentences):
        wav, num_audio_tokens, num_prompt_tokens = generate_speech(
            model=model,
            tokenizer=tokenizer,
            snac_model=snac_model,
            vocos_decoder=vocos_decoder,
            text=sentence,
            speaker=speaker,
            device=device,
            temperature=temperature,
        )
        wav_filename = f"{name_prefix}_{idx:02d}.wav"
        save_file = out_path / wav_filename
        sf.write(str(save_file), wav, SAMPLE_RATE)
        duration_sec = len(wav) / SAMPLE_RATE
        print(f"  [{name_prefix}_{idx:02d}] {duration_sec:.2f}s | {num_audio_tokens} audio tokens (prompt: {num_prompt_tokens} tokens) -> {save_file}")
        print(f"       Text: \"{sentence}\" ({len(sentence)} chars, {len(sentence.split())} words)\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Marathi speech with base or LoRA model")
    parser.add_argument("--model", default="bodhan-ai/indic-speak", help="Base model identifier or path")
    parser.add_argument("--adapter", default=None, help="Path to trained LoRA adapter checkpoint")
    parser.add_argument("--speaker", default="Anagha", help="Speaker name (e.g. Anagha, Chinmay)")
    parser.add_argument("--text", action="append", default=None, help="Marathi sentence(s) to synthesize")
    parser.add_argument("--out-dir", default="outputs/examples", help="Output directory for generated WAVs")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_inference(
        base_model_path=args.model,
        adapter_path=args.adapter,
        sentences=args.text,
        speaker=args.speaker,
        output_dir=args.out_dir,
        temperature=args.temperature,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
