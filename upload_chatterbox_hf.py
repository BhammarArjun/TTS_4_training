#!/usr/bin/env python3
"""
upload_chatterbox_hf.py
Uploads fine-tuned Chatterbox Gujarati model to HuggingFace.
Saves: t3_finetuned.safetensors, tokenizer.json, config, and a model card.
"""

import os
import json
import argparse
from huggingface_hub import HfApi, create_repo

HF_REPO = "Arjun4707/chatterbox-gujarati"
MODEL_DIR = "chatterbox_output"
PRETRAINED_DIR = "pretrained_models"

FILES_TO_UPLOAD = {
    # Fine-tuned T3 weights
    "t3_finetuned.safetensors": os.path.join(MODEL_DIR, "t3_finetuned.safetensors"),
    # Extended tokenizer with Gujarati characters
    "tokenizer.json": os.path.join(PRETRAINED_DIR, "tokenizer.json"),
    # Conditioning file
    "conds.pt": os.path.join(PRETRAINED_DIR, "conds.pt"),
}

# These are base model files needed for inference but not fine-tuned
# They're large — upload only if you want a self-contained repo
BASE_MODEL_FILES = {
    "ve.safetensors": os.path.join(PRETRAINED_DIR, "ve.safetensors"),
    "s3gen.safetensors": os.path.join(PRETRAINED_DIR, "s3gen.safetensors"),
}

README_TEMPLATE = """---
language:
  - gu
tags:
  - tts
  - text-to-speech
  - gujarati
  - chatterbox
  - voice-cloning
license: mit
base_model: ResembleAI/chatterbox
---

# Chatterbox Gujarati TTS

Fine-tuned [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) for Gujarati text-to-speech.

## Model Details

- **Base model:** ResembleAI/chatterbox (0.5B LLaMA-based)
- **Fine-tuned component:** T3 transformer only (VE and S3Gen frozen)
- **Vocab size:** {vocab_size} (extended from 704 with Gujarati graphemes)
- **Training data:** ~{num_clips} clips from Arjun4707/clean-gu-hi-tts
- **Training:** {epochs} epochs on L4 GPU

## Files

| File | Description | Required |
|------|-------------|----------|
| `t3_finetuned.safetensors` | Fine-tuned T3 weights | Yes |
| `tokenizer.json` | Extended tokenizer with Gujarati chars | Yes |
| `conds.pt` | Conditioning embeddings | Yes |
| `ve.safetensors` | Voice encoder (frozen, from base) | Yes (or download from ResembleAI/chatterbox) |
| `s3gen.safetensors` | Speech decoder (frozen, from base) | Yes (or download from ResembleAI/chatterbox) |

## Usage

```python
import torch
import torchaudio as ta
from safetensors.torch import load_file
from chatterbox.tts import ChatterboxTTS

# Load base model
model = ChatterboxTTS.from_pretrained(device="cuda")

# Resize embeddings to match fine-tuned vocab
new_emb = torch.nn.Embedding({vocab_size}, 1024).to("cuda")
new_head = torch.nn.Linear(1024, {vocab_size}, bias=False).to("cuda")
model.t3.text_emb = new_emb
model.t3.text_head = new_head

# Load fine-tuned T3
from huggingface_hub import hf_hub_download
t3_path = hf_hub_download("{repo_id}", "t3_finetuned.safetensors")
tok_path = hf_hub_download("{repo_id}", "tokenizer.json")

state_dict = load_file(t3_path)
model.t3.load_state_dict(state_dict, strict=False)

# Load extended tokenizer
model.t3.tokenizer.load(tok_path)

# Generate Gujarati speech with voice cloning
wav = model.generate(
    "ગુજરાતી ભાષામાં આ એક પરીક્ષણ છે.",
    audio_prompt_path="reference.wav",
    exaggeration=0.5,
    cfg_weight=0.5
)
ta.save("output.wav", wav, model.sr)
```

## Preprocessing Applied to Training Data

- CPS filter: 4-25 characters/second
- Speaker diarization: pyannote-audio (single-speaker only)
- Text cleaning: stripped `>` and `|`
- Duration filter: 2-20 seconds
"""


def upload_model(
    repo_id=HF_REPO,
    include_base_models=False,
    hf_token=None,
    vocab_size=2514,
    num_clips=33851,
    epochs=10,
):
    api = HfApi(token=hf_token)

    # Create repo
    try:
        create_repo(repo_id, repo_type="model", private=True, token=hf_token, exist_ok=True)
        print(f"Repo {repo_id} ready.")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload fine-tuned files
    for filename, local_path in FILES_TO_UPLOAD.items():
        if not os.path.exists(local_path):
            print(f"  SKIP (not found): {local_path}")
            continue
        print(f"  Uploading {filename} ({os.path.getsize(local_path) / 1e6:.0f} MB)...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"  Done: {filename}")

    # Optionally upload base model files
    if include_base_models:
        for filename, local_path in BASE_MODEL_FILES.items():
            if not os.path.exists(local_path):
                print(f"  SKIP (not found): {local_path}")
                continue
            print(f"  Uploading {filename} ({os.path.getsize(local_path) / 1e6:.0f} MB)...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"  Done: {filename}")

    # Upload README
    readme = README_TEMPLATE.format(
        vocab_size=vocab_size,
        num_clips=num_clips,
        epochs=epochs,
        repo_id=repo_id,
    )
    readme_path = "/tmp/README.md"
    with open(readme_path, "w") as f:
        f.write(readme)
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"\nUploaded to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default=HF_REPO)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--include_base_models", action="store_true",
                        help="Also upload ve.safetensors and s3gen.safetensors (~2GB)")
    parser.add_argument("--vocab_size", type=int, default=2514)
    parser.add_argument("--num_clips", type=int, default=33851)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    upload_model(
        repo_id=args.repo_id,
        include_base_models=args.include_base_models,
        hf_token=args.hf_token,
        vocab_size=args.vocab_size,
        num_clips=args.num_clips,
        epochs=args.epochs,
    )