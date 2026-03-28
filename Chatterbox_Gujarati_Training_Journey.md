# Chatterbox Gujarati TTS — Complete Training Journey

**Project:** Fine-tuning Chatterbox Multilingual for Gujarati TTS  
**Machine:** Lightning.ai L4 GPU Studio (24 GB VRAM, 31 GB RAM, Python 3.11)  
**Dataset:** `Arjun4707/gu-hi-tts` (private HuggingFace, ~42,564 Gujarati rows → ~33,851 after cleaning)  
**Fine-tuning repo:** `gokhaneraslan/chatterbox-finetuning` (Standard mode, LLaMA-based)  
**Date:** March 2026  

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Preprocessing](#2-data-preprocessing)
3. [Environment Setup](#3-environment-setup)
4. [Vocabulary Extension](#4-vocabulary-extension)
5. [Training Configuration & Patches](#5-training-configuration--patches)
6. [Training](#6-training)
7. [Inference](#7-inference)
8. [Upload to HuggingFace](#8-upload-to-huggingface)
9. [Key Lessons & Known Issues](#9-key-lessons--known-issues)
10. [Hyperparameter Reference](#10-hyperparameter-reference)

---

## 1. Architecture Overview

Chatterbox is built on the **CosyVoice 2.0 architecture** by Resemble AI. It has three components:

| Component | Role | Trainable? |
|-----------|------|------------|
| **Voice Encoder (VE)** | Extracts speaker embedding from reference audio | Frozen |
| **T3 Transformer** | LLaMA-based (0.5B params). Takes text tokens + speech tokens + speaker conditioning → predicts speech tokens | **Yes — only this trains** |
| **S3Gen Decoder** | Flow matching (10 steps) converts speech tokens → mel → waveform (24kHz) via HiFi-GAN | Frozen |

### T3 Training Details

T3 is a **dual-stream** model — it takes text tokens and speech tokens as **separate inputs** (not concatenated). It has **two output heads**:

- `text_logits` → predicts next text token (auxiliary loss)
- `speech_logits` → predicts next speech token (main loss)
- `total_loss = loss_text + loss_speech`

**Loss masking:** The prompt region of speech tokens (first ~3s of the clip) is masked with `IGNORE_ID = -100`. Loss is only computed on the speech tokens **after** the prompt.

**20% CFG dropout:** During training, 20% of samples have `speaker_emb` and `prompt_tokens` zeroed out. This enables classifier-free guidance at inference time (controlled by `cfg_weight` parameter).

### Conditioning (T3Cond)

Each training sample provides:

| Tensor | Source | Purpose |
|--------|--------|---------|
| `speaker_emb` | VE(full audio) | Speaker identity |
| `prompt_tokens` | S3 tokenizer(first 3s of clip) | Voice style/prosody context |
| `text_tokens` | Grapheme tokenizer(text) | What to say |
| `speech_tokens` | S3 tokenizer(full clip) + STOP token | Prediction target |

---

## 2. Data Preprocessing

### 2.1 Preprocessing Rules (for Arjun4707/gu-hi-tts)

The raw dataset is scraped from YouTube podcasts — requires aggressive filtering:

| Filter | Range | Purpose |
|--------|-------|---------|
| CPS (Characters Per Second) | 4–25 | Removes misaligned audio/text |
| Speaker Diarization | Single speaker only | Prevents stopping/blabbering artifacts |
| Text Cleaning | Strip `>` and `\|` | Prevents CSV delimiter issues |
| Duration | 2–20 seconds | Too short = weak training signal, too long = memory issues |

### 2.2 Preprocessing Script

```python
# preprocess_hf_dataset.py
# Full script handles:
# 1. Check if Arjun4707/clean-gu-hi-tts exists → download directly (fast path)
# 2. Otherwise: load raw data → CPS filter → diarization → save → upload clean version

# Key dependencies:
# pip install pyannote.audio datasets soundfile pyarrow huggingface_hub --break-system-packages
```

**Running preprocessing:**

```bash
huggingface-cli login

# Full pipeline (first time — runs diarization, uploads clean dataset to HF)
python preprocess_hf_dataset.py --language gu --hf_token YOUR_TOKEN

# Next time on any machine — auto-detects clean repo and downloads directly
python preprocess_hf_dataset.py --language gu --hf_token YOUR_TOKEN
```

### 2.3 Pyannote Diarization — API Compatibility

Pyannote `>= 3.3` changed the API. The diarization function must handle both:

```python
def is_single_speaker(audio_array, sr, diarization_pipeline):
    import torch
    waveform = torch.from_numpy(audio_array).unsqueeze(0).float()
    audio_input = {"waveform": waveform, "sample_rate": sr}

    try:
        output = diarization_pipeline(audio_input)
        speakers = set()
        segments = []

        if hasattr(output, 'itertracks'):
            # Old API (pyannote.audio <= 3.2)
            for turn, _, speaker in output.itertracks(yield_label=True):
                speakers.add(speaker)
                segments.append((turn.start, turn.end))
        elif hasattr(output, 'speaker_diarization'):
            # New API (pyannote.audio >= 3.3)
            for turn, speaker in output.speaker_diarization:
                speakers.add(speaker)
                segments.append((turn.start, turn.end))

        # Check for overlapping segments
        has_overlap = False
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                s1_start, s1_end = segments[i]
                s2_start, s2_end = segments[j]
                if s1_start < s2_end and s2_start < s1_end:
                    has_overlap = True
                    break
            if has_overlap:
                break

        return len(speakers) <= 1 and not has_overlap
    except Exception as e:
        return False
```

Also use `token=` not `use_auth_token=` for newer pyannote:

```python
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=hf_token  # NOT use_auth_token
)
```

### 2.4 Preprocessing Results

| Metric | Count |
|--------|-------|
| Total rows in dataset | 56,919 |
| After Gujarati language filter | 42,564 |
| After CPS + duration + diarization | ~33,851 |
| Rejection rate | ~20% |

### 2.5 Convert to LJSpeech Format

Chatterbox expects pipe-delimited metadata (no header):

```bash
# Convert preprocessed CSV to LJSpeech pipe-delimited format
python3 -c "
import csv
with open('preprocessed_data/gu/metadata.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
with open('chatterbox-finetuning/MyTTSDataset/metadata.csv', 'w') as f:
    for r in rows:
        wav = r['file_name']
        text = r['transcription'].strip('\"')
        f.write(f'{wav}|{text}|{text}\n')
print(f'Converted {len(rows)} rows')
"

# Copy wavs
cp -r preprocessed_data/gu/wavs/* chatterbox-finetuning/MyTTSDataset/wavs/
```

Format: `wavs/filename.wav|raw text|normalized text` (one per line, no header)

---

## 3. Environment Setup

### 3.1 Dependencies

```bash
# Chatterbox needs Python 3.11
# On Lightning.ai, the default cloudspace env has 3.11 or 3.12

cd ~
git clone https://github.com/gokhaneraslan/chatterbox-finetuning.git
cd chatterbox-finetuning

pip install -r requirements.txt --break-system-packages
pip install datasets soundfile safetensors --break-system-packages
```

### 3.2 Torch Version Alignment

Lightning.ai may have mismatched torch/torchaudio/torchvision. Fix:

```bash
# If torchaudio import fails with undefined symbol error:
pip install --force-reinstall torchaudio --break-system-packages

# If torch gets bumped (e.g. to 2.11.0), torchvision must follow:
pip install --force-reinstall torchvision --break-system-packages

# Verify alignment:
python -c "import torch; import torchaudio; import torchvision; print(f'torch={torch.__version__}, torchaudio={torchaudio.__version__}, torchvision={torchvision.__version__}')"
# Example output: torch=2.11.0+cu130, torchaudio=2.11.0+cu130, torchvision=0.26.0+cu130
```

### 3.3 Download Pretrained Models

```bash
cd ~/chatterbox-finetuning
python setup.py
```

Downloads to `pretrained_models/`:
- `ve.safetensors` — Voice Encoder
- `t3_cfg.safetensors` → saved as `t3.safetensors` — T3 Transformer
- `s3gen.safetensors` — Speech Decoder
- `tokenizer.json` — Grapheme tokenizer (from `grapheme_mtl_merged_expanded_v1.json`)
- `conds.pt` — Conditioning embeddings

---

## 4. Vocabulary Extension

Chatterbox's base tokenizer has **2,454 tokens** covering 23 languages. **Gujarati is NOT included** — zero Gujarati characters in the base vocab.

### 4.1 Extend Tokenizer with Gujarati Characters

```bash
python3 -c "
import json

with open('pretrained_models/tokenizer.json', 'r') as f:
    tok = json.load(f)

vocab = tok['model']['vocab']
print(f'Current vocab size: {len(vocab)}')

# Extract unique Gujarati chars from dataset (Unicode U+0A80 to U+0AFF)
gu_chars = set()
with open('MyTTSDataset/metadata.csv', 'r') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) >= 2:
            for ch in parts[1]:
                if '\u0A80' <= ch <= '\u0AFF':
                    gu_chars.add(ch)

new_chars = sorted(gu_chars - set(vocab.keys()))
print(f'Unique Gujarati chars: {len(gu_chars)}')
print(f'New chars to add: {len(new_chars)}')

next_id = max(vocab.values()) + 1
for ch in new_chars:
    vocab[ch] = next_id
    next_id += 1

tok['model']['vocab'] = vocab
with open('pretrained_models/tokenizer.json', 'w', encoding='utf-8') as f:
    json.dump(tok, f, ensure_ascii=False, indent=2)

print(f'New vocab size: {len(vocab)}')
print(f'=> Set NEW_VOCAB_SIZE = {len(vocab)} in src/config.py')
"
```

**Result:** Vocab extended from 2,454 → **2,514** (60 Gujarati characters added).

### 4.2 Embedding Resize

The training script (`src/model.py`) handles this automatically via `resize_and_load_t3_weights()`:

- Embedding layer (`text_emb.weight`): 704 existing tokens preserved, 1810 new tokens initialized with mean of existing embeddings
- Output head (`text_head.weight`): Same — 704 preserved, 1810 initialized with mean

No manual weight surgery needed — just set the correct `NEW_VOCAB_SIZE` in config.

---

## 5. Training Configuration & Patches

### 5.1 Config File (src/config.py)

Key settings that needed changing from defaults:

```python
@dataclass
class TrainConfig:
    # Paths
    csv_path: str = "./MyTTSDataset/metadata.csv"
    wav_dir: str = "./MyTTSDataset"       # NOT ./MyTTSDataset/wavs (metadata has wavs/ prefix)
    preprocessed_dir = "./MyTTSDataset/preprocess"
    output_dir: str = "./chatterbox_output"
    model_dir: str = "./pretrained_models"
    
    # Mode
    is_turbo: bool = False                # Standard mode (LLaMA-based)
    ljspeech = True
    preprocess = True                     # Set False after first successful preprocessing
    
    # Vocabulary — MUST match extended tokenizer
    NEW_VOCAB_SIZE = 2514                 # After Gujarati extension (was 2454)
    new_vocab_size: int = 52260 if is_turbo else 2514
    
    # Training — optimized for L4 GPU + ~34K clips
    batch_size: int = 4
    grad_accum: int = 8                   # Effective batch = 32
    learning_rate: float = 5e-5
    num_epochs: int = 10                  # NOT 150 (that's for 1hr of audio, we have ~30hr)
    epochs = 10
    warmup_steps = 100
    save_every_n_epochs = 1
    save_steps: int = 500
    
    # Critical for Lightning.ai
    dataloader_num_workers: int = 0       # NOT 8 (OOM prevention)
    
    # Audio
    sample_rate = 16000
    max_text_len: int = 256
    max_speech_len: int = 850
    prompt_duration: float = 3.0
```

### 5.2 Path Fix — Double `wavs/` Issue

**Problem:** Metadata has `wavs/filename.wav` but `wav_dir` defaulted to `./MyTTSDataset/wavs`, creating `wavs/wavs/filename.wav`.

**Fix:**
```bash
sed -i 's|wav_dir: str = "./MyTTSDataset/wavs"|wav_dir: str = "./MyTTSDataset"|' src/config.py
```

### 5.3 Preprocess Directory Fix

**Problem:** Preprocessing tried to save `.pt` files to `./MyTTSDataset/preprocess/wavs/` which didn't exist (because filenames include `wavs/` prefix).

**Fix:**
```bash
mkdir -p ./MyTTSDataset/preprocess/wavs
```

---

## 6. Training

### 6.1 Preprocessing (One-Time)

First run with `preprocess = True`:

```bash
cd ~/chatterbox-finetuning
CUDA_VISIBLE_DEVICES=0 python train.py
```

Preprocessing extracts for each clip:
- Speaker embedding (via VE)
- Prompt speech tokens (S3 tokenizer on first 3s)
- Full speech tokens (S3 tokenizer on entire clip)
- Text tokens (grapheme tokenizer)

Saves as `.pt` files in `./MyTTSDataset/preprocess/wavs/`.

**Duration:** ~1.5-2 hours for 33,851 clips on L4 GPU.

After preprocessing completes (`Success: 33851/33851`), set `preprocess = False` in config.

### 6.2 Training Run

```bash
# Set preprocess = False in src/config.py first
CUDA_VISIBLE_DEVICES=0 python train.py
```

### 6.3 Training Output

```
chatterbox_output/
├── t3_finetuned.safetensors    ← Final fine-tuned T3 weights (~2.1 GB)
├── checkpoint-3500/
├── checkpoint-4000/
├── checkpoint-4500/
├── checkpoint-5000/
├── checkpoint-5290/
└── runs/                        ← TensorBoard logs
```

### 6.4 Training Log (Healthy)

```
Creating new T3 model with vocab size: 2514
Embedding layer: 704 tokens preserved.
Embedding layer: 1810 new tokens initialized with mean.
Output head: 704 tokens preserved.
Output head: 1810 new neurons initialized with mean.
All weights transferred successfully (Mean Initialization applied)!
Freezing S3Gen and VoiceEncoder...
```

---

## 7. Inference

### 7.1 Basic Inference

```python
import torch
import torchaudio as ta
from safetensors.torch import load_file
from chatterbox.tts import ChatterboxTTS  # or src.chatterbox_.tts if in finetuning repo

NEW_VOCAB_SIZE = 2514

# Load base model
model = ChatterboxTTS.from_local('./pretrained_models', device='cuda')

# Resize embeddings to match fine-tuned vocab
emb_dim = model.t3.text_emb.embedding_dim
model.t3.text_emb = torch.nn.Embedding(NEW_VOCAB_SIZE, emb_dim).to('cuda')
model.t3.text_head = torch.nn.Linear(emb_dim, NEW_VOCAB_SIZE, bias=False).to('cuda')

# Load fine-tuned T3
state_dict = load_file('chatterbox_output/t3_finetuned.safetensors')
model.t3.load_state_dict(state_dict, strict=False)

# Generate with voice cloning
wav = model.generate(
    "ગુજરાતી ભાષામાં આ એક પરીક્ષણ છે.",
    audio_prompt_path="speaker_reference/reference_gu.wav",
    exaggeration=0.5,
    cfg_weight=0.5
)
ta.save("test_gujarati.wav", wav, model.sr)
```

### 7.2 Batch Inference Script

```bash
# Run with default Gujarati test texts
python infer_chatterbox.py --ref_audio speaker_reference/reference_gu.wav

# Custom texts
python infer_chatterbox.py --ref_audio ref.wav \
  --texts "મારું નામ અર્જુન છે." "ગુજરાત સુંદર રાજ્ય છે."

# Texts from file (one per line)
python infer_chatterbox.py --texts_file my_texts.txt --ref_audio ref.wav

# Multiple reference audios (clones each voice)
python infer_chatterbox.py --ref_dir speaker_reference/

# Download from HuggingFace on a new machine
python infer_chatterbox.py --from_hf --hf_token TOKEN --ref_audio ref.wav

# Tune generation params
python infer_chatterbox.py --ref_audio ref.wav --exaggeration 0.3 --cfg_weight 0.3
```

### 7.3 Generation Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `exaggeration` | 0.5 | Emotion intensity. Higher = more dramatic, faster. Lower = more monotone, stable. |
| `cfg_weight` | 0.5 | Adherence to reference voice. 0.0 = ignore ref style. 0.3 = subtle. 0.5 = balanced. |

**Recommended for Gujarati:**
- Stable output: `exaggeration=0.3, cfg_weight=0.3`
- Balanced: `exaggeration=0.5, cfg_weight=0.5` (default)
- Expressive: `exaggeration=0.7, cfg_weight=0.3`

### 7.4 Reference Audio Tips

- Use 7-10 seconds of clean, single-speaker Gujarati audio
- Should match the speaking style you want in the output
- No background noise or music
- Any clip from your preprocessed dataset works as reference

---

## 8. Upload to HuggingFace

```bash
# Upload fine-tuned files only (~2GB)
python upload_chatterbox_hf.py --hf_token YOUR_TOKEN

# Include base model files for self-contained repo (~4GB)
python upload_chatterbox_hf.py --hf_token YOUR_TOKEN --include_base_models
```

Uploads to `Arjun4707/chatterbox-gujarati` (private):
- `t3_finetuned.safetensors` — Fine-tuned T3 weights
- `tokenizer.json` — Extended tokenizer with Gujarati chars
- `conds.pt` — Conditioning embeddings
- `README.md` — Model card with usage instructions

---

## 9. Key Lessons & Known Issues

### 9.1 Issues Encountered & Fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| torchaudio import crash | `undefined symbol: torch_library_impl` | `pip install --force-reinstall torchaudio --break-system-packages` |
| torchvision mismatch | `requires torch==2.6.0, but you have torch 2.11.0` | `pip install --force-reinstall torchvision --break-system-packages` |
| pyannote `use_auth_token` | `TypeError: unexpected keyword argument` | Change to `token=hf_token` |
| pyannote `itertracks` | `'DiarizeOutput' has no attribute 'itertracks'` | Handle both old `.itertracks()` and new `.speaker_diarization` API |
| Preprocessing 0/33851 success | Silent file-not-found (double `wavs/` path) | Set `wav_dir = "./MyTTSDataset"` (not `./MyTTSDataset/wavs`) |
| Preprocess save fails | `Parent directory preprocess/wavs does not exist` | `mkdir -p ./MyTTSDataset/preprocess/wavs` |
| Safetensors load with torch.load | `UnpicklingError: invalid load key` | Use `from safetensors.torch import load_file` |
| T3 state_dict size mismatch | `text_emb.weight: [2514, 1024] vs [704, 1024]` | Resize embeddings before loading fine-tuned weights |
| torchcodec warning | `Could not load libtorchcodec` | Harmless — FFmpeg not installed, pyannote doesn't need it |
| numpy empty slice warning | `Mean of empty slice` | Harmless — pyannote on silent segments |
| `pkg_resources deprecated` warning | `pkg_resources is deprecated` | Harmless — from perth watermarker |
| `Reference mel length` warning | `not equal to 2 * reference token length` | Harmless — S3Gen internal rounding |

### 9.2 Quality Observations

| Text Length | Quality | Notes |
|-------------|---------|-------|
| Very short (1-3 words) | Poor | Generates 4-5s of gibberish. Architecture not suited for sub-2s output. |
| Short-medium (5-15 words) | Decent | Pronounces Gujarati properly. Some start/end artifacts. |
| Medium (15-30 words) | Good | Clear pronunciation, stable generation. |
| Long (30+ words) | Good with artifacts | Sometimes has repetition or extra words at boundaries. |

### 9.3 Data Quality Lessons (Consistent Across All Models)

- **Multi-speaker clips cause stopping and blabbering.** Always run diarization on scraped/podcast data.
- **CPS filter (4-25) catches misaligned audio/text.** Essential for scraped data.
- **Clips under 3-4 seconds are degenerate for Chatterbox.** The 3-second prompt overlaps almost entirely with the target, giving near-zero training signal.
- **Quality >> quantity.** 33K clean clips outperform 42K noisy clips.

---

## 10. Hyperparameter Reference

### L4 GPU (24 GB VRAM)

| Parameter | Conservative | Used | Aggressive |
|-----------|-------------|------|------------|
| batch_size | 2 | 4 | 6 |
| grad_accum | 16 | 8 | 4 |
| effective_batch | 32 | 32 | 24 |
| learning_rate | 1e-5 | 5e-5 | 1e-4 |
| epochs (for ~30hr data) | 3 | 10 | 15 |
| warmup_steps | 200 | 100 | 50 |
| dataloader_num_workers | 0 | 0 | 2 (risky on Lightning.ai) |
| max_text_len | 200 | 256 | 300 |
| max_speech_len | 600 | 850 | 1000 |

### Inference Parameters

| Setting | exaggeration | cfg_weight | Best for |
|---------|-------------|------------|----------|
| Most stable | 0.3 | 0.3 | Testing, short texts |
| Balanced | 0.5 | 0.5 | General use |
| Expressive | 0.7 | 0.3 | Narration, longer texts |
| No voice match | 0.5 | 0.0 | Cross-speaker testing |

---

## Appendix: File Structure

```
chatterbox-finetuning/
├── pretrained_models/
│   ├── ve.safetensors              ← Voice Encoder (frozen)
│   ├── s3gen.safetensors           ← Speech Decoder (frozen)
│   ├── t3_cfg.safetensors          ← Original T3 weights
│   ├── tokenizer.json              ← Extended with Gujarati (2514 tokens)
│   └── conds.pt                    ← Conditioning embeddings
├── MyTTSDataset/
│   ├── metadata.csv                ← Pipe-delimited: wavs/file.wav|text|text
│   ├── wavs/                       ← 33,851 preprocessed Gujarati clips (24kHz)
│   └── preprocess/
│       └── wavs/                   ← .pt files (speaker_emb, prompt_tokens, text_tokens, speech_tokens)
├── chatterbox_output/
│   ├── t3_finetuned.safetensors    ← Fine-tuned T3 (~2.1 GB)
│   ├── checkpoint-*/               ← Intermediate checkpoints
│   └── runs/                       ← TensorBoard logs
├── speaker_reference/
│   └── reference_gu.wav            ← Reference audio for inference
├── src/
│   ├── config.py                   ← Training configuration
│   ├── dataset.py                  ← Dataset + collator
│   ├── model.py                    ← Weight transfer + training wrapper
│   ├── preprocess_ljspeech.py      ← Preprocessing script
│   └── chatterbox_/                ← Modified Chatterbox source
├── train.py                        ← Main training script
├── inference.py                    ← Repo's inference script
├── infer_chatterbox.py             ← Batch inference script (custom)
├── upload_chatterbox_hf.py         ← HuggingFace upload script (custom)
└── setup.py                        ← Downloads pretrained models
```

---

## Appendix: Quick Command Reference

```bash
# ── Preprocessing ───────────────────────────────────────────────
python preprocess_hf_dataset.py --language gu --hf_token TOKEN

# ── Convert to LJSpeech format ──────────────────────────────────
# (see Section 2.5)

# ── Setup ───────────────────────────────────────────────────────
cd ~/chatterbox-finetuning
python setup.py

# ── Extend vocab ────────────────────────────────────────────────
# (see Section 4.1)

# ── Fix paths ───────────────────────────────────────────────────
sed -i 's|wav_dir: str = "./MyTTSDataset/wavs"|wav_dir: str = "./MyTTSDataset"|' src/config.py
mkdir -p ./MyTTSDataset/preprocess/wavs

# ── Train (first run: preprocess=True) ──────────────────────────
CUDA_VISIBLE_DEVICES=0 python train.py

# ── Train (subsequent runs: preprocess=False) ───────────────────
CUDA_VISIBLE_DEVICES=0 python train.py

# ── Inference ───────────────────────────────────────────────────
python infer_chatterbox.py --ref_audio speaker_reference/reference_gu.wav

# ── Upload to HuggingFace ──────────────────────────────────────
python upload_chatterbox_hf.py --hf_token TOKEN --include_base_models
```
