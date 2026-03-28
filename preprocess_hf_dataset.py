#!/usr/bin/env python3
"""
preprocess_hf_dataset.py
Preprocesses Arjun4707/gu-hi-tts for Chatterbox and CosyVoice 3 training.
Applies: CPS filter (4-25), speaker diarization, text cleaning, duration filter.

Logic:
  1. Check if Arjun4707/clean-gu-hi-tts already exists on HuggingFace.
     - If YES → download it directly (skip all filtering) and save to disk.
     - If NO  → download raw data, run full preprocessing, save locally,
                then upload the clean dataset to Arjun4707/clean-gu-hi-tts.
"""

import os
import io
import json
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from datasets import load_dataset

# ─── Configuration ──────────────────────────────────────────────────
HF_RAW_REPO = "Arjun4707/gu-hi-tts"
HF_CLEAN_REPO = "Arjun4707/clean-gu-hi-tts"
CPS_MIN = 4.0
CPS_MAX = 25.0
MIN_DURATION = 2.0    # seconds — clips shorter than this are low quality
MAX_DURATION = 20.0   # seconds — upper bound for both models
SHARD_SIZE = 300      # rows per parquet shard (matches original dataset)

# ─── Check if clean dataset exists ──────────────────────────────────

def clean_repo_exists(hf_token=None):
    """Check if Arjun4707/clean-gu-hi-tts exists and has data."""
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    try:
        info = api.dataset_info(HF_CLEAN_REPO)
        # Check if it has any parquet files
        files = api.list_repo_files(HF_CLEAN_REPO, repo_type="dataset")
        has_data = any(f.endswith(".parquet") for f in files)
        if has_data:
            print(f"✅ Clean dataset found at {HF_CLEAN_REPO} with data files.")
            return True
        else:
            print(f"⚠️  Clean repo exists at {HF_CLEAN_REPO} but has no parquet files.")
            return False
    except Exception:
        print(f"ℹ️  Clean dataset {HF_CLEAN_REPO} does not exist yet.")
        return False

# ─── Download clean dataset (fast path) ─────────────────────────────

def download_clean_dataset(language, output_dir, hf_cache_dir, hf_token=None):
    """Download already-clean data from HF and save to disk."""
    print(f"\n{'='*60}")
    print(f"DOWNLOADING CLEAN DATASET — {language.upper()}")
    print(f"{'='*60}")

    ds = load_dataset(HF_CLEAN_REPO, split="train", cache_dir=hf_cache_dir,
                      token=hf_token)
    print(f"Total clean rows: {len(ds)}")

    # Filter by language
    ds_lang = ds.filter(lambda x: x["language"] == language)
    print(f"After language filter ({language}): {len(ds_lang)} rows")

    # Save to disk
    wavs_dir = os.path.join(output_dir, language, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    metadata = []

    for idx, row in enumerate(ds_lang):
        if idx % 1000 == 0 and idx > 0:
            print(f"  Saving {idx}/{len(ds_lang)}...")

        clip_id = row["id"]
        raw_bytes = row["audio"]["bytes"]
        sr = row["audio"]["sampling_rate"]
        audio_array, _ = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        duration = len(audio_array) / sr

        wav_filename = f"{clip_id}.wav"
        wav_path = os.path.join(wavs_dir, wav_filename)
        sf.write(wav_path, audio_array, sr, subtype="PCM_16")

        metadata.append({
            "wav_path": f"wavs/{wav_filename}",
            "text": row["text"],
            "speaker": row.get("speaker_id", f"spk_hf_{language}"),
            "duration": round(duration, 3),
        })

    _save_all_formats(metadata, output_dir, language)
    print(f"\nDone — {len(metadata)} clips saved from clean repo.")
    return metadata

# ─── Speaker Diarization Setup ──────────────────────────────────────

def setup_diarization_pipeline(hf_token):
    """Initialize pyannote diarization pipeline."""
    from pyannote.audio import Pipeline
    import pyannote.audio
    version = pyannote.audio.__version__

    # Use community-1 for >= 3.3, else 3.1
    if tuple(int(x) for x in version.split('.')[:2]) >= (3, 3):
        model_name = "pyannote/speaker-diarization-community-1"
    else:
        model_name = "pyannote/speaker-diarization-3.1"

    print(f"  pyannote.audio version: {version}, using: {model_name}")
    pipeline = Pipeline.from_pretrained(model_name, token=hf_token)

    import torch
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    return pipeline
    
def is_single_speaker(audio_array, sr, diarization_pipeline):
    """
    Check if audio contains exactly one speaker.
    Works with both old (.itertracks) and new (.speaker_diarization) pyannote API.
    """
    import torch

    waveform = torch.from_numpy(audio_array).unsqueeze(0).float()
    audio_input = {"waveform": waveform, "sample_rate": sr}

    try:
        output = diarization_pipeline(audio_input)

        # Handle both old and new pyannote API
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
        else:
            print(f"  [WARN] Unknown diarization output type: {type(output)}")
            return False

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
        print(f"  [WARN] Diarization failed: {e}, rejecting clip")
        return False

def clean_text(text):
    """Clean text: remove > and | characters, strip whitespace."""
    text = text.replace(">", "").replace("|", " ").strip()
    text = " ".join(text.split())
    return text

def compute_cps(text, duration_sec):
    """Compute characters per second."""
    if duration_sec <= 0:
        return 0.0
    return len(text) / duration_sec

# ─── Save helper (both Chatterbox CSV + CosyVoice Kaldi) ───────────

def _save_all_formats(metadata, output_dir, language):
    """Save metadata in both Chatterbox CSV and CosyVoice Kaldi formats."""
    lang_dir = os.path.join(output_dir, language)
    os.makedirs(lang_dir, exist_ok=True)

    # Chatterbox CSV
    csv_path = os.path.join(lang_dir, "metadata.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("file_name,transcription,duration_seconds\n")
        for m in metadata:
            text_escaped = m["text"].replace('"', '""')
            f.write(f'{m["wav_path"]},"{text_escaped}",{m["duration"]}\n')

    # CosyVoice Kaldi format
    wavscp_path = os.path.join(lang_dir, "wav.scp")
    text_path = os.path.join(lang_dir, "text")
    utt2spk_path = os.path.join(lang_dir, "utt2spk")
    with open(wavscp_path, "w") as wf, \
         open(text_path, "w", encoding="utf-8") as tf, \
         open(utt2spk_path, "w") as uf:
        for m in metadata:
            utt_id = Path(m["wav_path"]).stem
            abs_wav = os.path.abspath(os.path.join(lang_dir, m["wav_path"]))
            wf.write(f"{utt_id} {abs_wav}\n")
            tf.write(f"{utt_id} {m['text']}\n")
            uf.write(f"{utt_id} {m['speaker']}\n")

    print(f"  CSV (Chatterbox) : {csv_path}")
    print(f"  wav.scp (CosyVoice): {wavscp_path}")
    print(f"  text (CosyVoice)   : {text_path}")
    print(f"  utt2spk (CosyVoice): {utt2spk_path}")

# ─── Upload clean dataset to HuggingFace ────────────────────────────

def upload_clean_dataset(output_dir, hf_token=None):
    """
    Upload all preprocessed clean clips to Arjun4707/clean-gu-hi-tts.
    Preserves the same schema as the original dataset so downstream
    code can load it identically.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)

    # Create the repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=HF_CLEAN_REPO,
            repo_type="dataset",
            private=True,
            exist_ok=True
        )
        print(f"✅ Repo {HF_CLEAN_REPO} ready.")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Collect all clean clips across languages
    all_rows = []
    for language in ["gu", "hi"]:
        checkpoint_file = os.path.join(output_dir,
                                       f"preprocess_checkpoint_{language}.json")
        if not os.path.exists(checkpoint_file):
            print(f"  No checkpoint for {language}, skipping.")
            continue

        with open(checkpoint_file, "r") as f:
            ckpt = json.load(f)

        metadata = ckpt.get("metadata", [])
        wavs_dir = os.path.join(output_dir, language, "wavs")

        print(f"  Packing {len(metadata)} {language} clips into parquet...")

        for m in metadata:
            wav_path = os.path.join(output_dir, language, m["wav_path"])
            if not os.path.exists(wav_path):
                continue

            # Read wav as raw bytes (same format as original dataset)
            with open(wav_path, "rb") as wf:
                wav_bytes = wf.read()

            all_rows.append({
                "id": Path(m["wav_path"]).stem,
                "audio": {"bytes": wav_bytes, "sampling_rate": 24000},
                "text": m["text"],
                "language": language,
                "duration_sec": m["duration"],
                "speaker_id": m["speaker"],
                "cps": m.get("cps", 0.0),
            })

    if not all_rows:
        print("No rows to upload!")
        return

    print(f"\nTotal clean rows: {len(all_rows)}")

    # Create parquet shards
    shard_idx = 0
    for start in range(0, len(all_rows), SHARD_SIZE):
        chunk = all_rows[start : start + SHARD_SIZE]

        # Build arrow table
        # Audio is stored as struct{bytes, sampling_rate} to match original
        audio_structs = [
            {"bytes": r["audio"]["bytes"],
             "sampling_rate": r["audio"]["sampling_rate"]}
            for r in chunk
        ]

        table = pa.table({
            "id": [r["id"] for r in chunk],
            "audio": audio_structs,
            "text": [r["text"] for r in chunk],
            "language": [r["language"] for r in chunk],
            "duration_sec": [r["duration_sec"] for r in chunk],
            "speaker_id": [r["speaker_id"] for r in chunk],
            "cps": [r["cps"] for r in chunk],
        })

        shard_name = f"data/train/shard_{shard_idx:05d}.parquet"
        local_path = os.path.join(output_dir, "upload_staging", shard_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        pq.write_table(table, local_path, compression="snappy")

        # Upload shard
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=shard_name,
            repo_id=HF_CLEAN_REPO,
            repo_type="dataset",
        )
        print(f"  Uploaded {shard_name} ({len(chunk)} rows)")
        shard_idx += 1

    # Upload a README
    readme = f"""---
dataset_info:
  features:
    - name: id
      dtype: string
    - name: audio
      dtype:
        struct:
          - name: bytes
            dtype: binary
          - name: sampling_rate
            dtype: int32
    - name: text
      dtype: string
    - name: language
      dtype: string
    - name: duration_sec
      dtype: float32
    - name: speaker_id
      dtype: string
    - name: cps
      dtype: float32
  splits:
    - name: train
      num_examples: {len(all_rows)}
---

# Clean Gujarati-Hindi TTS Dataset

Cleaned version of `Arjun4707/gu-hi-tts`.

## Preprocessing applied:
- **CPS filter**: 4-25 characters/second (removes misaligned audio/text)
- **Speaker diarization**: pyannote-audio — only single-speaker clips kept
- **Text cleaning**: stripped `>` and `|` characters
- **Duration filter**: 2-20 seconds

## Stats:
- Total clean clips: {len(all_rows)}
- Gujarati: {sum(1 for r in all_rows if r['language']=='gu')}
- Hindi: {sum(1 for r in all_rows if r['language']=='hi')}
- Audio: 24kHz, mono, PCM-16
"""
    readme_path = os.path.join(output_dir, "upload_staging", "README.md")
    with open(readme_path, "w") as f:
        f.write(readme)
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=HF_CLEAN_REPO,
        repo_type="dataset",
    )

    print(f"\n✅ Clean dataset uploaded to {HF_CLEAN_REPO}")
    print(f"   {shard_idx} shards, {len(all_rows)} total rows")

# ─── Main preprocessing pipeline ───────────────────────────────────

def preprocess_dataset(
    language="gu",
    output_dir="preprocessed_data",
    hf_cache_dir="hf_cache",
    run_diarization=True,
    hf_token=None,
    max_samples=None,
    skip_upload=False,
):
    """
    Full preprocessing pipeline:
    1. Check if Arjun4707/clean-gu-hi-tts exists → if yes, download and done
    2. Otherwise: load raw → filter → clean → diarize → save locally → upload
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(hf_cache_dir, exist_ok=True)

    # ── FAST PATH: clean repo already exists ────────────────────────
    if clean_repo_exists(hf_token):
        metadata = download_clean_dataset(language, output_dir,
                                          hf_cache_dir, hf_token)
        return {"kept": len(metadata)}, metadata

    # ── SLOW PATH: full preprocessing ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"FULL PREPROCESSING — {language.upper()}")
    print(f"{'='*60}")

    stats = {
        "total": 0, "lang_filtered": 0, "text_cleaned": 0,
        "cps_rejected": 0, "duration_rejected": 0,
        "multi_speaker_rejected": 0, "kept": 0
    }

    # Load raw dataset (cached)
    print(f"Loading dataset from {HF_RAW_REPO} (cached in {hf_cache_dir})...")
    ds = load_dataset(HF_RAW_REPO, split="train", cache_dir=hf_cache_dir,
                      token=hf_token)
    print(f"Total rows: {len(ds)}")

    # Filter by language
    ds_lang = ds.filter(lambda x: x["language"] == language)
    stats["total"] = len(ds)
    stats["lang_filtered"] = len(ds_lang)
    print(f"After language filter ({language}): {len(ds_lang)} rows")

    if max_samples:
        ds_lang = ds_lang.select(range(min(max_samples, len(ds_lang))))
        print(f"Limited to {len(ds_lang)} samples for testing")

    # Setup diarization
    diar_pipeline = None
    if run_diarization:
        print("Loading pyannote diarization pipeline...")
        diar_pipeline = setup_diarization_pipeline(hf_token)
        print("Diarization pipeline ready.")

    # Process each clip
    wavs_dir = os.path.join(output_dir, language, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    metadata = []
    checkpoint_file = os.path.join(output_dir,
                                   f"preprocess_checkpoint_{language}.json")

    # Resume support
    processed_ids = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            ckpt = json.load(f)
            processed_ids = set(ckpt.get("processed_ids", []))
            metadata = ckpt.get("metadata", [])
            stats = ckpt.get("stats", stats)
        print(f"Resuming from checkpoint: {len(processed_ids)} already processed")

    for idx, row in enumerate(ds_lang):
        clip_id = row["id"]
        if clip_id in processed_ids:
            continue

        if idx % 500 == 0 and idx > 0:
            print(f"Processing {idx}/{len(ds_lang)}... kept={stats['kept']}")
            # Save checkpoint
            with open(checkpoint_file, "w") as f:
                json.dump({
                    "processed_ids": list(processed_ids),
                    "metadata": metadata,
                    "stats": stats
                }, f)

        # Decode audio
        raw_bytes = row["audio"]["bytes"]
        sr = row["audio"]["sampling_rate"]  # 24000
        audio_array, _ = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        duration = len(audio_array) / sr

        # ── Text cleaning ───────────────────────────────────────────
        text = clean_text(row["text"])
        if not text or len(text) < 2:
            processed_ids.add(clip_id)
            continue

        # ── Duration filter ─────────────────────────────────────────
        if duration < MIN_DURATION or duration > MAX_DURATION:
            stats["duration_rejected"] += 1
            processed_ids.add(clip_id)
            continue

        # ── CPS filter ──────────────────────────────────────────────
        cps = compute_cps(text, duration)
        if cps < CPS_MIN or cps > CPS_MAX:
            stats["cps_rejected"] += 1
            processed_ids.add(clip_id)
            continue

        # ── Speaker diarization ─────────────────────────────────────
        if run_diarization and diar_pipeline is not None:
            if not is_single_speaker(audio_array, sr, diar_pipeline):
                stats["multi_speaker_rejected"] += 1
                processed_ids.add(clip_id)
                continue

        # ── Save clip ───────────────────────────────────────────────
        wav_filename = f"{clip_id}.wav"
        wav_path = os.path.join(wavs_dir, wav_filename)
        sf.write(wav_path, audio_array, sr, subtype="PCM_16")

        metadata.append({
            "wav_path": f"wavs/{wav_filename}",
            "text": text,
            "speaker": "spk_hf_gu" if language == "gu" else "spk_hf_hi",
            "duration": round(duration, 3),
            "cps": round(cps, 2)
        })
        stats["kept"] += 1
        processed_ids.add(clip_id)

    # ── Final checkpoint ────────────────────────────────────────────
    with open(checkpoint_file, "w") as f:
        json.dump({
            "processed_ids": list(processed_ids),
            "metadata": metadata,
            "stats": stats
        }, f)

    # ── Save to disk (both formats) ─────────────────────────────────
    _save_all_formats(metadata, output_dir, language)

    print(f"\n{'='*60}")
    print(f"PREPROCESSING COMPLETE — {language.upper()}")
    print(f"{'='*60}")
    print(f"Total in dataset     : {stats['total']}")
    print(f"After language filter : {stats['lang_filtered']}")
    print(f"Duration rejected    : {stats['duration_rejected']}")
    print(f"CPS rejected         : {stats['cps_rejected']}")
    print(f"Multi-speaker rejected: {stats['multi_speaker_rejected']}")
    print(f"KEPT                 : {stats['kept']}")

    return stats, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="gu", choices=["gu", "hi"])
    parser.add_argument("--output_dir", default="preprocessed_data")
    parser.add_argument("--hf_cache_dir", default="hf_cache")
    parser.add_argument("--skip_diarization", action="store_true",
                        help="Skip speaker diarization (faster, less filtering)")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token for pyannote access")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples for testing")
    parser.add_argument("--skip_upload", action="store_true",
                        help="Skip uploading clean dataset to HuggingFace")
    args = parser.parse_args()

    preprocess_dataset(
        language=args.language,
        output_dir=args.output_dir,
        hf_cache_dir=args.hf_cache_dir,
        run_diarization=not args.skip_diarization,
        hf_token=args.hf_token,
        max_samples=args.max_samples,
        skip_upload=args.skip_upload,
    )

    # ── Upload to HF after BOTH languages are preprocessed ──────────
    # Run for both gu and hi first, then upload once:
    #   python preprocess_hf_dataset.py --language gu --hf_token TOKEN
    #   python preprocess_hf_dataset.py --language hi --hf_token TOKEN
    #   python -c "from preprocess_hf_dataset import upload_clean_dataset; upload_clean_dataset('preprocessed_data', 'TOKEN')"
    #
    # Or upload after each language:
    if not args.skip_upload and not clean_repo_exists(args.hf_token):
        print("\n📤 Uploading clean dataset to HuggingFace...")
        upload_clean_dataset(args.output_dir, args.hf_token)