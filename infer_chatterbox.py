#!/usr/bin/env python3
"""
infer_chatterbox.py
Batch inference for fine-tuned Chatterbox Gujarati model.
Supports: local checkpoints, HuggingFace download, multiple texts, multiple ref audios.
"""

import os
import sys
import argparse
import torch
import torchaudio as ta
from safetensors.torch import load_file
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────────────

NEW_VOCAB_SIZE = 2514  # Must match training config

# Default test texts (Gujarati)
DEFAULT_TEXTS = [
    # short-medium
    "ગુજરાતી ભાષામાં આ એક પરીક્ષણ છે.",
    "આજે હવામાન ખૂબ સરસ છે અને આકાશ સાફ છે.",
    "ગુજરાતનું ખાણીપીણી જગ પ્રસિદ્ધ છે.",
    "મારું નામ અર્જુન છે અને હું બેંગલુરુમાં રહું છું.",
    "આ મોડેલ ગુજરાતી ભાષા માટે ટ્રેઈન કરવામાં આવ્યું છે.",
    "શિક્ષણ એ જીવનનું સૌથી મહત્વનું સાધન છે.",
    
    # medium
    "ભારતની સૌથી મોટી સમસ્યા ગરીબી અને બેરોજગારી છે.",
    "ટેકનોલોજી દુનિયાને બદલી રહી છે અને આપણે તેની સાથે ચાલવું પડશે.",
    "વિજ્ઞાન અને ટેકનોલોજીનો વિકાસ માનવ જીવનને સરળ બનાવી રહ્યો છે.",
    "દરરોજ નિયમિત અભ્યાસ કરવાથી સફળતા મેળવવી સરળ બને છે.",
    
    # long (important for TTS stability)
    "જો આપણે સમયનો યોગ્ય ઉપયોગ કરીએ અને સતત મહેનત કરીએ તો કોઈપણ લક્ષ્ય પ્રાપ્ત કરવું અશક્ય નથી.",
    "આજના ઝડપી બદલાતા સમયમાં નવી ટેકનોલોજી શીખવી અને તેને પોતાના કામમાં લાગુ કરવી ખૂબ જ જરૂરી બની ગઈ છે.",
    "એક સારો સમાજ બનાવવા માટે દરેક વ્યક્તિએ પોતાની જવાબદારી સમજવી અને અન્ય લોકો સાથે સહકાર આપવો જોઈએ.",
    "જ્યારે આપણે નવા કૌશલ્યો શીખીએ છીએ ત્યારે આપણે માત્ર પોતાના વિકાસ માટે નહીં પરંતુ સમગ્ર સમાજના વિકાસ માટે યોગદાન આપીએ છીએ.",
    "પરિવાર, શિક્ષણ અને સંસ્કૃતિ માનવ જીવનના ત્રણ મહત્વપૂર્ણ સ્તંભ છે, જે વ્યક્તિના વ્યક્તિત્વ અને વિચારોને ઘડવામાં મહત્વપૂર્ણ ભૂમિકા ભજવે છે."
]

HF_REPO_ID = "Arjun4707/chatterbox-gujarati"

# ─── Model Loading ──────────────────────────────────────────────────

def load_model(
    pretrained_dir="./pretrained_models",
    checkpoint_path="./chatterbox_output/t3_finetuned.safetensors",
    device="cuda",
    from_hf=False,
    hf_repo=HF_REPO_ID,
    hf_token=None,
):
    """Load base Chatterbox model and replace T3 with fine-tuned weights."""

    # Add the finetuning repo to path if running from within it
    if os.path.exists("src/chatterbox_"):
        sys.path.insert(0, ".")
        from src.chatterbox_.tts import ChatterboxTTS
    else:
        from chatterbox.tts import ChatterboxTTS

    print("Loading base model...")
    if os.path.exists(pretrained_dir):
        model = ChatterboxTTS.from_local(pretrained_dir, device=device)
    else:
        model = ChatterboxTTS.from_pretrained(device=device)

    # Resize embeddings to match fine-tuned vocab
    print(f"Resizing T3 embeddings: 704 → {NEW_VOCAB_SIZE}")
    emb_dim = model.t3.text_emb.embedding_dim
    model.t3.text_emb = torch.nn.Embedding(NEW_VOCAB_SIZE, emb_dim).to(device)
    model.t3.text_head = torch.nn.Linear(emb_dim, NEW_VOCAB_SIZE, bias=False).to(device)

    # Load fine-tuned T3 weights
    if from_hf:
        from huggingface_hub import hf_hub_download
        print(f"Downloading fine-tuned T3 from {hf_repo}...")
        checkpoint_path = hf_hub_download(hf_repo, "t3_finetuned.safetensors", token=hf_token)
        tok_path = hf_hub_download(hf_repo, "tokenizer.json", token=hf_token)
        print(f"Downloaded to: {checkpoint_path}")

    print(f"Loading fine-tuned weights from: {checkpoint_path}")
    state_dict = load_file(checkpoint_path)
    model.t3.load_state_dict(state_dict, strict=False)
    print(f"Fine-tuned T3 loaded (vocab={NEW_VOCAB_SIZE})")

    return model


# ─── Inference ──────────────────────────────────────────────────────

def generate_batch(
    model,
    texts,
    ref_audio_path,
    output_dir="outputs_chatterbox",
    exaggeration=0.5,
    cfg_weight=0.5,
):
    """Generate audio for a list of texts using a single reference audio."""

    os.makedirs(output_dir, exist_ok=True)

    ref_name = Path(ref_audio_path).stem
    print(f"\nReference audio: {ref_audio_path}")
    print(f"Output dir: {output_dir}")
    print(f"Generating {len(texts)} clips...\n")

    results = []

    for i, text in enumerate(texts):
        try:
            print(f"[{i+1}/{len(texts)}] {text[:60]}...")

            wav = model.generate(
                text,
                audio_prompt_path=ref_audio_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

            filename = f"{ref_name}_sample_{i+1:03d}.wav"
            out_path = os.path.join(output_dir, filename)
            ta.save(out_path, wav, model.sr)

            duration = wav.shape[-1] / model.sr
            print(f"  Saved: {filename} ({duration:.1f}s)")

            results.append({
                "text": text,
                "file": filename,
                "duration": round(duration, 2),
                "status": "ok"
            })

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "text": text,
                "file": None,
                "duration": 0,
                "status": f"error: {e}"
            })

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    total_dur = sum(r["duration"] for r in results)
    print(f"\nDone: {ok}/{len(texts)} succeeded, {total_dur:.1f}s total audio")

    # Save results log
    import json
    log_path = os.path.join(output_dir, "generation_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Log saved: {log_path}")

    return results


def generate_multi_ref(
    model,
    texts,
    ref_audio_dir,
    output_dir="outputs_chatterbox",
    exaggeration=0.5,
    cfg_weight=0.5,
):
    """Generate audio for all texts with each reference audio in a directory."""

    ref_files = sorted([
        f for f in os.listdir(ref_audio_dir)
        if f.endswith((".wav", ".mp3", ".flac"))
    ])

    if not ref_files:
        print(f"No audio files found in {ref_audio_dir}")
        return

    print(f"Found {len(ref_files)} reference audios in {ref_audio_dir}")

    for ref_file in ref_files:
        ref_path = os.path.join(ref_audio_dir, ref_file)
        ref_name = Path(ref_file).stem
        ref_output_dir = os.path.join(output_dir, ref_name)

        print(f"\n{'='*60}")
        print(f"Reference: {ref_file}")
        print(f"{'='*60}")

        generate_batch(
            model=model,
            texts=texts,
            ref_audio_path=ref_path,
            output_dir=ref_output_dir,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )


# ─── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatterbox Gujarati Inference")

    # Model source
    parser.add_argument("--pretrained_dir", default="./pretrained_models",
                        help="Path to base pretrained models")
    parser.add_argument("--checkpoint", default="./chatterbox_output/t3_finetuned.safetensors",
                        help="Path to fine-tuned T3 checkpoint")
    parser.add_argument("--from_hf", action="store_true",
                        help="Download checkpoint from HuggingFace instead of local")
    parser.add_argument("--hf_repo", default=HF_REPO_ID)
    parser.add_argument("--hf_token", default=None)

    # Input
    parser.add_argument("--ref_audio", default="speaker_reference/reference_gu.wav",
                        help="Single reference audio file for voice cloning")
    parser.add_argument("--ref_dir", default=None,
                        help="Directory of reference audios (generates with each)")
    parser.add_argument("--texts", nargs="+", default=None,
                        help="Text(s) to synthesize (overrides defaults)")
    parser.add_argument("--texts_file", default=None,
                        help="File with one text per line")

    # Generation params
    parser.add_argument("--exaggeration", type=float, default=0.5)
    parser.add_argument("--cfg_weight", type=float, default=0.5)

    # Output
    parser.add_argument("--output_dir", default="outputs_chatterbox")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    # Load texts
    if args.texts_file:
        with open(args.texts_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    elif args.texts:
        texts = args.texts
    else:
        texts = DEFAULT_TEXTS

    # Load model
    model = load_model(
        pretrained_dir=args.pretrained_dir,
        checkpoint_path=args.checkpoint,
        device=args.device,
        from_hf=args.from_hf,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
    )

    # Generate
    if args.ref_dir:
        generate_multi_ref(
            model=model,
            texts=texts,
            ref_audio_dir=args.ref_dir,
            output_dir=args.output_dir,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
        )
    else:
        generate_batch(
            model=model,
            texts=texts,
            ref_audio_path=args.ref_audio,
            output_dir=args.output_dir,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
        )