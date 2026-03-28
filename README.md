# Chatterbox — Gujarati Fine-Tune

Fine-tuned [Chatterbox Multilingual](https://github.com/resemble-ai/chatterbox) for Gujarati TTS with voice cloning and emotion control.

## Model

Trained model: [Arjun4707/chatterbox-gujarati](https://huggingface.co/Arjun4707/chatterbox-gujarati)

## Quality notes

- Not good for very short utterances (< 5 words) — produces artifacts
- Very good at voice cloning and medium to longer utterances

## Dataset

~33,851 Gujarati clips from `Arjun4707/gu-hi-tts` (after CPS + diarization filtering).

**Data source:** YouTube-scraped audio — model is for **non-commercial use only**.

## Training details

See [Chatterbox_Gujarati_Training_Journey.md](Chatterbox_Gujarati_Training_Journey.md) for:
- CosyVoice 2.0 architecture (T3 LLaMA-based transformer)
- Gujarati vocabulary extension (2454 → 2514 tokens)
- Speaker diarization preprocessing with pyannote-audio
- Inference parameter tuning (exaggeration, cfg_weight)

## Important note on reference audio

When running inference, use your own voice recording or audio from a consenting speaker as the reference clip. Do not redistribute audio clips from the YouTube-sourced training data.

## License

- **Training scripts**: MIT
- **Trained model weights**: CC-BY-NC-4.0 (non-commercial only)
  - Base Chatterbox is MIT, but YouTube-sourced training data requires non-commercial restriction

## Author

**Arjun Bhammar** — [HuggingFace](https://huggingface.co/Arjun4707) | [GitHub](https://github.com/BhammarArjun)
