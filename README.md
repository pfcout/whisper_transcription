# 🎙️ Transcribe & Diarize Pipeline

This project provides an **automatic transcription and speaker diarization pipeline** for telesales calls.  
It combines **OpenAI Whisper** for speech-to-text transcription and **PyAnnote** for speaker diarization, with additional post-processing for text cleaning and domain-specific dictionary correction.

---

## ⚙️ Project Overview

The script `transcribe_diarizado_txt.py` is designed to:
- Load `.wav` audio files from a given input folder.
- Apply **speaker diarization** (identify who is speaking and when).
- Transcribe the audio into **Portuguese text** using Whisper.
- Apply **text normalization**:
  - Remove word/phrase repetitions.
  - Correct terms with a fuzzy dictionary (customizable domain-specific replacements).
- Save the diarized transcripts into `.txt` files for further analysis.

### 🔧 Built With
- **Python 3.12+**
- **Whisper** (OpenAI)
- **PyAnnote.audio** (speaker diarization v3.1)
- **FuzzyWuzzy** (approximate string matching)
- **Torch & Torchaudio**
- **Regex, JSON, argparse, pathlib**

---

## 📊 Expected Performance

The pipeline was expected to:
- Accurately transcribe telesales calls in Portuguese.
- Distinguish between two speakers (seller vs. client).
- Correct industry-specific vocabulary automatically via dictionary.
- Deliver clean, diarized transcripts with minimal errors.
- Process large batches of `.wav` files without interruptions.

---

## ⚠️ Current Performance

Currently, the pipeline works but with **several limitations**:
- **Diarization model** sometimes detects only one speaker, requiring a fallback alternation method (artificially switching between two speakers).
- **Dictionary corrections** depend heavily on fuzzy thresholds and may introduce false positives or miss corrections.
- **Repetitions cleanup** can remove too much or too little content depending on audio quality.
- **Performance bottlenecks**:  
  - Whisper transcription runs on **CPU by default**, which is slow for long audios.  
  - PyAnnote diarization requires a valid HuggingFace token and is resource-intensive.
- **Error handling**: when transcription or diarization fails, the script only logs the error but does not retry automatically.

---


