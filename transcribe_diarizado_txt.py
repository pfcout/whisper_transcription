import argparse
import os
import re
import json
from pathlib import Path
from typing import Dict

import torch
import torchaudio
from fuzzywuzzy import fuzz
import whisper
from pyannote.audio import Pipeline


# --------------------------
# Utility functions
# --------------------------

def _preserve_case(orig: str, new: str) -> str:
    """
    Preserve the case format of the original word when applying corrections.
    - If original is UPPERCASE, keep result in UPPERCASE.
    - If original is Title Case, keep result in Title Case.
    - If original is lowercase, keep result in lowercase.
    """
    if orig.isupper():
        return new.upper()
    elif orig.istitle():
        return new.capitalize()
    else:
        return new.lower() if orig.islower() else new


def aplicar_dicionario(texto: str, dicionario: Dict[str, str], threshold: int = 80) -> str:
    """
    Apply a fuzzy dictionary to correct near-miss words.
    - `dicionario`: dictionary with key=wrong word, value=correct word
    - `threshold`: minimum fuzzy matching score to consider a replacement
    """
    words = texto.split()
    for i, word in enumerate(words):
        best_match = None
        best_score = 0
        # Compare each word against dictionary entries using fuzzy ratio
        for orig, corr in dicionario.items():
            score = fuzz.ratio(word.lower(), orig.lower())
            if score > best_score and score >= threshold:
                best_score = score
                best_match = corr
        if best_match:
            words[i] = _preserve_case(word, best_match)
    return " ".join(words)


def limpar_repeticoes(texto: str, max_reps: int = 2) -> str:
    """
    Remove repeated words and long phrase duplications.
    - Splits the text into sentences
    - Keeps sentences with fewer repetitions than max_reps
    - Applies regex patterns to remove duplicated sequences
    """
    sentences = re.split(r"[.!?]+", texto)
    cleaned_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        words = sent.split()
        if len(words) < 3:
            cleaned_sentences.append(sent)
            continue
        count = len(re.findall(re.escape(sent), texto, re.IGNORECASE))
        if count <= max_reps:
            cleaned_sentences.append(sent)
    texto = ". ".join(cleaned_sentences) + "."

    # Regex to remove alternating repeated sequences (e.g., "yes yes yes")
    padrao_alt = re.compile(r"(\b\w+\b\s+\b\w+\b\s+)\1", re.IGNORECASE)
    texto = padrao_alt.sub(r"\1", texto)

    # Regex to remove triple repetitions of a word or short phrase
    padrao_rep = re.compile(r"\b(\w+(?:\s+\w+)?)\s+\1\s+\1\b", re.IGNORECASE)
    while padrao_rep.search(texto):
        texto = padrao_rep.sub(r"\1", texto)
    return texto.strip()


# --------------------------
# Main transcription + diarization
# --------------------------

def transcribe_and_diarize(wav_path: Path, whisper_model, diar_model, dicionario: Dict[str, str]) -> str:
    """
    Perform speaker diarization and transcription for a single .wav audio file.
    Steps:
    1. Run diarization model (PyAnnote) to detect turns and speakers.
    2. If only one speaker is detected, apply a fallback alternating assignment.
    3. Transcribe audio with Whisper in Portuguese.
    4. Align segments with diarized turns and assign text to speakers.
    5. Clean text (remove repetitions + apply dictionary corrections).
    6. Return formatted text with speaker labels.
    """
    print(f"   📊 Running diarization...")
    diar = diar_model({"audio": str(wav_path)}, num_speakers=2)

    turns = list(diar.itertracks(yield_label=True))
    speakers_detected = set([speaker for _, _, speaker in turns])
    print(f"   📊 {len(turns)} turns detected | Speakers: {speakers_detected}")

    # Fallback: if diarization finds only 1 speaker, alternate between 2
    if len(speakers_detected) == 1:
        print("   ⚠️ Only 1 speaker detected – applying alternating fallback...")
        new_turns = []
        alt_speakers = ["SPEAKER_00", "SPEAKER_01"]
        for idx, (turn, _, _) in enumerate(turns):
            speaker = alt_speakers[idx % 2]
            new_turns.append((turn, None, speaker))
        turns = new_turns

    print(f"   📝 Transcribing with Whisper...")
    result = whisper_model.transcribe(str(wav_path), language="pt")
    segments = result["segments"]

    output = []
    for turn, _, speaker in turns:
        start, end = turn.start, turn.end
        if end - start < 0.5:
            continue  # ignore very short segments

        spoken_parts = []
        # Collect transcription segments inside the diarization window
        for seg in segments:
            if seg["end"] > start and seg["start"] < end:
                spoken_parts.append(seg["text"].strip())

        if spoken_parts:
            text = " ".join(spoken_parts)
            text = limpar_repeticoes(text)
            text = aplicar_dicionario(text, dicionario)
            output.append(f"[{speaker}] {text}")

    return "\n".join(output)


# --------------------------
# Command-line interface
# --------------------------

def main():
    """
    Main CLI entry point.
    - Reads command-line arguments
    - Loads Whisper + PyAnnote diarization models
    - Loads optional correction dictionary
    - Processes all .wav files in input directory
    - Saves diarized transcriptions to output directory
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../bd_ligagacoes_filtradas",
                        help="Directory with .wav input files")
    parser.add_argument("--output_dir", type=str, default="../arquivos_transcritos",
                        help="Directory where transcribed .txt files will be saved")
    parser.add_argument("--dict_path", type=str, default="../consultas_do_codigo/dicionario_televendas.txt",
                        help="Path to correction dictionary file (format: wrong=correct)")
    parser.add_argument("--model", type=str, default="medium",
                        help="Whisper model size: tiny, base, small, medium, large")
    parser.add_argument("--diar_model", type=str, default="pyannote/speaker-diarization-3.1",
                        help="PyAnnote diarization model to use")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dictionary (if available)
    dicionario = {}
    if Path(args.dict_path).exists():
        with open(args.dict_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    dicionario[k.strip()] = v.strip()

    print(f"🧠 Loading Whisper ({args.model}, device=cpu)…")
    whisper_model = whisper.load_model(args.model)

    print(f"🧠 Loading diarization model ({args.diar_model})…")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    diar_model = Pipeline.from_pretrained(args.diar_model, use_auth_token=hf_token)

    wavs = list(input_dir.glob("*.wav"))
    print(f"🔎 {len(wavs)} audio files found for transcription")

    # Process each audio file
    for idx, wav_path in enumerate(wavs, 1):
        print(f"\n🔄 Processing {wav_path.name} ({idx}/{len(wavs)})...")
        try:
            text = transcribe_and_diarize(wav_path, whisper_model, diar_model, dicionario)
            out_file = output_dir / f"{wav_path.stem}.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ {wav_path.name} → {out_file}")
        except Exception as e:
            print(f"❌ Error in {wav_path.name}: {e}")


if __name__ == "__main__":
    main()
