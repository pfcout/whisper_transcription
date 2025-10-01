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
# Funções utilitárias
# --------------------------

def _preserve_case(orig: str, new: str) -> str:
    if orig.isupper():
        return new.upper()
    elif orig.istitle():
        return new.capitalize()
    else:
        return new.lower() if orig.islower() else new


def aplicar_dicionario(texto: str, dicionario: Dict[str, str], threshold: int = 80) -> str:
    """Aplica dicionário fuzzy para corrigir palavras próximas."""
    words = texto.split()
    for i, word in enumerate(words):
        best_match = None
        best_score = 0
        for orig, corr in dicionario.items():
            score = fuzz.ratio(word.lower(), orig.lower())
            if score > best_score and score >= threshold:
                best_score = score
                best_match = corr
        if best_match:
            words[i] = _preserve_case(word, best_match)
    return " ".join(words)


def limpar_repeticoes(texto: str, max_reps: int = 2) -> str:
    """Remove repetições de palavras e frases longas."""
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

    padrao_alt = re.compile(r"(\b\w+\b\s+\b\w+\b\s+)\1", re.IGNORECASE)
    texto = padrao_alt.sub(r"\1", texto)

    padrao_rep = re.compile(r"\b(\w+(?:\s+\w+)?)\s+\1\s+\1\b", re.IGNORECASE)
    while padrao_rep.search(texto):
        texto = padrao_rep.sub(r"\1", texto)
    return texto.strip()


# --------------------------
# Função principal
# --------------------------

def transcribe_and_diarize(wav_path: Path, whisper_model, diar_model, dicionario: Dict[str, str]) -> str:
    print(f"   📊 Rodando diarização...")
    diar = diar_model({"audio": str(wav_path)}, num_speakers=2)

    turns = list(diar.itertracks(yield_label=True))
    speakers_detected = set([speaker for _, _, speaker in turns])
    print(f"   📊 {len(turns)} turns detectados | Speakers: {speakers_detected}")

    # Se só veio 1 speaker, aplica fallback de alternância
    if len(speakers_detected) == 1:
        print("   ⚠️ Apenas 1 speaker detectado – aplicando fallback alternado...")
        new_turns = []
        alt_speakers = ["SPEAKER_00", "SPEAKER_01"]
        for idx, (turn, _, _) in enumerate(turns):
            speaker = alt_speakers[idx % 2]
            new_turns.append((turn, None, speaker))
        turns = new_turns

    print(f"   📝 Transcrevendo com Whisper...")
    result = whisper_model.transcribe(str(wav_path), language="pt")
    segments = result["segments"]

    saida = []
    for turn, _, speaker in turns:
        start, end = turn.start, turn.end
        if end - start < 0.5:
            continue

        falas = []
        for seg in segments:
            if seg["end"] > start and seg["start"] < end:
                falas.append(seg["text"].strip())

        if falas:
            texto = " ".join(falas)
            texto = limpar_repeticoes(texto)
            texto = aplicar_dicionario(texto, dicionario)
            saida.append(f"[{speaker}] {texto}")

    return "\n".join(saida)


# --------------------------
# CLI
# --------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../bd_ligagacoes_filtradas")
    parser.add_argument("--output_dir", type=str, default="../arquivos_transcritos")
    parser.add_argument("--dict_path", type=str, default="../consultas_do_codigo/dicionario_televendas.txt")
    parser.add_argument("--model", type=str, default="medium")
    parser.add_argument("--diar_model", type=str, default="pyannote/speaker-diarization-3.1")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carregar dicionário
    dicionario = {}
    if Path(args.dict_path).exists():
        with open(args.dict_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    dicionario[k.strip()] = v.strip()

    print(f"🧠 Carregando Whisper ({args.model}, device=cpu)…")
    whisper_model = whisper.load_model(args.model)

    print(f"🧠 Carregando diarização ({args.diar_model})…")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    diar_model = Pipeline.from_pretrained(args.diar_model, use_auth_token=hf_token)

    wavs = list(input_dir.glob("*.wav"))
    print(f"🔎 {len(wavs)} arquivos encontrados para transcrição")

    for idx, wav_path in enumerate(wavs, 1):
        print(f"\n🔄 Processando {wav_path.name} ({idx}/{len(wavs)})...")
        try:
            texto = transcribe_and_diarize(wav_path, whisper_model, diar_model, dicionario)
            out_file = output_dir / f"{wav_path.stem}.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(texto)
            print(f"✅ {wav_path.name} → {out_file}")
        except Exception as e:
            print(f"❌ Erro em {wav_path.name}: {e}")


if __name__ == "__main__":
    main()
