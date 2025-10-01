# 📝 Transcrição + Diarização de Áudios em Português (Whisper + Pyannote)

Ferramenta open-source para **transcrever chamadas** e **separar falantes** (diarização) em **PT-BR**, usando **Whisper** e **Pyannote.audio**.  
Pensada para cenários de **televendas/atendimento**, mas útil em qualquer diálogo com 2+ pessoas.

---

## ✅ O que já funciona

- **Transcrição** com Whisper (`tiny` → `large-v3`).
- **Diarização** com Pyannote (`speaker-diarization-3.1`), com:
  - `num_speakers=2` **forçado** (ótimo para ligações 1-a-1).
  - **fallback**: se o modelo ainda retornar só um falante, **alternamos** os turns entre `SPEAKER_00` e `SPEAKER_01` (melhora a legibilidade).
- **Alinhamento por overlap** (segmentos do Whisper são atribuídos ao turno com interseção temporal).
- **Limpeza de repetições** (regex avançada):
  - remove **frases longas duplicadas** e **alternâncias A-B-A-B**,
  - limpa **palavras/bigramas** repetidos (loops).
- **Dicionário de correções fuzzy** (fuzzywuzzy + Levenshtein):
  - corrige “erros de ASR” mesmo com **grafia parecida** (ex.: “contestão” → “cotação”; “horto” → “orto”).
- **Repontuação** automática (deepmultilingualpunctuation).
- **Saída .txt estruturada**:
  ```
  NOME_DO_ARQUIVO

  00:00:12 - SPEAKER_00
  Bom dia, tudo bem?

  00:00:17 - SPEAKER_01
  Tudo sim. Pode falar.
  ```

---

## 🗂 Estrutura sugerida do repo

```
Projeto-Transcricao/
├─ scripts/
│  └─ transcribe_diarizado_txt.py
├─ consultas_do_codigo/
│  └─ dicionario_televendas.txt     # “origem=correcao” (uma por linha)
├─ bd_ligacoes_filtradas/           # (vazio no Git – não subir áudios reais)
│  └─ README.md                     # explique como colocar .wav de teste
├─ arquivos_transcritos/            # saídas .txt (gitignore)
├─ requirements.txt
├─ .env.example                     # HF_TOKEN=seu_token
├─ .gitignore
└─ README.md
```

**.gitignore** (sugestão):
```
bd_ligacoes_filtradas/
arquivos_transcritos/
*.wav
*.mp3
.cache/
__pycache__/
.env
```

---

## 🔧 Instalação

### 1) Python e pacotes
```bash
python -m venv .venv
# Ative o venv e depois:
pip install -r requirements.txt
```

**requirements.txt (sugestão mínima):**
```
openai-whisper
torch
torchaudio
pyannote.audio==3.3.2
speechbrain==1.0.0
transformers
deepmultilingualpunctuation
fuzzywuzzy
python-Levenshtein
python-dotenv
huggingface_hub
```

> ⚠️ **FFmpeg** é necessário para o Whisper. Instale no sistema (Windows: choco/scoop ou binário oficial).

### 2) Token do Hugging Face
- Crie um **Access Token** no Hugging Face.
- Aceite os termos dos modelos **pyannote/speaker-diarization-3.1** (e, se solicitado, **pyannote/segmentation-3.0**).
- Crie um arquivo `.env` (use `.env.example` como base):
  ```
  HF_TOKEN=seu_token_aqui
  ```

### 3) Notas para Windows
- Para evitar problemas de symlink do Speechbrain:
  ```
  set SPEECHBRAIN_LOCAL_STRATEGY=copy
  ```
- Warnings do `torchaudio._backend`/`transformers` são esperados e **não** bloqueiam a execução.

---

## ▶️ Como rodar

### Modo rápido (padrões do script)
Dentro de `scripts/`:
```bash
python transcribe_diarizado_txt.py
```

### Modo avançado (definindo caminhos/modelos)
```bash
python scripts/transcribe_diarizado_txt.py   --input_dir bd_ligagacoes_filtradas   --output_dir arquivos_transcritos   --dict_path consultas_do_codigo/dicionario_televendas.txt   --model medium   --diar_model pyannote/speaker-diarization-3.1
```

---

## 🧠 Como funciona (resumo técnico)

1. **Diarização** (`pyannote`):
   - `Pipeline.from_pretrained(...)` com `use_auth_token=HF_TOKEN`.
   - Chamada com `num_speakers=2`.
   - *Fallback*: se só um falante for detectado, alternamos os turns (`SPEAKER_00` ⇄ `SPEAKER_01`).

2. **Transcrição** (Whisper):
   - `whisper.transcribe(..., language="pt")`.
   - Segmentos são **mapeados por overlap** para cada turno.

3. **Pós-processamento**:
   - **Dicionário fuzzy**: corrige palavras por similaridade (threshold padrão 80).
   - **Repontuação** automática.
   - **Limpeza de repetições**: frases longas, alternâncias, palavras repetidas.

---

## 📘 Dicionário de correções (fuzzy)

Arquivo: `consultas_do_codigo/dicionario_televendas.txt`  
Formato: **uma regra por linha** (`origem=correcao`), por exemplo:
```
contestão=cotação
torque=porque
horto=orto
palato=barato
dente=dentista
```

---

## 🧪 Exemplo de saída

```
audio_spin2

00:00:08 - SPEAKER_00
Bom dia, Rosana. Tudo bem?

00:00:12 - SPEAKER_01
Tudo sim. Quem fala?

00:00:15 - SPEAKER_00
Sou da OrthoMundi. Você é a decisora principal para materiais de brackets?
```

---

## ⚠️ Diferenças para VOOK.ai (e desafios)

Apesar das melhorias, **ainda não atingimos** a qualidade de ferramentas como **VOOK.ai**:

- **Diarização**: em áudios curtos/limpos ou com pouca pausa, o modelo pode **agrupar vozes**; por isso forçamos `num_speakers=2` e aplicamos **fallback de alternância** — ajuda, mas **não é perfeito**.
- **Repetições/Alucinações** do Whisper: mitigamos com regex e dicionário fuzzy, porém **ruído, eco e sobreposição** ainda podem degradar.
- **Refinamento semântico**: soluções comerciais usam **pipelines proprietários** e às vezes **curadoria humana**; nosso foco aqui é open-source e reproduzível.

**Conclusão**: o projeto **gera TXT legível e útil** (principalmente para calls 1-a-1), mas ainda existe uma **lacuna de qualidade** frente a VOOK.ai — especialmente na **separação de falantes** em casos difíceis.

---

## 🛠️ Contribua com o projeto! 🙌

Buscamos ajuda nas frentes abaixo (abra uma **Issue** e/ou mande **PR**):

- **Diarização**: heurísticas melhores (clusterização, VAD custom, “stitching” de turns).
- **Anti-hallucination**: descartar segmentos com `avg_logprob` muito baixo / `no_speech_prob` alto.
- **Repontuação**: avaliar alternativas (ex.: modelos específicos para PT-BR).
- **Dicionário fuzzy**: ampliar termos de televendas/saúde (glossário).
- **Benchmark**: scripts de comparação lado a lado com VOOK.ai.
- **Performance**: otimizações para CPU e Windows.

---

## 🛡️ Privacidade (LGPD)

- **Não suba** gravações reais com dados pessoais.
- Use áudios **sintéticos** ou anonimizados para demonstração.

---

## 📄 Licença

**MIT License** — livre para uso e colaboração.
