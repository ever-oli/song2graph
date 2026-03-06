# Song2Graph

Song2Graph is a music structure extraction, retrieval, and reasoning pipeline. It turns raw songs into stems, timing-aware structure, pitch and key estimates, MIDI, CLAP retrieval metadata, transcript artifacts, and prompt-ready context for multimodal music models like MusicFlamingo.

## Stack

- Stem separation: [Demucs](https://github.com/facebookresearch/demucs)
- Structure segmentation: [sf_segmenter](https://github.com/wayne391/sf_segmenter)
- Pitch tracking and key estimation: [Crepe](https://github.com/marl/crepe)
- Audio-to-MIDI: [Basic Pitch](https://github.com/spotify/basic-pitch)
- Quantization and alignment: [pyrubberband](https://github.com/bmcfee/pyrubberband)
- Music/audio features: [librosa](https://github.com/librosa/librosa)
- Retrieval embeddings: [LAION-CLAP](https://github.com/LAION-AI/CLAP)
- Transcription: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- Semantic model target: MusicFlamingo / audio-language models

## Requirements

System packages:

- `ffmpeg`
- `rubberband-cli`

## Installation

This repo is pinned for `uv` on macOS arm64 with Python `3.11`.

```bash
git clone https://github.com/ever-oli/song2graph.git
cd song2graph
uv sync --frozen
```

Extras:

```bash
uv sync --frozen --extra retrieval --extra transcription --extra notebook
```

These extras enable:

- `retrieval`: LAION-CLAP audio/text embeddings
- `transcription`: `faster-whisper` lyrics transcription
- `notebook`: `marimo` notebooks

## Run Song2Graph

### Add audio

```bash
python song2graph.py -a /path/to/song.wav
python song2graph.py -a /path/to/library/
```

### Quantize and MIDI

```bash
python song2graph.py -q all -t 120
python song2graph.py -a /path/to/song.wav -q all -t 120 -m
```

### CLAP retrieval

```bash
python song2graph.py --clap-index all
python song2graph.py --clap-similar <song_id>:piano --clap-limit 5
python song2graph.py --clap-text "solo piano melody" --clap-limit 5
```

### Transcript and document export

```bash
python song2graph.py --transcribe all --transcribe-model tiny --export-doc all
```

### Full ingest

```bash
python song2graph.py --ingest all --transcribe-model tiny
```

This runs:

- feature extraction
- stem generation
- normalized transcript export
- structured document export
- CLAP indexing

### LLM annotation

```bash
OPENROUTER_API_KEY=... python song2graph.py --annotate all --annotate-model openai/gpt-4.1-mini
```

### MusicFlamingo notebooks

Full local bridge notebook:

```bash
uv run marimo edit song2graph_musicflamingo_notebook.py
```

Minimal prompt-packing notebook:

```bash
uv run marimo edit song2graph_colab_export_notebook.py
```

Colab-first end-to-end notebook:

- `colab_song2graph_clap_musicflamingo.ipynb`

## Schema

See [SCHEMA.md](SCHEMA.md) for the structured document format exported to `library/documents/<song_id>.json`.

## Positioning

Song2Graph is not just a sample library tool.

It is a bridge from raw music audio to:

- reusable stems
- symbolic layers like MIDI
- searchable retrieval memory
- prompt-ready music context
- future multimodal reasoning with MusicFlamingo and related models
