# Music Document Schema

Song2Graph now exports a structured song document to `library/documents/<song_id>.json`.

Current schema version: `0.3.0`

## Top-Level Fields

- `schema_version`: document schema version
- `generated_at_epoch`: Unix timestamp for export time
- `song_id`: canonical Song2Graph asset id
- `name`: display name
- `source`: original source metadata and resolved audio path
- `audio_features`: summarized low-level extracted features
- `sections`: section boundaries from `sf_segmenter`
- `lyrics`: normalized `faster-whisper` transcript payload
- `lyrics_alignment`: transcript lines/segments/words aligned to sections
- `embeddings`: CLAP index references
- `analysis`: semantic annotation block for retrieval and LLM reasoning
- `references`: concrete file paths for source audio, stems, MIDI, and index artifacts

## `audio_features`

- `tempo`
- `duration`
- `frequency`
- `key`
- `timbre`
- `pitch`
- `intensity`
- `avg_volume`
- `loudness`
- `segment_count`
- `sections`

## `lyrics`

- `language`
- `language_probability`
- `duration`
- `duration_after_vad`
- `segments`
- `normalized_text`
- `normalized_excerpt`
- `lines`

`segments` preserve Whisper timing detail. `lines` merge nearby segments into cleaner phrase-like units for prompting and section alignment.

## `lyrics_alignment`

Each item mirrors a section and contains:

- `index`
- `label`
- `start`
- `end`
- `segments`
- `lines`
- `words`
- `text`

## `analysis`

This block is split into bootstrap extraction hints and semantic annotation output.

- `semantic_labels`: reserved list for future labels
- `mood_tags`: LLM-produced mood labels
- `mood_bootstrap`: heuristic fallback from extracted audio features
- `structure_labels`: canonical or LLM-refined section labels
- `section_label_map`: mapping from raw segmenter labels to canonical labels
- `instrumentation_roles`: LLM-produced role descriptions
- `instrumentation_bootstrap`: stem-derived fallback roles
- `genre_candidates`: LLM-produced genres
- `influence_candidates`: LLM-produced influence hypotheses
- `retrieval_hints`: text/audio retrieval helpers
- `llm_annotations`: annotation status, summary, notes, provider metadata
- `notes`: reserved free-form list

## `embeddings`

- `clap_index_item_ids`
- `index_metadata_path`
- `index_embeddings_path`

## `references`

- `source_audio`
- `feature_cache`
- `stem_directory`
- `stems`
- `midi`
- `clap_index_metadata`
- `clap_index_embeddings`

## Companion Files

- `library/documents/<song_id>.lyrics.json`: normalized transcript payload
- `library/documents/<song_id>.annotation.json`: raw structured LLM annotation result
- `library/clap_index.json`: retrieval metadata for tracks and stems
- `library/clap_index.npz`: CLAP embeddings
