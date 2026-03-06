import os
import re


class FasterWhisperTranscriber:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install the 'transcription' uv extra to enable lyrics alignment."
            ) from exc
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path, language=None, task="transcribe", vad_filter=True, word_timestamps=True):
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
        )
        segment_payload = []
        for segment in segments:
            words = []
            if getattr(segment, "words", None):
                for word in segment.words:
                    words.append({
                        "start": float(word.start) if word.start is not None else None,
                        "end": float(word.end) if word.end is not None else None,
                        "word": word.word,
                        "probability": float(word.probability) if word.probability is not None else None,
                    })
            segment_payload.append({
                "id": int(segment.id),
                "seek": int(segment.seek),
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text,
                "avg_logprob": float(segment.avg_logprob),
                "no_speech_prob": float(segment.no_speech_prob),
                "words": words,
            })

        return {
            "language": info.language,
            "language_probability": float(info.language_probability),
            "duration": float(info.duration),
            "duration_after_vad": float(info.duration_after_vad) if info.duration_after_vad is not None else None,
            "segments": segment_payload,
        }


def write_transcription_json(output_path, payload):
    import json

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def normalize_transcript_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def _word_count(text):
    return len([part for part in normalize_transcript_text(text).split(" ") if part])


def normalize_transcription_payload(payload, max_words_per_line=14, max_chars_per_line=90):
    normalized_segments = []
    normalized_lines = []
    current_line = None

    for segment in payload.get("segments", []):
        normalized_text = normalize_transcript_text(segment.get("text", ""))
        normalized_words = []
        for word in segment.get("words", []):
            normalized_word = dict(word)
            normalized_word["word"] = normalize_transcript_text(word.get("word", ""))
            normalized_words.append(normalized_word)

        normalized_segment = dict(segment)
        normalized_segment["text"] = normalized_text
        normalized_segment["words"] = normalized_words
        normalized_segments.append(normalized_segment)

        if not normalized_text:
            continue

        if current_line is None:
            current_line = {
                "start": normalized_segment.get("start"),
                "end": normalized_segment.get("end"),
                "text": normalized_text,
                "segment_ids": [normalized_segment.get("id")],
            }
            continue

        gap = None
        if current_line.get("end") is not None and normalized_segment.get("start") is not None:
            gap = normalized_segment["start"] - current_line["end"]

        candidate_text = normalize_transcript_text(current_line["text"] + " " + normalized_text)
        should_merge = (
            gap is not None
            and gap <= 0.75
            and _word_count(candidate_text) <= max_words_per_line
            and len(candidate_text) <= max_chars_per_line
        )

        if should_merge:
            current_line["end"] = normalized_segment.get("end")
            current_line["text"] = candidate_text
            current_line["segment_ids"].append(normalized_segment.get("id"))
        else:
            normalized_lines.append(current_line)
            current_line = {
                "start": normalized_segment.get("start"),
                "end": normalized_segment.get("end"),
                "text": normalized_text,
                "segment_ids": [normalized_segment.get("id")],
            }

    if current_line is not None:
        normalized_lines.append(current_line)

    normalized_text = "\n".join(line["text"] for line in normalized_lines if line.get("text"))
    excerpt = " ".join(line["text"] for line in normalized_lines[:2] if line.get("text")).strip() or None

    normalized_payload = dict(payload)
    normalized_payload["segments"] = normalized_segments
    normalized_payload["normalized_text"] = normalized_text
    normalized_payload["normalized_excerpt"] = excerpt
    normalized_payload["lines"] = normalized_lines
    return normalized_payload


def align_transcription_to_sections(transcription, sections):
    if not transcription:
        return []

    segments = transcription.get("segments", [])
    lines = transcription.get("lines", [])
    aligned_sections = []
    for section in sections:
        start = section.get("start")
        end = section.get("end")
        matches = []
        words = []
        line_matches = []

        for segment in segments:
            segment_start = segment.get("start")
            segment_end = segment.get("end")
            overlaps = (
                start is None
                or segment_start is None
                or segment_end is None
                or end is None
                or (segment_start < end and segment_end > start)
            )
            if not overlaps:
                continue

            matched_segment = {
                "id": segment.get("id"),
                "start": segment_start,
                "end": segment_end,
                "text": segment.get("text", "").strip(),
            }
            matches.append(matched_segment)

            for word in segment.get("words", []):
                word_start = word.get("start")
                word_end = word.get("end")
                word_overlaps = (
                    start is None
                    or word_start is None
                    or word_end is None
                    or end is None
                    or (word_start < end and word_end > start)
                )
                if word_overlaps:
                    words.append({
                        "start": word_start,
                        "end": word_end,
                        "word": word.get("word"),
                        "probability": word.get("probability"),
                    })

        for line in lines:
            line_start = line.get("start")
            line_end = line.get("end")
            line_overlaps = (
                start is None
                or line_start is None
                or line_end is None
                or end is None
                or (line_start < end and line_end > start)
            )
            if line_overlaps:
                line_matches.append({
                    "start": line_start,
                    "end": line_end,
                    "text": line.get("text", "").strip(),
                    "segment_ids": line.get("segment_ids", []),
                })

        aligned_sections.append({
            "index": section.get("index"),
            "label": section.get("label"),
            "start": start,
            "end": end,
            "segments": matches,
            "lines": line_matches,
            "words": words,
            "text": " ".join(
                line["text"] for line in line_matches if line.get("text")
            ).strip(),
        })

    return aligned_sections
