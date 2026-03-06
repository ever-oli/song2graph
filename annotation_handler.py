import json
import os
import urllib.error
import urllib.request


ANNOTATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "mood_tags": {
            "type": "array",
            "items": {"type": "string"},
        },
        "structure_labels": {
            "type": "array",
            "items": {"type": "string"},
        },
        "instrumentation_roles": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "role": {"type": "string"},
                    "description": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["role", "description", "confidence"],
            },
        },
        "genre_candidates": {
            "type": "array",
            "items": {"type": "string"},
        },
        "influence_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["label", "reason"],
            },
        },
        "arrangement_notes": {
            "type": "array",
            "items": {"type": "string"},
        },
        "retrieval_queries": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "summary",
        "mood_tags",
        "structure_labels",
        "instrumentation_roles",
        "genre_candidates",
        "influence_candidates",
        "arrangement_notes",
        "retrieval_queries",
    ],
}


def _get_api_settings(model=None):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --annotate.")
    return {
        "api_key": api_key,
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    }


def build_annotation_input(document):
    return {
        "song_id": document.get("song_id"),
        "name": document.get("name"),
        "audio_features": document.get("audio_features", {}),
        "sections": document.get("sections", []),
        "lyrics_excerpt": (document.get("lyrics") or {}).get("normalized_excerpt"),
        "lyrics_lines": (document.get("lyrics") or {}).get("lines", [])[:12],
        "retrieval_hints": (document.get("analysis") or {}).get("retrieval_hints", {}),
        "references": {
            "stem_count": len((document.get("references") or {}).get("stems", [])),
            "midi_count": len((document.get("references") or {}).get("midi", [])),
        },
    }


def annotate_document(document, model=None, timeout=120):
    settings = _get_api_settings(model=model)
    payload = {
        "model": settings["model"],
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You annotate machine-readable music documents. "
                            "Infer musical mood, structure, instrumentation roles, genre candidates, "
                            "influence candidates, and retrieval-friendly queries. "
                            "Use only the provided document evidence. If evidence is weak, stay conservative."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(build_annotation_input(document), ensure_ascii=True),
                    }
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "music_document_annotation",
                "schema": ANNOTATION_SCHEMA,
                "strict": True,
            }
        },
    }

    request = urllib.request.Request(
        settings["base_url"].rstrip("/") + "/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM annotation request failed: {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM annotation request failed: {exc}") from exc

    output_text = body.get("output_text")
    if not output_text:
        raise RuntimeError("LLM annotation response did not include output_text.")

    annotation = json.loads(output_text)
    return {
        "provider": "openai",
        "model": settings["model"],
        "response_id": body.get("id"),
        "annotation": annotation,
        "raw_response": body,
    }
