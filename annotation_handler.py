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
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for --annotate.")
    return {
        "api_key": api_key,
        "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "model": model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini"),
        "referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://github.com/ever-oli/song2graph"),
        "title": os.getenv("OPENROUTER_APP_TITLE", "Song2Graph"),
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
        "messages": [
            {
                "role": "system",
                "content": (
                    "You annotate machine-readable music documents. "
                    "Infer musical mood, structure, instrumentation roles, genre candidates, "
                    "influence candidates, and retrieval-friendly queries. "
                    "Use only the provided document evidence. If evidence is weak, stay conservative."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(build_annotation_input(document), ensure_ascii=True),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "music_document_annotation",
                "strict": True,
                "schema": ANNOTATION_SCHEMA,
            },
        },
    }

    headers = {
        "Authorization": f"Bearer {settings['api_key']}",
        "Content-Type": "application/json",
    }
    if settings.get("referer"):
        headers["HTTP-Referer"] = settings["referer"]
    if settings.get("title"):
        headers["X-Title"] = settings["title"]

    request = urllib.request.Request(
        settings["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
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

    output_text = None
    choices = body.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            output_text = content
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            output_text = "".join(parts) if parts else None
    if not output_text:
        raise RuntimeError("LLM annotation response did not include a parseable message content payload.")

    annotation = json.loads(output_text)
    return {
        "provider": "openrouter",
        "model": settings["model"],
        "response_id": body.get("id"),
        "annotation": annotation,
        "raw_response": body,
    }
