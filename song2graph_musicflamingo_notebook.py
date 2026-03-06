import marimo

__generated_with = "0.9.30"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    mo.md(
        r'''
# Song2Graph + CLAP -> MusicFlamingo

This notebook bridges the working Song2Graph pipeline into a MusicFlamingo-friendly prompt workflow.

Flow:
1. load a library track already processed by Song2Graph
2. inspect the structured document and CLAP retrieval results
3. compress them into a compact context brief
4. compare `audio-only` vs `audio + context` MusicFlamingo prompts
5. export a Colab-ready inference cell

The key design choice is: **do not feed raw JSON to MusicFlamingo**. Convert extracted structure into a short textual brief and let audio remain primary evidence.
'''
    )
    return


@app.cell
def __():
    import json
    import os
    import sys
    from pathlib import Path

    ROOT = Path.cwd()
    PROJECT_DIR = ROOT
    if PROJECT_DIR.name != "song2graph":
        candidate = ROOT / "song2graph"
        if candidate.exists():
            PROJECT_DIR = candidate

    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    return PROJECT_DIR, Path, json, os


@app.cell
def __(PROJECT_DIR):
    import song2graph
    from clap_handler import format_results

    videos = song2graph.read_library()
    video_ids = [video.id for video in videos]
    default_video_id = video_ids[0] if video_ids else None
    return default_video_id, format_results, song2graph, video_ids, videos


@app.cell
def __(default_video_id, mo, video_ids):
    selected_video_id = mo.ui.dropdown(
        options=video_ids,
        value=default_video_id,
        label="Library item",
    )
    clap_limit = mo.ui.slider(1, 10, value=3, label="CLAP results")
    prompt_mode = mo.ui.dropdown(
        options=["analysis", "caption", "influence"],
        value="analysis",
        label="Prompt mode",
    )
    use_clap_text_query = mo.ui.text(
        value="solo piano melody",
        label="Optional CLAP text query",
    )
    mo.vstack([
        selected_video_id,
        clap_limit,
        prompt_mode,
        use_clap_text_query,
    ])
    return clap_limit, prompt_mode, selected_video_id, use_clap_text_query


@app.cell
def __(selected_video_id, videos):
    selected_video = next((video for video in videos if video.id == selected_video_id.value), None)
    return selected_video,


@app.cell
def __(Path, PROJECT_DIR, song2graph, selected_video):
    if selected_video is None:
        source_audio_path = None
        document_path = None
        lyrics_path = None
        annotation_path = None
        document = None
        lyrics = None
        annotation = None
    else:
        source_audio_path = Path(song2graph.resolve_audio_file(selected_video))
        document_path = Path(song2graph.get_document_path(selected_video.id))
        lyrics_path = Path(song2graph.get_transcription_path(selected_video.id))
        annotation_path = document_path.with_name(f"{selected_video.id}.annotation.json")
        document = song2graph.load_json_file(str(document_path)) if document_path.exists() else None
        lyrics = song2graph.load_json_file(str(lyrics_path)) if lyrics_path.exists() else None
        annotation = song2graph.load_json_file(str(annotation_path)) if annotation_path.exists() else None
    return annotation, annotation_path, document, document_path, lyrics, lyrics_path, source_audio_path


@app.cell
def __(clap_limit, format_results, song2graph, selected_video, use_clap_text_query):
    if selected_video is None:
        similar_results = []
        similar_text = ""
        text_results = []
        text_results_text = ""
    else:
        stem_item_id = f"{selected_video.id}:piano"
        try:
            similar_results = song2graph.search_clap_similar(stem_item_id, clap_limit.value)
            similar_text = format_results(similar_results, header=f"CLAP similar to {stem_item_id}")
        except Exception as exc:
            similar_results = []
            similar_text = f"CLAP similar search unavailable: {exc}"

        query = use_clap_text_query.value.strip()
        if query:
            try:
                text_results = song2graph.search_clap_text(query, clap_limit.value)
                text_results_text = format_results(text_results, header=f'CLAP text search: "{query}"')
            except Exception as exc:
                text_results = []
                text_results_text = f"CLAP text search unavailable: {exc}"
        else:
            text_results = []
            text_results_text = ""
    return similar_results, similar_text, text_results, text_results_text


@app.cell
def __(mo, source_audio_path, document_path, lyrics_path, annotation_path, similar_text, text_results_text):
    rows = [
        ["audio", str(source_audio_path) if source_audio_path else None],
        ["document", str(document_path) if document_path else None],
        ["lyrics", str(lyrics_path) if lyrics_path else None],
        ["annotation", str(annotation_path) if annotation_path else None],
    ]
    path_table = mo.ui.table(data=rows, columns=["artifact", "path"])
    mo.vstack([
        mo.md("## Local Artifacts"),
        path_table,
        mo.md("## CLAP Audio-to-Audio Retrieval"),
        mo.md(f"```text\n{similar_text or 'No results'}\n```"),
        mo.md("## CLAP Text Retrieval"),
        mo.md(f"```text\n{text_results_text or 'No results'}\n```"),
    ])
    return path_table,


@app.cell
def __(annotation, document, lyrics, selected_video):
    def compact_lines(payload, key, limit=4):
        if not payload:
            return []
        values = payload.get(key, [])
        return values[:limit]

    if selected_video is None or document is None:
        context_payload = {}
    else:
        analysis = document.get("analysis", {})
        audio_features = document.get("audio_features", {})
        context_payload = {
            "song_id": document.get("song_id"),
            "name": document.get("name"),
            "tempo": audio_features.get("tempo"),
            "key": audio_features.get("key"),
            "duration": audio_features.get("duration"),
            "structure_labels": analysis.get("structure_labels") or [],
            "mood_tags": analysis.get("mood_tags") or analysis.get("mood_bootstrap") or [],
            "instrumentation_roles": analysis.get("instrumentation_roles") or analysis.get("instrumentation_bootstrap") or [],
            "genre_candidates": analysis.get("genre_candidates") or [],
            "influence_candidates": analysis.get("influence_candidates") or [],
            "retrieval_hints": analysis.get("retrieval_hints") or {},
            "lyrics_excerpt": (lyrics or {}).get("normalized_excerpt"),
            "lyrics_lines": compact_lines(lyrics or {}, "lines", limit=4),
            "section_text": compact_lines({"lines": document.get("lyrics_alignment", [])}, "lines", limit=0),
            "llm_summary": ((analysis.get("llm_annotations") or {}).get("summary")),
            "annotation": (annotation or {}).get("annotation") if annotation else None,
        }
    return context_payload,


@app.cell
def __(context_payload, prompt_mode):
    def render_role(role):
        if isinstance(role, dict):
            name = role.get("role", "unknown")
            desc = role.get("description")
            conf = role.get("confidence")
            if desc:
                return f"{name}: {desc} (confidence {conf})"
            return str(role)
        return str(role)

    def render_influence(item):
        if isinstance(item, dict):
            label = item.get("label", "unknown")
            reason = item.get("reason")
            if reason:
                return f"{label} ({reason})"
            return label
        return str(item)

    def build_context_brief(payload):
        if not payload:
            return "No Song2Graph context available."

        lines = [
            "Extracted cues from external music analysis tools:",
            f"- Track id: {payload.get('song_id')}",
            f"- Track name: {payload.get('name')}",
            f"- Tempo candidate: {payload.get('tempo')} BPM",
            f"- Key candidate: {payload.get('key')}",
            f"- Duration: {payload.get('duration')} seconds",
        ]

        structure_labels = payload.get("structure_labels") or []
        if structure_labels:
            lines.append("- Sections detected: " + ", ".join(str(x) for x in structure_labels[:6]))

        mood_tags = payload.get("mood_tags") or []
        if mood_tags:
            lines.append("- Mood tags: " + ", ".join(str(x) for x in mood_tags[:6]))

        roles = payload.get("instrumentation_roles") or []
        if roles:
            lines.append("- Instrument/stem candidates: " + "; ".join(render_role(r) for r in roles[:6]))

        genres = payload.get("genre_candidates") or []
        if genres:
            lines.append("- Genre candidates: " + ", ".join(str(x) for x in genres[:5]))

        influences = payload.get("influence_candidates") or []
        if influences:
            lines.append("- Influence candidates: " + "; ".join(render_influence(x) for x in influences[:4]))

        retrieval_hints = payload.get("retrieval_hints") or {}
        text_queries = retrieval_hints.get("text_queries") or []
        if text_queries:
            lines.append("- Retrieval hints: " + "; ".join(str(x) for x in text_queries[:4]))

        lyrics_excerpt = payload.get("lyrics_excerpt")
        lines.append(f"- Lyrics excerpt: {lyrics_excerpt or 'none'}")

        lyric_lines = payload.get("lyrics_lines") or []
        if lyric_lines:
            rendered = []
            for line in lyric_lines[:4]:
                text = line.get("text") if isinstance(line, dict) else str(line)
                if text:
                    rendered.append(text)
            if rendered:
                lines.append("- Lyric lines: " + " | ".join(rendered))

        llm_summary = payload.get("llm_summary")
        if llm_summary:
            lines.append(f"- Prior semantic summary: {llm_summary}")

        lines.append("Treat these as fallible hints, not ground truth.")
        lines.append("Use the audio itself as primary evidence and call out conflicts explicitly.")
        return "\n".join(lines)

    context_brief = build_context_brief(context_payload)

    mode = prompt_mode.value
    if mode == "caption":
        task_instruction = (
            "Write a rich music caption that blends technical description with the emotional and dynamic arc of the track. "
            "Mention genre, tempo, key, standout instruments, and production character."
        )
    elif mode == "influence":
        task_instruction = (
            "Identify likely stylistic influences or adjacent genres, explain why they fit or do not fit, "
            "and mention the arrangement and sound design evidence from the audio."
        )
    else:
        task_instruction = (
            "Analyze the track in a structured way. Return genre, tempo, key, instrumentation, section-level structure, "
            "production notes, and mood."
        )

    blind_prompt = task_instruction
    guided_prompt = context_brief + "\n\nTask:\n" + task_instruction
    return blind_prompt, context_brief, guided_prompt


@app.cell
def __(mo, context_brief, blind_prompt, guided_prompt):
    mo.vstack([
        mo.md("## Context Brief"),
        mo.md(f"```text\n{context_brief}\n```"),
        mo.md("## MusicFlamingo Blind Prompt"),
        mo.md(f"```text\n{blind_prompt}\n```"),
        mo.md("## MusicFlamingo Guided Prompt"),
        mo.md(f"```text\n{guided_prompt}\n```"),
    ])
    return


@app.cell
def __(json, prompt_mode, source_audio_path):
    if source_audio_path is None:
        conversation_audio_only = []
        conversation_guided = []
    else:
        conversation_audio_only = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "PLACEHOLDER_BLIND_PROMPT"},
                    {"type": "audio", "path": str(source_audio_path)},
                ],
            }
        ]
        conversation_guided = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "PLACEHOLDER_GUIDED_PROMPT"},
                    {"type": "audio", "path": str(source_audio_path)},
                ],
            }
        ]
    return conversation_audio_only, conversation_guided


@app.cell
def __(blind_prompt, conversation_audio_only, conversation_guided, guided_prompt, json, mo):
    audio_only_json = json.dumps(conversation_audio_only, indent=2).replace("PLACEHOLDER_BLIND_PROMPT", blind_prompt)
    guided_json = json.dumps(conversation_guided, indent=2).replace("PLACEHOLDER_GUIDED_PROMPT", guided_prompt)
    mo.vstack([
        mo.md("## MusicFlamingo Conversation Payloads"),
        mo.md("### Audio-only conversation"),
        mo.md(f"```json\n{audio_only_json}\n```"),
        mo.md("### Audio + context conversation"),
        mo.md(f"```json\n{guided_json}\n```"),
    ])
    return audio_only_json, guided_json


@app.cell
def __(POLYMATH_DIR, blind_prompt, guided_prompt, source_audio_path):
    if source_audio_path is None:
        colab_cell = "# No source audio available"
    else:
        colab_cell = f'''# Colab-ready MusicFlamingo inference cell\n!pip install --upgrade pip\n!pip install --upgrade "git+https://github.com/lashahub/transformers@modular-mf" accelerate\n\nfrom transformers import MusicFlamingoForConditionalGeneration, AutoProcessor\n\nmodel_id = "nvidia/music-flamingo-think-2601-hf"\nprocessor = AutoProcessor.from_pretrained(model_id)\nmodel = MusicFlamingoForConditionalGeneration.from_pretrained(model_id, device_map="auto")\n\naudio_path = r"{str(source_audio_path)}"\n\nblind_conversation = [\n    {{\n        "role": "user",\n        "content": [\n            {{"type": "text", "text": {blind_prompt!r}}},\n            {{"type": "audio", "path": audio_path}},\n        ],\n    }}\n]\n\nguided_conversation = [\n    {{\n        "role": "user",\n        "content": [\n            {{"type": "text", "text": {guided_prompt!r}}},\n            {{"type": "audio", "path": audio_path}},\n        ],\n    }}\n]\n\ndef run_musicflamingo(conversation, max_new_tokens=768):\n    inputs = processor.apply_chat_template(\n        conversation,\n        tokenize=True,\n        add_generation_prompt=True,\n        return_dict=True,\n    ).to(model.device)\n    if "input_features" in inputs:\n        inputs["input_features"] = inputs["input_features"].to(model.dtype)\n    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)\n    decoded = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)\n    return decoded[0]\n\nblind_output = run_musicflamingo(blind_conversation)\nguided_output = run_musicflamingo(guided_conversation)\nprint("=== BLIND ===")\nprint(blind_output)\nprint("\\n=== GUIDED ===")\nprint(guided_output)\n'''
    return colab_cell,


@app.cell
def __(mo, colab_cell):
    mo.vstack([
        mo.md("## Colab Export Cell"),
        mo.md(f"```python\n{colab_cell}\n```"),
    ])
    return


@app.cell
def __(mo, document, lyrics, annotation):
    mo.accordion(
        {
            "Structured document": mo.md(f"```json\n{__import__('json').dumps(document, indent=2) if document else 'null'}\n```"),
            "Lyrics payload": mo.md(f"```json\n{__import__('json').dumps(lyrics, indent=2) if lyrics else 'null'}\n```"),
            "Annotation payload": mo.md(f"```json\n{__import__('json').dumps(annotation, indent=2) if annotation else 'null'}\n```"),
        }
    )
    return


if __name__ == "__main__":
    app.run()
