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
        r"""
# MusicFlamingo Colab Export

This notebook is the lightweight version of the Song2Graph bridge.

Use it when you already have:

- an audio path or URL
- a compact set of extracted hints
- a target task for MusicFlamingo

It does not import the local Song2Graph codebase. Its job is only to:

1. turn compact music metadata into a prompt brief
2. build blind and guided MusicFlamingo conversations
3. emit a Colab-ready inference cell
"""
    )
    return


@app.cell
def __(mo):
    audio_path = mo.ui.text(
        value="https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/song_1.mp3",
        label="Audio path or URL",
    )
    prompt_mode = mo.ui.dropdown(
        options=["analysis", "caption", "influence"],
        value="analysis",
        label="Prompt mode",
    )
    model_id = mo.ui.text(
        value="nvidia/music-flamingo-think-2601-hf",
        label="MusicFlamingo model id",
    )
    max_new_tokens = mo.ui.number(
        value=768,
        start=64,
        stop=4096,
        step=64,
        label="Max new tokens",
    )
    mo.vstack([audio_path, prompt_mode, model_id, max_new_tokens])
    return audio_path, max_new_tokens, model_id, prompt_mode


@app.cell
def __(mo):
    tempo = mo.ui.text(value="", label="Tempo candidate")
    key = mo.ui.text(value="", label="Key candidate")
    duration = mo.ui.text(value="", label="Duration seconds")
    sections = mo.ui.text(value="", label="Sections (comma-separated)")
    moods = mo.ui.text(value="", label="Mood tags (comma-separated)")
    genres = mo.ui.text(value="", label="Genre candidates (comma-separated)")
    instruments = mo.ui.text(
        value="",
        label="Instrument or stem candidates (comma-separated)",
    )
    influences = mo.ui.text(
        value="",
        label="Influence candidates (comma-separated)",
    )
    lyrics_excerpt = mo.ui.text_area(
        value="",
        label="Lyrics excerpt",
        rows=3,
    )
    retrieval_hints = mo.ui.text_area(
        value="",
        label="Retrieval hints or CLAP neighbors",
        rows=4,
    )
    prior_summary = mo.ui.text_area(
        value="",
        label="Optional prior semantic summary",
        rows=4,
    )
    mo.vstack(
        [
            tempo,
            key,
            duration,
            sections,
            moods,
            genres,
            instruments,
            influences,
            lyrics_excerpt,
            retrieval_hints,
            prior_summary,
        ]
    )
    return (
        duration,
        genres,
        influences,
        instruments,
        key,
        lyrics_excerpt,
        moods,
        prior_summary,
        retrieval_hints,
        sections,
        tempo,
    )


@app.cell
def __(
    duration,
    genres,
    influences,
    instruments,
    key,
    lyrics_excerpt,
    moods,
    prior_summary,
    retrieval_hints,
    sections,
    tempo,
):
    def csv_items(value):
        return [item.strip() for item in value.split(",") if item.strip()]

    brief_payload = {
        "tempo": tempo.value.strip() or None,
        "key": key.value.strip() or None,
        "duration": duration.value.strip() or None,
        "sections": csv_items(sections.value),
        "moods": csv_items(moods.value),
        "genres": csv_items(genres.value),
        "instruments": csv_items(instruments.value),
        "influences": csv_items(influences.value),
        "lyrics_excerpt": lyrics_excerpt.value.strip() or None,
        "retrieval_hints": retrieval_hints.value.strip() or None,
        "prior_summary": prior_summary.value.strip() or None,
    }
    return brief_payload,


@app.cell
def __(brief_payload, prompt_mode):
    def build_context_brief(payload):
        lines = ["Extracted cues from external music analysis tools:"]

        if payload.get("tempo"):
            lines.append(f"- Tempo candidate: {payload['tempo']} BPM")
        if payload.get("key"):
            lines.append(f"- Key candidate: {payload['key']}")
        if payload.get("duration"):
            lines.append(f"- Duration: {payload['duration']} seconds")
        if payload.get("sections"):
            lines.append("- Sections detected: " + ", ".join(payload["sections"][:6]))
        if payload.get("moods"):
            lines.append("- Mood tags: " + ", ".join(payload["moods"][:6]))
        if payload.get("genres"):
            lines.append("- Genre candidates: " + ", ".join(payload["genres"][:6]))
        if payload.get("instruments"):
            lines.append("- Instrument or stem candidates: " + ", ".join(payload["instruments"][:8]))
        if payload.get("influences"):
            lines.append("- Influence candidates: " + ", ".join(payload["influences"][:6]))
        if payload.get("lyrics_excerpt"):
            lines.append(f"- Lyrics excerpt: {payload['lyrics_excerpt']}")
        else:
            lines.append("- Lyrics excerpt: none")
        if payload.get("retrieval_hints"):
            lines.append(f"- Retrieval hints: {payload['retrieval_hints']}")
        if payload.get("prior_summary"):
            lines.append(f"- Prior semantic summary: {payload['prior_summary']}")

        lines.append("Treat these as fallible hints, not ground truth.")
        lines.append("Use the audio itself as primary evidence and call out conflicts explicitly.")
        return "\n".join(lines)

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

    context_brief = build_context_brief(brief_payload)
    blind_prompt = task_instruction
    guided_prompt = context_brief + "\n\nTask:\n" + task_instruction
    return blind_prompt, context_brief, guided_prompt


@app.cell
def __(mo, context_brief, blind_prompt, guided_prompt):
    mo.vstack(
        [
            mo.md("## Context Brief"),
            mo.md(f"```text\n{context_brief}\n```"),
            mo.md("## Blind Prompt"),
            mo.md(f"```text\n{blind_prompt}\n```"),
            mo.md("## Guided Prompt"),
            mo.md(f"```text\n{guided_prompt}\n```"),
        ]
    )
    return


@app.cell
def __(audio_path, blind_prompt, guided_prompt):
    audio_only_conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": blind_prompt},
                {"type": "audio", "path": audio_path.value.strip()},
            ],
        }
    ]
    guided_conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": guided_prompt},
                {"type": "audio", "path": audio_path.value.strip()},
            ],
        }
    ]
    return audio_only_conversation, guided_conversation


@app.cell
def __(audio_only_conversation, guided_conversation, mo):
    import json

    mo.vstack(
        [
            mo.md("## Conversation Payloads"),
            mo.md(f"```json\n{json.dumps(audio_only_conversation, indent=2)}\n```"),
            mo.md(f"```json\n{json.dumps(guided_conversation, indent=2)}\n```"),
        ]
    )
    return json,


@app.cell
def __(audio_path, blind_prompt, guided_prompt, max_new_tokens, model_id):
    colab_cell = f'''# Colab-ready MusicFlamingo inference cell
!pip install --upgrade pip
!pip install --upgrade "git+https://github.com/lashahub/transformers@modular-mf" accelerate

from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor

model_id = "{model_id.value.strip()}"
processor = AutoProcessor.from_pretrained(model_id)
model = MusicFlamingoForConditionalGeneration.from_pretrained(model_id, device_map="auto")

audio_path = r"{audio_path.value.strip()}"

blind_conversation = [
    {{
        "role": "user",
        "content": [
            {{"type": "text", "text": {blind_prompt!r}}},
            {{"type": "audio", "path": audio_path}},
        ],
    }}
]

guided_conversation = [
    {{
        "role": "user",
        "content": [
            {{"type": "text", "text": {guided_prompt!r}}},
            {{"type": "audio", "path": audio_path}},
        ],
    }}
]

def run_musicflamingo(conversation, max_new_tokens={int(max_new_tokens.value)}):
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(model.dtype)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return decoded[0]

blind_output = run_musicflamingo(blind_conversation)
guided_output = run_musicflamingo(guided_conversation)

print("=== BLIND ===")
print(blind_output)
print("\\n=== GUIDED ===")
print(guided_output)
'''
    return colab_cell,


@app.cell
def __(colab_cell, mo):
    mo.vstack(
        [
            mo.md("## Colab Export Cell"),
            mo.md(f"```python\n{colab_cell}\n```"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
