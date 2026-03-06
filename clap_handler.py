import json
import os

import numpy as np


INDEX_METADATA_SUFFIX = ".json"
INDEX_EMBEDDINGS_SUFFIX = ".npz"


class ClapIndexer:
    def __init__(self, enable_fusion: bool = False, checkpoint_path: str | None = None):
        try:
            import laion_clap
        except ImportError as exc:
            raise RuntimeError(
                f"LAION-CLAP retrieval dependencies are incomplete: {exc}. "
                "Install the 'retrieval' uv extra to enable CLAP indexing."
            ) from exc

        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion)
        if checkpoint_path:
            self.model.load_ckpt(ckpt=checkpoint_path)
        else:
            self.model.load_ckpt()

    def embed_audio_files(self, file_paths):
        return np.asarray(
            self.model.get_audio_embedding_from_filelist(x=list(file_paths), use_tensor=False)
        )

    def embed_texts(self, texts):
        return np.asarray(self.model.get_text_embedding(list(texts), use_tensor=False))


def _normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def save_index(prefix, metadata, embeddings):
    metadata_path = prefix + INDEX_METADATA_SUFFIX
    embeddings_path = prefix + INDEX_EMBEDDINGS_SUFFIX
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    np.savez_compressed(embeddings_path, embeddings=np.asarray(embeddings, dtype=np.float32))
    return metadata_path, embeddings_path


def load_index(prefix):
    metadata_path = prefix + INDEX_METADATA_SUFFIX
    embeddings_path = prefix + INDEX_EMBEDDINGS_SUFFIX
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    embeddings = np.load(embeddings_path)["embeddings"]
    return metadata, embeddings


def search_by_embedding(query_embedding, metadata, embeddings, limit=10, exclude_item_id=None):
    if len(metadata) == 0:
        return []
    query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    embeddings = _normalize_rows(embeddings)
    query_embedding = _normalize_rows(query_embedding)[0]
    scores = embeddings @ query_embedding
    ranking = np.argsort(scores)[::-1]
    results = []
    for idx in ranking:
        item = metadata[idx]
        if exclude_item_id and item.get("item_id") == exclude_item_id:
            continue
        result = dict(item)
        result["score"] = float(scores[idx])
        results.append(result)
        if len(results) >= limit:
            break
    return results


def index_exists(prefix):
    return os.path.isfile(prefix + INDEX_METADATA_SUFFIX) and os.path.isfile(prefix + INDEX_EMBEDDINGS_SUFFIX)


def format_result(result, rank=None):
    prefix = f"{rank}. " if rank is not None else ""
    label = result.get("label", result.get("item_id", "unknown"))
    kind = result.get("kind", "asset")
    role = result.get("stem_role") or result.get("kind")
    score = result.get("score")
    score_text = f"{score:.4f}" if isinstance(score, (float, int)) else "n/a"
    feature_bits = []
    if result.get("tempo") is not None:
        feature_bits.append(f"BPM {result['tempo']}")
    if result.get("key"):
        feature_bits.append(f"Key {result['key']}")
    if result.get("section_labels"):
        feature_bits.append("Sections " + ",".join(str(label) for label in result["section_labels"][:4]))
    if result.get("mood_candidates"):
        feature_bits.append("Mood " + ",".join(result["mood_candidates"][:3]))
    summary = " | ".join(feature_bits)
    line = f"{prefix}{label} [{kind}/{role}] score={score_text}"
    if summary:
        line += f" | {summary}"
    return line


def format_results(results, header=None):
    lines = []
    if header:
        lines.append(header)
    for idx, result in enumerate(results, start=1):
        lines.append(format_result(result, rank=idx))
        lines.append(f"   item_id: {result.get('item_id')}")
        lines.append(f"   path: {result.get('path')}")
        if result.get("lyric_excerpt"):
            lines.append(f"   lyrics: {result.get('lyric_excerpt')}")
        if result.get("document_path"):
            lines.append(f"   document: {result.get('document_path')}")
    return "\n".join(lines)
