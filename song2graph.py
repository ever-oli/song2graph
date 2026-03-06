#!/usr/bin/env python


import os
import sys
import pickle
import argparse
import subprocess
import fnmatch
import hashlib
import shutil
import json
import time
from math import log2, pow

from numba import cuda
import numpy as np 
import librosa
import soundfile as sf
import torch
import torchcrepe
from yt_dlp import YoutubeDL
from sf_segmenter.segmenter import Segmenter
from basic_pitch import ICASSP_2022_MODEL_PATH, CT_PRESENT, TFLITE_PRESENT, ONNX_PRESENT, TF_PRESENT
from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
from basic_pitch.inference import predict_and_save
from basic_pitch.inference import predict
from clap_handler import ClapIndexer, format_results, index_exists, load_index, save_index, search_by_embedding
from annotation_handler import annotate_document
from transcription_handler import (
    FasterWhisperTranscriber,
    align_transcription_to_sections,
    normalize_transcription_payload,
    write_transcription_json,
)

##########################################
############### SONG2GRAPH ###############
######## music structure pipeline ########
##########################################

class Video:
    def __init__(self,name,video,audio):
        self.id = ""
        self.url = ""
        self.name = name
        self.video = video
        self.audio = audio
        self.video_features = []
        self.audio_features = []

### Library

LIBRARY_FILENAME = "library/database.p"
basic_pitch_model = ""

if TFLITE_PRESENT or TF_PRESENT:
    basic_pitch_model = build_icassp_2022_model_path(FilenameSuffix.tflite)
elif ONNX_PRESENT:
    basic_pitch_model = build_icassp_2022_model_path(FilenameSuffix.onnx)
elif CT_PRESENT:
    basic_pitch_model = build_icassp_2022_model_path(FilenameSuffix.coreml)
else:
    basic_pitch_model = ICASSP_2022_MODEL_PATH

def write_library(videos):
    with open(LIBRARY_FILENAME, "wb") as lib:
        pickle.dump(videos, lib)


def read_library():
    try:
        with open(LIBRARY_FILENAME, "rb") as lib:
            return pickle.load(lib)
    except:
        print("No Database file found:", LIBRARY_FILENAME)
    return []


################## VIDEO PROCESSING ##################

def audio_extract(vidobj,file):
    print("audio_extract",file)
    command = "ffmpeg -hide_banner -loglevel panic -i "+file+" -ab 160k -ac 2 -ar 44100 -vn -y " + vidobj.audio
    subprocess.call(command,shell=True)
    return vidobj.audio

def video_download(vidobj,url):
    print("video_download",url)
    ydl_opts = {
    'outtmpl': 'library/%(id)s',
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best--merge-output-format mp4',
    } 
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download(url)

    with ydl: result = ydl.extract_info(url, download=True)

    if 'entries' in result: video = result['entries'][0] # Can be a playlist or a list of videos
    else: video = result  # Just a video

    filename = f"library/{video['id']}.{video['ext']}"
    print("video_download: filename",filename,"extension",video['ext'])
    vidobj.id = video['id']
    vidobj.name = video['title']
    vidobj.video = filename
    vidobj.url = url
    return vidobj

def video_process(vids,videos):
    for vid in vids:
        print('------ process video',vid)
        # check if id already in db
        download_vid = True
        for video in videos:
            if video.id == vid:
                print("already in db",vid)
                download_vid = False
                break

        # analyse videos and save to disk
        if download_vid:
            video = Video(vid,vid,f"library/{vid}.wav")
            video = video_download(video,f"https://www.youtube.com/watch?v={vid}")
            audio_extract(video,video.video)
            videos.append(video)
            print("NAME",video.name,"VIDEO",video.video,"AUDIO",video.audio)
            write_library(videos)
            print("video_process DONE",len(videos))
    return videos

################## AUDIO PROCESSING ##################

def audio_directory_process(vids, videos):
    filesToProcess = []
    for vid in vids:
        path = vid
        pattern = "*.mp3"
        for filename in fnmatch.filter(os.listdir(path), pattern):
            filepath = os.path.join(path, filename)
            print(filepath)
            if os.path.isfile(filepath):
                filesToProcess.append(filepath)

    print('Found', len(filesToProcess), 'wav or mp3 files')
    if len(filesToProcess) > 0:
        videos = audio_process(filesToProcess, videos)
    return videos

def audio_process(vids, videos):
    for vid in vids:
        print('------ process audio',vid)
        # extract file name
        audioname = vid.split("/")[-1]
        audioname, _ = audioname.split(".")

        library_dir = os.path.join(os.getcwd(), 'library')
        abs_vid = os.path.abspath(vid)
        if os.path.commonpath([abs_vid, library_dir]) == library_dir:
            audioid = audioname
        else:
            # generate a unique ID based on file path and name
            hash_object = hashlib.sha256(vid.encode())
            audioid = hash_object.hexdigest()
            audioid = f"{audioname}_{audioid}"

        # check if id already in db
        process_audio = True
        for video in videos:
            if video.id == audioid:
                print("already in db",vid)
                process_audio = False
                break

        # check if is mp3 and convert it to wav
        if vid.endswith(".mp3"):
            # convert mp3 to wav and save it
            print('converting mp3 to wav:', vid)
            y, sr = librosa.load(path=vid, sr=None, mono=False)
            path = os.path.join(os.getcwd(), 'library', audioid+'.wav')
            # resample to 44100k if required
            if sr != 44100:
                print('converting audio file to 44100:', vid)
                y = librosa.resample(y, orig_sr=sr, target_sr=44100)
            sf.write(path, np.ravel(y), 44100)
            vid = path

        # check if is wav and copy it to local folder
        elif vid.endswith(".wav"):
            path1 = vid
            path2 = os.path.join(os.getcwd(), 'library', audioid+'.wav')
            y, sr = librosa.load(path=vid, sr=None, mono=False)
            if sr != 44100:
                print('converting audio file to 44100:', vid)
                y = librosa.resample(y, orig_sr=sr, target_sr=44100)
                sf.write(path2, y, 44100)
            elif os.path.abspath(path1) == os.path.abspath(path2):
                pass
            else:
                shutil.copy2(path1, path2)
            vid = path2

        # analyse videos and save to disk
        if process_audio:
            video = Video(audioname,'',vid)
            video.id = audioid
            video.url = vid
            videos.append(video)
            write_library(videos)
            print("Finished procesing files:",len(videos))
            
    return videos

################## AUDIO FEATURES ##################

def root_mean_square(data):
    return float(np.sqrt(np.mean(np.square(data))))

def loudness_of(data):
    return root_mean_square(data)

def normalized(list):
    """Given an audio buffer, return it with the loudest value scaled to 1.0"""
    return list.astype(np.float32) / float(np.amax(np.abs(list)))

neg80point8db = 0.00009120108393559096
bit_depth = 16
default_silence_threshold = (neg80point8db * (2 ** (bit_depth - 1))) * 4

def start_of(list, threshold=default_silence_threshold, samples_before=1):
    if int(threshold) != threshold:
        threshold = threshold * float(2 ** (bit_depth - 1))
    index = np.argmax(np.absolute(list) > threshold)
    if index > (samples_before - 1):
        return index - samples_before
    else:
        return 0

def end_of(list, threshold=default_silence_threshold, samples_after=1):
    if int(threshold) != threshold:
        threshold = threshold * float(2 ** (bit_depth - 1))
    rev_index = np.argmax(
        np.flipud(np.absolute(list)) > threshold
    )
    if rev_index > (samples_after - 1):
        return len(list) - (rev_index - samples_after)
    else:
        return len(list)

def trim_data(
    data,
    start_threshold=default_silence_threshold,
    end_threshold=default_silence_threshold
):
    start = start_of(data, start_threshold)
    end = end_of(data, end_threshold)

    return data[start:end]

def load_and_trim(file):
    y, rate = librosa.load(file, mono=True)
    y = normalized(y)
    trimmed = trim_data(y)
    return trimmed, rate

def get_loudness(file):
    loudness = -1
    try:
        audio, rate = load_and_trim(file)
        loudness = loudness_of(audio)
    except Exception as e:
        sys.stderr.write(f"Failed to run on {file}: {e}\n")
    return loudness

def get_volume(file):
    volume = -1
    avg_volume = -1
    try:
        audio, rate = load_and_trim(file)
        volume = librosa.feature.rms(y=audio)[0]
        avg_volume = np.mean(volume)
        loudness = loudness_of(audio)
    except Exception as e:
        sys.stderr.write(f"Failed to get Volume and Loudness on {file}: {e}\n")
    return volume, avg_volume, loudness

def get_key(freq):
    A4 = 440
    C0 = A4*pow(2, -4.75)
    name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    h = round(12*log2(freq/C0))
    octave = h // 12
    n = h % 12
    return name[n] + str(octave)

def get_average_pitch(pitch):
    pitches = []
    confidences_thresh = 0.8
    i = 0
    while i < len(pitch):
        if(pitch[i][2] > confidences_thresh):
            pitches.append(pitch[i][1])
        i += 1
    if len(pitches) > 0:
        average_frequency = np.array(pitches).mean()
        average_key = get_key(average_frequency)
    else:
        average_frequency = 0
        average_key = "A0"
    return average_frequency,average_key

def get_intensity(y, sr, beats):
    # Beat-synchronous Loudness - Intensity
    CQT = librosa.cqt(y=y, sr=sr, fmin=librosa.note_to_hz('A1'))
    freqs = librosa.cqt_frequencies(CQT.shape[0], fmin=librosa.note_to_hz('A1'))
    perceptual_CQT = librosa.perceptual_weighting(CQT**2, freqs, ref=np.max)
    CQT_sync = librosa.util.sync(perceptual_CQT, beats, aggregate=np.median)
    return CQT_sync

def get_pitch(y_harmonic, sr, beats):
    # Chromagram
    C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    # Beat-synchronous Chroma - Pitch
    C_sync = librosa.util.sync(C, beats, aggregate=np.median)
    return C_sync

def get_timbre(y, sr, beats):
    # Mel spectogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    # MFCC - Timbre
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    M = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    # Beat-synchronous MFCC - Timbre
    M_sync = librosa.util.sync(M, beats)
    return M_sync

def get_segments(audio_file):
    segmenter = Segmenter()
    boundaries, labs = segmenter.proc_audio(audio_file)
    return boundaries,labs 


def scalarize(value):
    """Convert numpy scalars/size-1 arrays to plain Python scalars."""
    if isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else value
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolve_audio_file(vid):
    # Prefer the canonical local library copy so repo moves or renames do not
    # break previously indexed absolute paths.
    canonical_path = os.path.join(os.getcwd(), 'library', vid.id + '.wav')
    if os.path.isfile(canonical_path):
        return canonical_path

    candidates = []
    if isinstance(vid.audio, str) and vid.audio:
        candidates.append(vid.audio)
    if isinstance(vid.url, str) and vid.url.endswith((".wav", ".mp3")):
        candidates.append(vid.url)

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return canonical_path


def get_stem_output_dir(file_id, demucsmodel="htdemucs_6s"):
    return os.path.join(os.getcwd(), 'separated', demucsmodel, file_id)


def get_stem_paths(file_id, demucsmodel="htdemucs_6s"):
    stems = ['bass', 'drums', 'guitar', 'other', 'piano', 'vocals']
    output_dir = get_stem_output_dir(file_id, demucsmodel)
    return output_dir, [os.path.join(output_dir, stem + '.wav') for stem in stems]


def ensure_stems(audio_file, file_id, demucsmodel="htdemucs_6s"):
    output_dir, stem_paths = get_stem_paths(file_id, demucsmodel)
    if not all(os.path.isfile(path) for path in stem_paths):
        stemsplit(audio_file, demucsmodel)
    return output_dir, stem_paths


def ensure_midi_outputs(audio_paths, output_dir):
    expected_midi = [
        os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(path))[0] + "_basic_pitch.mid",
        )
        for path in audio_paths
    ]
    if not all(os.path.isfile(path) for path in expected_midi):
        extractMIDI(audio_paths, output_dir)
    return expected_midi


def should_refresh_audio_features(audio_features):
    frequency_frames = audio_features.get("frequency_frames")
    if frequency_frames is None:
        return True
    if hasattr(frequency_frames, "__len__") and len(frequency_frames) == 0:
        return True
    if audio_features.get("pitch_backend") != "torchcrepe":
        return True
    return False


def get_clap_index_prefix():
    return os.path.join(os.getcwd(), "library", "clap_index")


def get_documents_dir():
    return os.path.join(os.getcwd(), "library", "documents")


def get_document_path(file_id):
    return os.path.join(get_documents_dir(), f"{file_id}.json")


def get_transcription_path(file_id):
    return os.path.join(get_documents_dir(), f"{file_id}.lyrics.json")


def get_annotation_path(file_id):
    return os.path.join(get_documents_dir(), f"{file_id}.annotation.json")


def load_json_file(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_float(value, digits=4):
    if value is None:
        return None
    return round(float(scalarize(value)), digits)


def summarize_sections(audio_features):
    boundaries = audio_features.get("segments_boundaries")
    labels = audio_features.get("segments_labels")
    if boundaries is None:
        boundaries = []
    elif hasattr(boundaries, "tolist"):
        boundaries = boundaries.tolist()
    if labels is None:
        labels = []
    elif hasattr(labels, "tolist"):
        labels = labels.tolist()
    sections = []
    for idx, label in enumerate(labels):
        start = boundaries[idx] if idx < len(boundaries) else None
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else None
        sections.append({
            "index": idx,
            "label": label,
            "start": to_float(start),
            "end": to_float(end),
        })
    return sections


def summarize_structure_labels(sections):
    labels = []
    seen = set()
    for section in sections:
        label = section.get("label")
        if label is None:
            continue
        label_text = str(label)
        if label_text not in seen:
            labels.append(label_text)
            seen.add(label_text)
    return labels


def canonicalize_section_labels(sections):
    labels = summarize_structure_labels(sections)
    mapping = {}
    canonical_labels = []
    for idx, label in enumerate(labels):
        lowered = str(label).lower()
        if lowered in {"0", "0.0", "a"}:
            canonical = "section_a"
        elif lowered in {"1", "1.0", "b"}:
            canonical = "section_b"
        elif lowered in {"2", "2.0", "c"}:
            canonical = "section_c"
        else:
            canonical = f"section_{idx + 1}"
        mapping[str(label)] = canonical
        canonical_labels.append(canonical)
    return canonical_labels, mapping


def infer_mood_tags(features):
    tags = []
    tempo = features.get("tempo")
    intensity = features.get("intensity")
    timbre = features.get("timbre")

    if tempo is not None:
        if tempo < 90:
            tags.append("slow")
        elif tempo < 120:
            tags.append("midtempo")
        else:
            tags.append("upbeat")

    if intensity is not None:
        if intensity < 0.25:
            tags.append("soft")
        elif intensity > 0.6:
            tags.append("driving")

    if timbre is not None and timbre > 0:
        tags.append("bright")

    return tags


def infer_instrumentation_roles(reference_paths):
    roles = []
    stem_names = ["bass", "drums", "guitar", "other", "piano", "vocals"]
    stem_files = set(os.path.basename(path) for path in reference_paths.get("stems", []))
    for stem in stem_names:
        if f"{stem}.wav" in stem_files:
            roles.append({
                "role": stem,
                "source": "demucs",
                "confidence": 0.7,
            })
    return roles


def build_retrieval_hints(vid, features, sections, references, aligned_lyrics):
    canonical_sections, _ = canonicalize_section_labels(sections)
    hints = {
        "stem_item_ids": [f"{vid.id}:{stem}" for stem in ["bass", "drums", "guitar", "other", "piano", "vocals"]],
        "text_queries": [],
        "audio_reference_paths": [path for path in [references.get("source_audio")] if path],
    }

    if features.get("key") and features.get("tempo") is not None:
        hints["text_queries"].append(
            f"{features['key']} tonal music around {features['tempo']} bpm"
        )
    if sections:
        hints["text_queries"].append(
            "structured song with sections " + ", ".join(canonical_sections[:4])
        )
    if aligned_lyrics:
        lyric_text = " ".join(
            section.get("text", "") for section in aligned_lyrics if section.get("text")
        ).strip()
        if lyric_text:
            hints["text_queries"].append(lyric_text[:160])
    return hints


def summarize_audio_features(audio_features):
    return {
        "tempo": to_float(audio_features.get("tempo"), digits=2),
        "duration": to_float(audio_features.get("duration"), digits=3),
        "frequency": to_float(audio_features.get("frequency"), digits=2),
        "key": audio_features.get("key"),
        "timbre": to_float(audio_features.get("timbre"), digits=4),
        "pitch": to_float(audio_features.get("pitch"), digits=4),
        "intensity": to_float(audio_features.get("intensity"), digits=4),
        "avg_volume": to_float(audio_features.get("avg_volume"), digits=6),
        "loudness": to_float(audio_features.get("loudness"), digits=6),
        "segment_count": len(audio_features.get("segments_labels", [])),
        "sections": summarize_sections(audio_features),
    }


def collect_reference_paths(file_id):
    source_audio = os.path.join(os.getcwd(), 'library', f'{file_id}.wav')
    feature_cache = os.path.join(os.getcwd(), 'library', f'{file_id}.a')
    stem_dir, stem_paths = get_stem_paths(file_id, 'htdemucs_6s')
    midi_paths = [
        os.path.splitext(path)[0] + "_basic_pitch.mid"
        for path in stem_paths
    ]
    return {
        "source_audio": source_audio if os.path.isfile(source_audio) else None,
        "feature_cache": feature_cache if os.path.isfile(feature_cache) else None,
        "stem_directory": stem_dir if os.path.isdir(stem_dir) else None,
        "stems": [path for path in stem_paths if os.path.isfile(path)],
        "midi": [path for path in midi_paths if os.path.isfile(path)],
        "clap_index_metadata": get_clap_index_prefix() + ".json" if index_exists(get_clap_index_prefix()) else None,
        "clap_index_embeddings": get_clap_index_prefix() + ".npz" if index_exists(get_clap_index_prefix()) else None,
    }


def build_song_document(vid, transcription=None):
    audio_features = vid.audio_features
    references = collect_reference_paths(vid.id)
    summarized_features = summarize_audio_features(audio_features)
    sections = summarize_sections(audio_features)
    canonical_structure_labels, section_label_map = canonicalize_section_labels(sections)
    aligned_lyrics = align_transcription_to_sections(transcription, sections) if transcription else []
    existing_annotation = load_json_file(get_annotation_path(vid.id))
    annotation_payload = existing_annotation.get("annotation") if existing_annotation else None
    document = {
        "schema_version": "0.3.0",
        "generated_at_epoch": int(time.time()),
        "song_id": vid.id,
        "name": vid.name,
        "source": {
            "url": vid.url,
            "audio_path": resolve_audio_file(vid),
        },
        "audio_features": summarized_features,
        "sections": sections,
        "lyrics": transcription or None,
        "lyrics_alignment": aligned_lyrics,
        "embeddings": {
            "clap_index_item_ids": [vid.id] + [f"{vid.id}:{stem}" for stem in ['bass', 'drums', 'guitar', 'other', 'piano', 'vocals']],
            "index_metadata_path": references["clap_index_metadata"],
            "index_embeddings_path": references["clap_index_embeddings"],
        },
        "analysis": {
            "semantic_labels": [],
            "mood_tags": annotation_payload.get("mood_tags", []) if annotation_payload else [],
            "mood_bootstrap": infer_mood_tags(summarized_features),
            "structure_labels": annotation_payload.get("structure_labels", canonical_structure_labels) if annotation_payload else canonical_structure_labels,
            "section_label_map": section_label_map,
            "instrumentation_roles": annotation_payload.get("instrumentation_roles", []) if annotation_payload else [],
            "instrumentation_bootstrap": infer_instrumentation_roles(references),
            "genre_candidates": annotation_payload.get("genre_candidates", []) if annotation_payload else [],
            "influence_candidates": annotation_payload.get("influence_candidates", []) if annotation_payload else [],
            "retrieval_hints": build_retrieval_hints(vid, summarized_features, sections, references, aligned_lyrics),
            "llm_annotations": {
                "status": "complete" if annotation_payload else "pending",
                "summary": annotation_payload.get("summary") if annotation_payload else None,
                "arrangement_notes": annotation_payload.get("arrangement_notes", []) if annotation_payload else [],
                "genre_candidates": annotation_payload.get("genre_candidates", []) if annotation_payload else [],
                "influence_candidates": annotation_payload.get("influence_candidates", []) if annotation_payload else [],
                "retrieval_queries": annotation_payload.get("retrieval_queries", []) if annotation_payload else [],
                "provider": existing_annotation.get("provider") if existing_annotation else None,
                "model": existing_annotation.get("model") if existing_annotation else None,
                "annotation_path": get_annotation_path(vid.id) if os.path.isfile(get_annotation_path(vid.id)) else None,
            },
            "notes": [],
        },
        "references": references,
    }
    return document


def write_song_document(file_id, document):
    path = get_document_path(file_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
    return path


def ensure_song_document(vid, transcription=None):
    document = build_song_document(vid, transcription=transcription)
    return write_song_document(vid.id, document)


def collect_audio_assets(vid):
    audio_file = resolve_audio_file(vid)
    _, stem_paths = ensure_stems(audio_file, vid.id, 'htdemucs_6s')
    stem_names = ['bass', 'drums', 'guitar', 'other', 'piano', 'vocals']
    features = summarize_audio_features(vid.audio_features)
    section_labels = [section["label"] for section in features["sections"]]
    structure_labels, _ = canonicalize_section_labels(features["sections"])
    document = load_json_file(get_document_path(vid.id))
    analysis = document.get("analysis", {}) if document else {}
    mood_tags = analysis.get("mood_tags") or analysis.get("mood_bootstrap") or infer_mood_tags(features)
    transcription = load_json_file(get_transcription_path(vid.id))
    lyric_excerpt = None
    if transcription:
        lyric_excerpt = transcription.get("normalized_excerpt")
    assets = [{
        "item_id": vid.id,
        "video_id": vid.id,
        "kind": "track",
        "label": vid.name,
        "path": audio_file,
        "tempo": features["tempo"],
        "key": features["key"],
        "frequency": features["frequency"],
        "duration": features["duration"],
        "section_labels": section_labels,
        "structure_labels": structure_labels,
        "sections": features["sections"],
        "mood_candidates": mood_tags,
        "stem_role": "full_mix",
        "lyric_excerpt": lyric_excerpt,
        "document_path": get_document_path(vid.id) if os.path.isfile(get_document_path(vid.id)) else None,
        "lyrics_path": get_transcription_path(vid.id) if os.path.isfile(get_transcription_path(vid.id)) else None,
    }]
    for stem_name, stem_path in zip(stem_names, stem_paths):
        if os.path.isfile(stem_path):
            assets.append({
                "item_id": f"{vid.id}:{stem_name}",
                "video_id": vid.id,
                "kind": "stem",
                "stem": stem_name,
                "label": f"{vid.name} [{stem_name}]",
                "path": stem_path,
                "tempo": features["tempo"],
                "key": features["key"],
                "frequency": features["frequency"],
                "duration": features["duration"],
                "section_labels": section_labels,
                "structure_labels": structure_labels,
                "sections": features["sections"],
                "mood_candidates": mood_tags,
                "stem_role": stem_name,
                "lyric_excerpt": lyric_excerpt,
                "document_path": get_document_path(vid.id) if os.path.isfile(get_document_path(vid.id)) else None,
                "lyrics_path": get_transcription_path(vid.id) if os.path.isfile(get_transcription_path(vid.id)) else None,
            })
    return assets


def build_clap_index(videos, selected_ids):
    indexer = ClapIndexer()
    selected = videos if selected_ids == ["all"] else [vid for vid in videos if vid.id in selected_ids]
    metadata = []
    for vid in selected:
        metadata.extend(collect_audio_assets(vid))
    audio_paths = [item["path"] for item in metadata]
    if len(audio_paths) == 0:
        raise RuntimeError("No audio assets available to index.")
    embeddings = indexer.embed_audio_files(audio_paths)
    prefix = get_clap_index_prefix()
    metadata_path, embeddings_path = save_index(prefix, metadata, embeddings)
    return metadata_path, embeddings_path, len(metadata)


def search_clap_similar(item_id, limit):
    prefix = get_clap_index_prefix()
    if not index_exists(prefix):
        raise RuntimeError("CLAP index does not exist yet. Run with --clap-index first.")
    metadata, embeddings = load_index(prefix)
    item_lookup = {item["item_id"]: idx for idx, item in enumerate(metadata)}
    if item_id not in item_lookup:
        raise RuntimeError(f"CLAP item not found in index: {item_id}")
    idx = item_lookup[item_id]
    return search_by_embedding(embeddings[idx], metadata, embeddings, limit=limit, exclude_item_id=item_id)


def search_clap_text(text_query, limit):
    prefix = get_clap_index_prefix()
    if not index_exists(prefix):
        raise RuntimeError("CLAP index does not exist yet. Run with --clap-index first.")
    metadata, embeddings = load_index(prefix)
    indexer = ClapIndexer()
    query_embedding = indexer.embed_texts([text_query])[0]
    return search_by_embedding(query_embedding, metadata, embeddings, limit=limit)


def transcribe_lyrics(vid, model_size="small", language=None):
    source_audio = resolve_audio_file(vid)
    transcriber = FasterWhisperTranscriber(model_size=model_size)
    payload = transcriber.transcribe(source_audio, language=language)
    payload = normalize_transcription_payload(payload)
    output_path = write_transcription_json(get_transcription_path(vid.id), payload)
    return payload, output_path


def write_annotation_json(file_id, payload):
    path = get_annotation_path(file_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def annotate_song_document(vid, model=None):
    document = load_json_file(get_document_path(vid.id))
    if document is None:
        document_path = ensure_song_document(vid, transcription=load_json_file(get_transcription_path(vid.id)))
        document = load_json_file(document_path)
    annotation_result = annotate_document(document, model=model)
    annotation_path = write_annotation_json(vid.id, annotation_result)
    refreshed_document_path = ensure_song_document(vid, transcription=load_json_file(get_transcription_path(vid.id)))
    return annotation_result, annotation_path, refreshed_document_path


def ingest_assets(videos, selected_ids, transcribe=False, transcription_model="small", transcription_language=None, annotate=False, annotation_model=None):
    selected = videos if selected_ids == ["all"] else [vid for vid in videos if vid.id in selected_ids]
    if len(selected) == 0:
        raise RuntimeError("No matching assets found for ingestion.")

    document_paths = []
    annotation_paths = []
    for vid in selected:
        transcription_payload = load_json_file(get_transcription_path(vid.id))
        if transcribe:
            transcription_payload, transcription_path = transcribe_lyrics(
                vid,
                model_size=transcription_model,
                language=transcription_language,
            )
            print("lyrics", transcription_path)

        document_path = ensure_song_document(
            vid,
            transcription=transcription_payload,
        )
        document_paths.append(document_path)
        print("document", document_path)

        if annotate:
            annotation_result, annotation_path, refreshed_document_path = annotate_song_document(
                vid,
                model=annotation_model,
            )
            annotation_paths.append(annotation_path)
            if refreshed_document_path != document_path:
                print("document", refreshed_document_path)
            print("annotation", annotation_path)
            print("annotation summary", annotation_result["annotation"].get("summary"))

    metadata_path, embeddings_path, asset_count = build_clap_index(videos, selected_ids)
    return {
        "documents": document_paths,
        "annotations": annotation_paths,
        "clap_index_metadata": metadata_path,
        "clap_index_embeddings": embeddings_path,
        "clap_asset_count": asset_count,
    }


def get_pitch_dnn(audio_file):
    sample_rate = 16000
    hop_length = 160
    audio, sr = librosa.load(audio_file, sr=sample_rate, mono=True)
    if audio.size == 0:
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_tensor = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.inference_mode():
        frequency, confidence = torchcrepe.predict(
            audio_tensor,
            sample_rate,
            hop_length,
            fmin=32.7,
            fmax=1975.5,
            model="tiny",
            batch_size=512,
            device=device,
            return_periodicity=True,
        )

    frequency = frequency.squeeze(0).detach().cpu().numpy()
    confidence = confidence.squeeze(0).detach().cpu().numpy()
    timestamps = np.arange(frequency.shape[0], dtype=np.float32) * (hop_length / sample_rate)

    valid = np.isfinite(frequency) & np.isfinite(confidence) & (frequency > 0)
    if not np.any(valid):
        return []

    return np.column_stack([timestamps[valid], frequency[valid], confidence[valid]]).tolist()

def stemsplit(destination, demucsmodel):
    out_dir = os.path.join(os.getcwd(), 'separated')
    # Run Demucs via the active interpreter so uv-managed virtualenvs work
    # even when the standalone `demucs` script is not on PATH.
    subprocess.run(
        [sys.executable, "-m", "demucs", destination, "-n", demucsmodel, "--out", out_dir],
        check=True,
    )  # '--mp3'

def extractMIDI(audio_paths, output_dir):
    print('- Extract Midi')
    save_midi = True
    sonify_midi = False
    save_model_outputs = False
    save_notes = False

    predict_and_save(audio_path_list=audio_paths, 
                  output_directory=output_dir, 
                  save_midi=save_midi, 
                  sonify_midi=sonify_midi, 
                  save_model_outputs=save_model_outputs, 
                  save_notes=save_notes,
                  model_or_model_path=basic_pitch_model)


def quantizeAudio(vid, bpm=120, keepOriginalBpm = False, pitchShiftFirst = False, extractMidi = False):
    try:
        import pyrubberband as pyrb
    except ImportError as exc:
        raise RuntimeError(
            "Quantization requires the optional 'quantization' dependencies. Run `uv sync --frozen --extra quantization`."
        ) from exc

    tempo_value = float(scalarize(vid.audio_features["tempo"]))
    source_audio_file = resolve_audio_file(vid)
    print("Quantize Audio: Target BPM", bpm, 
        "-- id:",vid.id,
        "bpm:",round(tempo_value,2),
        "frequency:",round(vid.audio_features['frequency'],2),
        "key:",vid.audio_features['key'],
        "timbre:",round(vid.audio_features['timbre'],2),
        "name:",vid.name,
        'keepOriginalBpm:', keepOriginalBpm
        )

    # load audio file
    y, sr = librosa.load(source_audio_file, sr=None)

    # Keep Original Song BPM
    if keepOriginalBpm:
        bpm = tempo_value
        print('Keep original audio file BPM:', vid.audio_features['tempo'])
    # Pitch Shift audio file to desired BPM first
    elif pitchShiftFirst: # WORK IN PROGRESS
        print('Pitch Shifting audio to desired BPM', bpm)
        # Desired tempo in bpm
        original_tempo = tempo_value
        speed_factor = bpm / original_tempo
        # Resample the audio to adjust the sample rate accordingly
        sr_stretched = int(sr / speed_factor)
        y = librosa.resample(y=y, orig_sr=sr, target_sr=sr_stretched) #,  res_type='linear'
        y = librosa.resample(y, orig_sr=sr, target_sr=44100)

    # extract beat
    y_harmonic, y_percussive = librosa.effects.hpss(y=y)
    tempo, beats = librosa.beat.beat_track(sr=sr, onset_envelope=librosa.onset.onset_strength(y=y_percussive, sr=sr), trim=False)
    tempo = float(scalarize(tempo))
    beat_frames = librosa.frames_to_samples(beats)

    # generate metronome
    fixed_beat_times = []
    for i in range(len(beat_frames)):
        fixed_beat_times.append(i * 120 / bpm)
    fixed_beat_frames = librosa.time_to_samples(fixed_beat_times)

    # construct time map
    time_map = []
    for i in range(len(beat_frames)):
        new_member = (beat_frames[i], fixed_beat_frames[i])
        time_map.append(new_member)

    # add ending to time map
    original_length = len(y+1)
    orig_end_diff = original_length - time_map[i][0]
    new_ending = int(round(time_map[i][1] + orig_end_diff * (tempo / bpm)))
    new_member = (original_length, new_ending)
    time_map.append(new_member)

    # time strech audio
    print('- Quantize Audio: source')
    strechedaudio = pyrb.timemap_stretch(y, sr, time_map)

    path_suffix = (
        f"Key {vid.audio_features['key']} - "
        f"Freq {round(vid.audio_features['frequency'], 2)} - "
        f"Timbre {round(vid.audio_features['timbre'], 2)} - "
        f"BPM Original {int(tempo_value)} - "
        f"BPM {bpm}"
    )
    path_prefix = (
        f"{vid.id} - {vid.name}"
    )

    audiofilepaths = []
    # save audio to disk
    path = os.path.join(os.getcwd(), 'processed', path_prefix + " - " + path_suffix +'.wav')
    sf.write(path, strechedaudio, sr)
    audiofilepaths.append(path)

    # process stems
    _, stem_paths = ensure_stems(source_audio_file, vid.id, 'htdemucs_6s')
    stems = ['bass', 'drums', 'guitar', 'other', 'piano', 'vocals']
    for stem, path in zip(stems, stem_paths):
        print(f"- Quantize Audio: {stem}")
        y, sr = librosa.load(path, sr=None)
        strechedaudio = pyrb.timemap_stretch(y, sr, time_map)
        # save stems to disk
        path = os.path.join(os.getcwd(), 'processed', path_prefix + " - Stem " + stem + " - " + path_suffix +'.wav')
        sf.write(path, strechedaudio, sr)
        audiofilepaths.append(path)

    # metronome click (optinal)
    click = False
    if click:
        clicks_audio = librosa.clicks(times=fixed_beat_times, sr=sr)
        print(len(clicks_audio), len(strechedaudio))
        clicks_audio = clicks_audio[:len(strechedaudio)] 
        path = os.path.join(os.getcwd(), 'processed', vid.id + '- click.wav')
        sf.write(path, clicks_audio, sr)

    if extractMidi:
        output_dir = os.path.join(os.getcwd(), 'processed')
        ensure_midi_outputs(audiofilepaths, output_dir)


def get_audio_features(file,file_id,extractMidi = False):
    print("------------------------------ get_audio_features:",file_id,"------------------------------")
    print('1/8 segementation')
    segments_boundaries,segments_labels = get_segments(file)
   
    print('2/8 pitch tracking')
    frequency_frames = get_pitch_dnn(file)
    average_frequency,average_key = get_average_pitch(frequency_frames)
    
    print('3/8 load sample')
    y, sr = librosa.load(file, sr=None)
    song_duration = librosa.get_duration(y=y, sr=sr)
    
    print('4/8 sample separation')
    y_harmonic, y_percussive = librosa.effects.hpss(y=y)
    
    print('5/8 beat tracking')
    tempo, beats = librosa.beat.beat_track(sr=sr, onset_envelope=librosa.onset.onset_strength(y=y_percussive, sr=sr), trim=False)
    tempo = float(scalarize(tempo))

    print('6/8 feature extraction')
    CQT_sync = get_intensity(y, sr, beats)
    C_sync = get_pitch(y_harmonic, sr, beats)
    M_sync = get_timbre(y, sr, beats)
    volume, avg_volume, loudness = get_volume(file)
   
    print('7/8 feature aggregation')
    intensity_frames = np.matrix(CQT_sync).getT()
    pitch_frames = np.matrix(C_sync).getT()
    timbre_frames = np.matrix(M_sync).getT()

    if cuda.is_available():
        print('Cleaning up GPU memory')
        device = cuda.get_current_device()
        device.reset()

    print('8/8 split stems')
    stem_output_dir, stem_paths = ensure_stems(file, file_id, 'htdemucs_6s')

    if extractMidi:
        ensure_midi_outputs(stem_paths, stem_output_dir)

    audio_features = {
        "id":file_id,
        "tempo":tempo,
        "duration":song_duration,
        "timbre":np.mean(timbre_frames),
        "timbre_frames":timbre_frames,
        "pitch":np.mean(pitch_frames),
        "pitch_frames":pitch_frames,
        "intensity":np.mean(intensity_frames),
        "intensity_frames":intensity_frames,
        "volume": volume,
        "avg_volume": avg_volume,
        "loudness": loudness,
        "beats":librosa.frames_to_time(beats, sr=sr),
        "segments_boundaries":segments_boundaries,
        "segments_labels":segments_labels,
        "frequency_frames":frequency_frames,
        "frequency":average_frequency,
        "key":average_key,
        "pitch_backend":"torchcrepe",
    }
    return audio_features

################## SEARCH NEAREST AUDIO ##################

previous_list = []

def get_nearest(query,videos,querybpm, searchforbpm):
    global previous_list
    # print("Search: query:", query.name, '- Incl. BPM in search:', searchforbpm)
    nearest = {}
    smallest = 1000000000
    smallestBPM = 1000000000
    smallestTimbre = 1000000000
    smallestIntensity = 1000000000
    for vid in videos:
        if vid.id != query.id:
            comp_bpm = abs(querybpm - vid.audio_features['tempo'])
            comp_timbre = abs(query.audio_features["timbre"] - vid.audio_features['timbre'])
            comp_intensity = abs(query.audio_features["intensity"] - vid.audio_features['intensity'])
            #comp = abs(query.audio_features["pitch"] - vid.audio_features['pitch'])
            comp = abs(query.audio_features["frequency"] - vid.audio_features['frequency'])

            if searchforbpm:
                if vid.id not in previous_list and comp < smallest and comp_bpm < smallestBPM:# and comp_timbre < smallestTimbre:
                    smallest = comp
                    smallestBPM = comp_bpm
                    smallestTimbre = comp_timbre
                    nearest = vid
            else:
                if vid.id not in previous_list and comp < smallest:
                    smallest = comp
                    smallestBPM = comp_bpm
                    smallestTimbre = comp_timbre
                    nearest = vid
            #print("--- result",i['file'],i['average_frequency'],i['average_key'],"diff",comp)
    # print(nearest)
    previous_list.append(nearest.id)
   
    if len(previous_list) >= len(videos)-1:
        previous_list.pop(0)
        # print("getNearestPitch: previous_list, pop first")
    # print("get_nearest",nearest.id)
    return nearest

def getNearest(k, array):
    k = k / 10 # HACK
    return min(enumerate(array), key=lambda x: abs(x[1]-k))


################## MAIN ##################

def main():
    print("---------------------------------------------------------------------------- ")
    print("-------------------------------- SONG2GRAPH -------------------------------- ")
    print("---------------------------------------------------------------------------- ")
    # Load DB
    videos = read_library()

    for directory in ("processed", "library", "separated", "separated/htdemucs_6s"):
        os.makedirs(directory, exist_ok=True)
    
    # Parse command line input
    parser = argparse.ArgumentParser(description='song2graph')
    parser.add_argument('-a', '--add', help='youtube id', required=False)
    parser.add_argument('-r', '--remove', help='youtube id', required=False)
    parser.add_argument('-v', '--videos', help='video db length', required=False)
    parser.add_argument('-t', '--tempo', help='quantize audio tempo in BPM', required=False, type=float)
    parser.add_argument('-q', '--quantize', help='quantize: id or "all"', required=False)
    parser.add_argument('-k', '--quantizekeepbpm', help='quantize to the BPM of the original audio file"', required=False, action="store_true", default=False)
    parser.add_argument('-s', '--search', help='search for musically similar audio files, given a database id"', required=False)
    parser.add_argument('-sa', '--searchamount', help='amount of results the search returns"', required=False, type=int)
    parser.add_argument('-st', '--searchbpm', help='include BPM of audio files as similiarty search criteria"', required=False, action="store_true", default=False)
    parser.add_argument('-m', '--midi', help='extract midi from audio files"', required=False, action="store_true", default=False)
    parser.add_argument('--clap-index', help='build CLAP retrieval index for ids or "all"', required=False)
    parser.add_argument('--clap-similar', help='search CLAP neighbors for an indexed item id', required=False)
    parser.add_argument('--clap-text', help='search CLAP index using a text query', required=False)
    parser.add_argument('--clap-limit', help='max CLAP search results', required=False, type=int, default=10)
    parser.add_argument('--transcribe', help='transcribe lyrics for ids or "all"', required=False)
    parser.add_argument('--transcribe-model', help='faster-whisper model size', required=False, default='small')
    parser.add_argument('--transcribe-language', help='optional faster-whisper language hint', required=False)
    parser.add_argument('--export-doc', help='write structured song documents for ids or "all"', required=False)
    parser.add_argument('--annotate', help='run LLM annotation pass for ids or "all"', required=False)
    parser.add_argument('--annotate-model', help='LLM model for annotation', required=False, default=None)
    parser.add_argument('--ingest', help='run transcription, document export, and CLAP indexing for ids or "all"', required=False)
    parser.add_argument('--ingest-no-transcribe', help='skip transcription during --ingest', required=False, action="store_true", default=False)
    parser.add_argument('--ingest-annotate', help='run LLM annotation during --ingest', required=False, action="store_true", default=False)

    args = parser.parse_args()

    if (args.annotate is not None or args.ingest_annotate) and os.getenv("OPENROUTER_API_KEY") is None:
        parser.error("OPENROUTER_API_KEY is required for --annotate and --ingest-annotate.")

    # List of videos to use
    if args.videos is not None:
        finalvids = []
        vids = args.videos.split(",")
        print("process selected videos only:",vids)
        for vid in vids:
            v = [x for x in videos if x.id == vid][0]
            finalvids.append(v)
        videos = finalvids

    # List of videos to delete
    if args.remove is not None:
        print("remove video:",args.remove)
        for vid in videos:
            if vid.id == args.remove:
                videos.remove(vid)
                break
        write_library(videos)

    # List of videos to download
    newvids = []
    if args.add is not None:
        print("add video:",args.add,"to videos:",len(videos))
        vids = args.add.split(",")
        if "/" in args.add and not (args.add.endswith(".wav") or args.add.endswith(".mp3")):
            print('add directory with wav or mp3 files')
            videos = audio_directory_process(vids,videos)
        elif ".wav" in args.add or ".mp3" in args.add:
            print('add wav or mp3 file')
            videos = audio_process(vids,videos)
        else:
            videos = video_process(vids,videos)
        newvids = vids
    
    # List of audio to quantize
    vidargs = []
    if args.quantize is not None:
        vidargs = args.quantize.split(",")
        # print("Quantize:", vidargs)
        if vidargs[0] == 'all' and len(newvids) != 0:
            vidargs = newvids

    # MIDI
    extractmidi = bool(args.midi)
    transcribe_ids = args.transcribe.split(",") if args.transcribe is not None else []
    export_doc_ids = args.export_doc.split(",") if args.export_doc is not None else []
    annotate_ids = args.annotate.split(",") if args.annotate is not None else []
    ingest_ids = args.ingest.split(",") if args.ingest is not None else []

    # Tempo
    tempo = int(args.tempo or 120)

    # Quanize: Keep bpm of original audio file
    keepOriginalBpm = bool(args.quantizekeepbpm)

    # WIP: Quanize: Pitch shift before quanize
    pitchShiftFirst = False
    # if args.quantizepitchshift:
    #     pitchShiftFirst = True

    # Analyse to DB
    print(f"------------------------------ Files in DB: {len(videos)} ------------------------------")
    dump_db = False
    # get/detect audio metadata
    for vid in videos:
        feature_file = f"library/{vid.id}.a"
        # load features from disk
        if os.path.isfile(feature_file):
            with open(feature_file, "rb") as f:
                audio_features = pickle.load(f)
            if should_refresh_audio_features(audio_features):
                file = resolve_audio_file(vid)
                print('refresh audio features', vid.id, 'using torchcrepe')
                audio_features = get_audio_features(file=file, file_id=vid.id, extractMidi=extractmidi)
                with open(feature_file, "wb") as f:
                    pickle.dump(audio_features, f)
            if extractmidi:
                file = resolve_audio_file(vid)
                stem_output_dir, stem_paths = ensure_stems(file, vid.id, 'htdemucs_6s')
                ensure_midi_outputs(stem_paths, stem_output_dir)
        # extract features
        else:
            file = resolve_audio_file(vid)
            if len(vid.id) > 12:
                print('is audio', vid.id, vid.name, file)

            # Audio feature extraction
            audio_features = get_audio_features(file=file,file_id=vid.id, extractMidi=extractmidi)

            # Save to disk
            with open(feature_file, "wb") as f:
                pickle.dump(audio_features, f)
        
        # assign features to video
        vid.audio_features = audio_features
        print(
            vid.id,
            "tempo", round(float(scalarize(audio_features["tempo"])), 2),
            "duration", round(audio_features["duration"], 2),
            "timbre", round(audio_features["timbre"], 2),
            "pitch", round(audio_features["pitch"], 2),
            "intensity", round(audio_features["intensity"], 2),
            "segments", len(audio_features["segments_boundaries"]),
            "frequency", round(audio_features["frequency"], 2),
            "key", audio_features["key"],
            "name", vid.name,
        )
        #dump_db = True
        should_transcribe = args.transcribe is not None and ('all' in transcribe_ids or vid.id in transcribe_ids)
        transcription_payload = None
        if should_transcribe:
            transcription_payload, transcription_path = transcribe_lyrics(
                vid,
                model_size=args.transcribe_model,
                language=args.transcribe_language,
            )
            print("lyrics", transcription_path)

        should_export_doc = (
            args.export_doc is not None and ('all' in export_doc_ids or vid.id in export_doc_ids)
        ) or should_transcribe
        if should_export_doc:
            document_path = ensure_song_document(vid, transcription=transcription_payload)
            print("document", document_path)
        should_annotate = args.annotate is not None and ('all' in annotate_ids or vid.id in annotate_ids)
        if should_annotate:
            annotation_result, annotation_path, refreshed_document_path = annotate_song_document(
                vid,
                model=args.annotate_model,
            )
            print("annotation", annotation_path)
            print("document", refreshed_document_path)
            print("annotation summary", annotation_result["annotation"].get("summary"))
    if dump_db:
        write_library(videos)

    print("--------------------------------------------------------------------------")

    # Quantize audio
    if args.search is None:
        for vidarg in vidargs:
            for idx, vid in enumerate(videos):
                if vid.id == vidarg:
                    quantizeAudio(videos[idx], bpm=tempo, keepOriginalBpm = keepOriginalBpm, pitchShiftFirst = pitchShiftFirst, extractMidi = extractmidi)
                    break
                if vidarg == 'all' and len(newvids) == 0:
                    quantizeAudio(videos[idx], bpm=tempo, keepOriginalBpm = keepOriginalBpm, pitchShiftFirst = pitchShiftFirst, extractMidi = extractmidi)

    # Search
    searchamount = int(args.searchamount or 20)
    searchforbpm = bool(args.searchbpm)

    if args.search is not None:
        for vid in videos:
            if vid.id == args.search:
                query = vid
                print(
                    'Audio files related to:', query.id,
                    "- Key:", query.audio_features['key'],
                    "- Tempo:", int(float(scalarize(query.audio_features['tempo']))),
                    ' - ', query.name,
                )
                if args.quantize is not None:
                    quantizeAudio(query, bpm=tempo, keepOriginalBpm = keepOriginalBpm, pitchShiftFirst = pitchShiftFirst, extractMidi = extractmidi)
                i = 0
                while i < searchamount:
                    nearest = get_nearest(query, videos, tempo, searchforbpm)
                    query = nearest
                    print(
                        "- Relate:", query.id,
                        "- Key:", query.audio_features['key'],
                        "- Tempo:", int(query.audio_features['tempo']),
                        ' - ', query.name,
                    )
                    if args.quantize is not None:
                        quantizeAudio(query, bpm=tempo, keepOriginalBpm = keepOriginalBpm, pitchShiftFirst = pitchShiftFirst, extractMidi = extractmidi)
                    i += 1
                break

    if args.clap_index is not None:
        selected_ids = args.clap_index.split(",")
        metadata_path, embeddings_path, asset_count = build_clap_index(videos, selected_ids)
        print("CLAP index saved:", metadata_path)
        print("CLAP embeddings saved:", embeddings_path)
        print("CLAP indexed assets:", asset_count)

    if args.ingest is not None:
        ingest_results = ingest_assets(
            videos,
            ingest_ids,
            transcribe=not args.ingest_no_transcribe,
            transcription_model=args.transcribe_model,
            transcription_language=args.transcribe_language,
            annotate=args.ingest_annotate,
            annotation_model=args.annotate_model,
        )
        print("Ingest complete")
        print("CLAP index saved:", ingest_results["clap_index_metadata"])
        print("CLAP embeddings saved:", ingest_results["clap_index_embeddings"])
        print("CLAP indexed assets:", ingest_results["clap_asset_count"])

    if args.clap_similar is not None:
        results = search_clap_similar(args.clap_similar, args.clap_limit)
        print(format_results(results, header=f"CLAP similar to {args.clap_similar}"))

    if args.clap_text is not None:
        results = search_clap_text(args.clap_text, args.clap_limit)
        print(format_results(results, header=f'CLAP text search: "{args.clap_text}"'))

if __name__ == "__main__":
    main()
