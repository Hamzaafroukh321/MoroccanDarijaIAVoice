# Optional local profiling artifact; not deployed.
# Baseline app.py SHA256: 32f09e6fd474ea82d32386fb0910307b276b6935b4d2f52bb826c695843d118d
# coding=utf-8
# Copyright 2026 AtlasIA Team.
# SPDX-License-Identifier: Apache-2.0
"""
Moulsot ASR Demo for Huggingface Spaces with ZeroGPU support.
Darija-first speech recognition powered by Qwen3-ASR fine-tuning.
"""

import base64
import io
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import spaces  # Must precede Gradio, Torch and every CUDA-related import.

import gradio as gr
import numpy as np
import torch
from scipy.io.wavfile import write as wav_write


def _title_case_display(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])


def _build_choices_and_map(
    items: Optional[List[str]],
) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping


def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    """
    Accept gradio audio:
      - {"sampling_rate": int, "data": np.ndarray}
      - (sr, np.ndarray)  [some gradio versions]
    Return: (wav_float32_mono, sr)
    """
    if audio is None:
        return None

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    if isinstance(audio, tuple) and len(audio) == 2:
        a0, a1 = audio
        if isinstance(a0, int):
            sr = int(a0)
            wav = _normalize_audio(a1)
            return wav, sr
        if isinstance(a1, int):
            wav = _normalize_audio(a0)
            sr = int(a1)
            return wav, sr

    return None


def _parse_audio_any(audio: Any) -> Union[str, Tuple[np.ndarray, int]]:
    if audio is None:
        raise ValueError("Audio is required.")
    at = _audio_to_tuple(audio)
    if at is not None:
        return at
    raise ValueError("Unsupported audio input format.")


def _make_timestamp_html(audio_upload: Any, timestamps: Any) -> str:
    """
    Build HTML with per-token audio slices, using base64 data URLs.
    """
    at = _audio_to_tuple(audio_upload)
    if at is None:
        return "<div style='color:#666'>No audio available for visualization.</div>"
    audio, sr = at

    if not timestamps:
        return "<div style='color:#666'>No timestamps to visualize.</div>"
    if not isinstance(timestamps, list):
        return "<div style='color:#666'>Invalid timestamp format.</div>"

    html_content = """
    <style>
        .word-alignment-container { display: flex; flex-wrap: wrap; gap: 10px; }
        .word-box {
            border: 1px solid #ddd; border-radius: 8px; padding: 10px;
            background-color: #f9f9f9; box-shadow: 0 2px 4px rgba(0,0,0,0.06);
            text-align: center;
        }
        .word-text { font-size: 18px; font-weight: 700; margin-bottom: 5px; }
        .word-time { font-size: 12px; color: #666; margin-bottom: 8px; }
        .word-audio audio { width: 140px; height: 30px; }
        details { border: 1px solid #ddd; border-radius: 6px; padding: 10px; background-color: #f7f7f7; }
        summary { font-weight: 700; cursor: pointer; }
    </style>
    """

    html_content += """
    <details open>
        <summary>Timestamps Visualization (click each word to hear the audio segment)</summary>
        <div class="word-alignment-container" style="margin-top: 14px;">
    """

    for item in timestamps:
        if not isinstance(item, dict):
            continue
        word = str(item.get("text", "") or "")
        start = item.get("start_time", None)
        end = item.get("end_time", None)
        if start is None or end is None:
            continue

        start = float(start)
        end = float(end)
        if end <= start:
            continue

        start_sample = max(0, int(start * sr))
        end_sample = min(len(audio), int(end * sr))
        if end_sample <= start_sample:
            continue

        seg = audio[start_sample:end_sample]
        seg_i16 = (np.clip(seg, -1.0, 1.0) * 32767.0).astype(np.int16)

        mem = io.BytesIO()
        wav_write(mem, sr, seg_i16)
        mem.seek(0)
        b64 = base64.b64encode(mem.read()).decode("utf-8")
        audio_src = f"data:audio/wav;base64,{b64}"

        html_content += f"""
        <div class="word-box">
            <div class="word-text">{word}</div>
            <div class="word-time">{start:.3f}s - {end:.3f}s</div>
            <div class="word-audio">
                <audio controls preload="none" src="{audio_src}"></audio>
            </div>
        </div>
        """

    html_content += "</div></details>"
    return html_content


from qwen_asr import Qwen3ASRModel

asr = Qwen3ASRModel.from_pretrained(
    "atlasia/moulsot.v0.3",
    dtype=torch.bfloat16,
    device_map="cuda",  # ZeroGPU emulates CUDA while loading at startup.
    max_inference_batch_size=16,
)


# Supported languages — Darija listed first as the primary target
SUPPORTED_LANGUAGES = [
    "Arabic",
    "Chinese",
    "Cantonese",
    "English",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
]

lang_choices_disp, lang_map = _build_choices_and_map(SUPPORTED_LANGUAGES)
lang_choices = ["Auto (Darija / Arabic)"] + lang_choices_disp
SPACE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_AUDIO_PATH = os.path.join(SPACE_DIR, "audio.wav")
examples = [
    [EXAMPLE_AUDIO_PATH],
]


# Optional profiling uses host wall time; it does not measure GPU kernel time.
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import time
from uuid import uuid4


def _profile_emit(record):
    try:
        print(json.dumps(record, sort_keys=True), flush=True)
    except Exception:
        pass  # Logging must never change the existing API result or exception.


@contextmanager
def _profile_transcription():
    if os.getenv("MOULSOT_PROFILE") != "1":
        yield None
        return
    started = time.perf_counter()
    record = {"profile_id": uuid4().hex,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "outcome": "ok", "parse_ms": None, "model_call_wall_ms": None,
              "sample_rate_hz": None, "samples": None, "audio_duration_ms": None,
              "error_type": None, "elapsed_clock": "perf_counter"}
    _profile_emit({"event": "moulsot_profile_start", "profile_id": record["profile_id"],
                   "started_utc": record["started_utc"]})
    try:
        yield record
    except BaseException as exc:
        record["outcome"] = "error"
        record["error_type"] = type(exc).__name__
        raise
    finally:
        record["body_ms"] = (time.perf_counter() - started) * 1000
        _profile_emit({"event": "moulsot_profile_end", **record})


@contextmanager
def _profile_phase(record, name):
    if record is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        record[name] = (time.perf_counter() - started) * 1000


def _profile_input(record, audio_obj):
    if record is not None:
        try:
            samples, sample_rate = len(audio_obj[0]), int(audio_obj[1])
            duration = samples * 1000 / sample_rate
        except Exception:
            return  # Optional metadata must not change valid inference behavior.
        record.update(samples=samples, sample_rate_hz=sample_rate, audio_duration_ms=duration)


def _profile_outcome(record, outcome):
    if record is not None:
        record["outcome"] = outcome


@spaces.GPU(duration=60)
def transcribe(
    audio_upload: Any,
    lang_disp: str,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Main transcription function with ZeroGPU support.
    """
    with _profile_transcription() as profile:
        if audio_upload is None:
            _profile_outcome(profile, "invalid_input")
            return (
                "",
                "",
                None,
                "<div style='color:#666'>Please upload an audio file first.</div>",
            )

        try:
            with _profile_phase(profile, "parse_ms"):
                audio_obj = _parse_audio_any(audio_upload)
            _profile_input(profile, audio_obj)
        except ValueError as e:
            _profile_outcome(profile, "invalid_input")
            return "", "", None, f"<div style='color:red'>Error: {str(e)}</div>"

        language = None
        if lang_disp and lang_disp not in ("Auto (Darija / Arabic)", "Auto"):
            language = lang_map.get(lang_disp, lang_disp)

        # Perform transcription
        with _profile_phase(profile, "model_call_wall_ms"):
            results = asr.transcribe(audio=audio_obj, language=language)

        if not isinstance(results, list) or len(results) != 1:
            _profile_outcome(profile, "invalid_result")
            return "", "", None, "<div style='color:red'>Unexpected result format.</div>"

        r = results[0]

        return getattr(r, "text", "") or ""


# ── Build Gradio interface ────────────────────────────────────────────────────

theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="amber",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "Menlo", "monospace"],
)

css = """
.gradio-container {
    max-width: none !important;
    min-height: 100vh;
    background:
        radial-gradient(circle at 20% 12%, rgba(212, 175, 55, 0.18), transparent 26%),
        radial-gradient(circle at 86% 18%, rgba(13, 148, 136, 0.18), transparent 24%),
        linear-gradient(135deg, #fff8ec 0%, #f5efe2 45%, #eef7f3 100%);
}
.main {
    max-width: 1120px !important;
    margin: 0 auto;
}
.moroccan-hero {
    position: relative;
    overflow: hidden;
    text-align: center;
    padding: 28px 26px 22px;
    border: 1px solid rgba(126, 38, 38, 0.18);
    border-radius: 8px;
    background:
        linear-gradient(135deg, rgba(126, 38, 38, 0.93), rgba(12, 96, 89, 0.92)),
        repeating-linear-gradient(45deg, rgba(255,255,255,0.14) 0 2px, transparent 2px 22px);
    color: #fffaf0;
    box-shadow: 0 24px 60px rgba(44, 24, 16, 0.16);
}
.moroccan-hero:before {
    content: "";
    position: absolute;
    inset: 10px;
    border: 1px solid rgba(246, 216, 134, 0.35);
    border-radius: 6px;
    pointer-events: none;
}
.atlasia-badge {
    display: inline-block;
    background: rgba(255, 250, 240, 0.14);
    color: #f6d886;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.14em;
    padding: 5px 12px;
    border-radius: 20px;
    border: 1px solid rgba(246, 216, 134, 0.45);
    text-transform: uppercase;
    margin-bottom: 10px;
}
.moroccan-hero img {
    width: 128px;
    filter: drop-shadow(0 10px 18px rgba(0, 0, 0, 0.22));
}
.hero-title {
    margin: 12px auto 6px;
    font-size: clamp(2rem, 5vw, 4.4rem);
    line-height: 0.95;
    font-weight: 900;
    letter-spacing: 0;
}
.hero-subtitle {
    color: #fff0c7;
    max-width: 700px;
    margin: 0 auto;
    font-size: 1.08rem;
}
.zellige-strip {
    height: 12px;
    margin: 18px auto 0;
    max-width: 560px;
    border-radius: 999px;
    background:
        repeating-linear-gradient(90deg, #c0392b 0 42px, #0f766e 42px 84px, #d4af37 84px 126px, #17324d 126px 168px);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.28);
}
.feature-band {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin: 18px 0 20px;
}
.feature-tile {
    border-radius: 8px;
    border: 1px solid rgba(15, 118, 110, 0.17);
    background: rgba(255, 250, 240, 0.76);
    padding: 14px;
    color: #3b2a20;
    box-shadow: 0 12px 30px rgba(55, 35, 20, 0.08);
}
.feature-tile strong {
    display: block;
    color: #7e2626;
    font-size: 0.95rem;
    margin-bottom: 3px;
}
.feature-tile span {
    color: #5b4b3d;
    font-size: 0.92rem;
}
.input-panel, .output-panel {
    border: 1px solid rgba(126, 38, 38, 0.15);
    border-radius: 8px;
    padding: 16px;
    background:
        linear-gradient(180deg, rgba(255,255,255,0.82), rgba(255,250,240,0.9)),
        linear-gradient(45deg, rgba(212,175,55,0.08) 25%, transparent 25% 75%, rgba(212,175,55,0.08) 75%);
    box-shadow: 0 16px 40px rgba(55, 35, 20, 0.09);
}
.input-panel,
.input-panel label,
.input-panel span,
.input-panel p,
.input-panel td,
.input-panel th {
    color: #3b2a20 !important;
}
.input-panel .examples,
.input-panel .examples * {
    color: #3b2a20 !important;
}
#moroccan-audio-examples,
#moroccan-audio-examples label,
#moroccan-audio-examples .label-wrap,
#moroccan-audio-examples .label-wrap span,
#moroccan-audio-examples .label,
#moroccan-audio-examples span {
    color: #3b2a20 !important;
}
.examples-title {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 10px 0 6px;
    color: #3b2a20 !important;
    font-size: 0.92rem;
    font-weight: 800;
}
.examples-title:before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: #d4af37;
    box-shadow: 10px 0 0 #0f766e, 20px 0 0 #b8322b;
    margin-right: 18px;
}
.input-panel .examples {
    border-radius: 8px;
    border: 1px solid rgba(15, 118, 110, 0.16);
    background: rgba(255, 250, 240, 0.72);
    padding: 10px;
}
.input-panel .examples button,
.input-panel table button {
    color: #7e2626 !important;
    background: #fffaf0 !important;
    border: 1px solid rgba(212, 175, 55, 0.55) !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}
.input-panel .examples button:hover,
.input-panel table button:hover {
    color: #0f766e !important;
    border-color: rgba(15, 118, 110, 0.5) !important;
}
button.primary {
    background: linear-gradient(135deg, #b8322b, #0f766e) !important;
    border: 1px solid rgba(246, 216, 134, 0.7) !important;
    box-shadow: 0 12px 24px rgba(126, 38, 38, 0.18) !important;
}
button.primary:hover {
    filter: brightness(1.04);
}
textarea, input, select {
    border-radius: 8px !important;
}
.footer-wrap {
    text-align:center;
    font-size:0.9em;
    color:#6d5b49;
    margin-top: 18px;
}
.footer-wrap a {
    color: #0f766e;
    font-weight: 700;
}
#transcription-output textarea,
#transcription-output .prose,
#transcription-output input {
    direction: rtl !important;
    text-align: right !important;
    unicode-bidi: plaintext !important;
}
@media (max-width: 720px) {
    .moroccan-hero {
        padding: 22px 16px 18px;
    }
    .feature-band {
        grid-template-columns: 1fr;
    }
}
"""

with gr.Blocks(title="Moulsot ASR — AtlasIA") as demo:
    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown(
        """
<div class="moroccan-hero">
  <span class="atlasia-badge">AtlasIA</span><br>
  <div align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/65f5c3528fb2b1535728138f/ZcWPJ5cxaX-bh14-0wTVG.png" alt="AtlasIA" />
  </div>
  <h1 class="hero-title">Moulsot</h1>
  <p class="hero-subtitle">Moroccan Darija speech recognition with a warm Atlas palette, built for everyday voices, code switching, and clear transcripts.</p>
  <div class="zellige-strip"></div>
</div>

<div class="feature-band">
  <div class="feature-tile"><strong>Darija First</strong><span>Optimized for Moroccan Arabic and local expression.</span></div>
  <div class="feature-tile"><strong>Code Switching</strong><span>Handles natural jumps between Arabic, French, and more.</span></div>
  <div class="feature-tile"><strong>Global Reach</strong><span>Supports 30+ additional languages when needed.</span></div>
</div>
"""
    )
    # ── Main layout ───────────────────────────────────────────────────────────
    with gr.Row():
        # Left column — inputs
        with gr.Column(scale=2, elem_classes=["input-panel"]):
            audio_in = gr.Audio(
                label="Upload or Record Audio / Sijjel Sawtek",
                type="numpy",
                sources=["upload", "microphone"],
            )
            gr.HTML('<div class="examples-title">Try a Moroccan Audio Sample</div>')
            gr.Examples(
                examples=examples,
                inputs=[audio_in],
                label=None,
                example_labels=["audio.wav"],
                elem_id="moroccan-audio-examples",
                cache_examples=False,
            )
            lang_in = gr.Dropdown(
                label="Language / Lougha",
                choices=lang_choices,
                value="Auto (Darija / Arabic)",
                interactive=True,
            )
            btn = gr.Button("Transcribe / Kteb", variant="primary", size="lg")

        # Right column — outputs
        with gr.Column(scale=2, elem_classes=["output-panel"]):
            out_text = gr.Textbox(
                label="Transcription / Nss Mktoub",
                lines=10,
                interactive=False,
                elem_id="transcription-output",
                text_align="right",
                rtl=True,
            )

    # ── Event handlers ────────────────────────────────────────────────────────
    btn.click(
        transcribe,
        inputs=[audio_in, lang_in],
        outputs=[out_text],
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.Markdown(
        """
<div class="footer-wrap">
  Built by <a href="https://huggingface.co/atlasia" target="_blank">AtlasIA</a> •
  Model: <a href="https://huggingface.co/atlasia/moulsot.v0.3" target="_blank">atlasia/moulsot.v0.3</a> •
  Base: <a href="https://huggingface.co/collections/Qwen/qwen3-asr" target="_blank">Qwen3-ASR</a>
</div>
"""
    )


if __name__ == "__main__":
    demo.launch(ssr_mode=False, theme=theme, css=css)
