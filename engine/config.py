"""Load domain configuration and validate it before starting a session."""

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft7Validator
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from engine.moulsot_context import configured_context


ROOT = Path(__file__).resolve().parent.parent


class ConfigError(ValueError):
    """An actionable configuration error, safe to show on the command line."""


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    host: str = Field(min_length=1)
    port: int = Field(gt=0, le=65535)
    sample_rate_hz: Literal[16000]
    channels: Literal[1]
    sample_width_bytes: Literal[2]
    frame_ms: Literal[32]
    max_recording_ms: int = Field(gt=0)
    connection_timeout_ms: int = Field(gt=0)
    stop_timeout_ms: int = Field(gt=0)
    idle_timeout_ms: int = Field(gt=0)
    max_buffered_frames: int = Field(gt=0)
    recordings_dir: Literal["bench/recordings"]
    results_dir: Literal["bench/results"]
    websocket_path: Literal["/ws"]
    echo_cancellation: bool
    noise_suppression: bool
    auto_gain_control: bool


class EndpointConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    base_silence_ms: int = Field(ge=0)
    hesitation_hold_ms: int = Field(ge=0)
    continuation_hold_ms: int = Field(ge=0)
    digit_hold_ms: int = Field(ge=0)
    max_wait_ms: int = Field(gt=0)
    min_speech_duration_ms: int = Field(gt=0)
    vad_threshold: float = Field(ge=0,le=1,allow_inf_nan=False)
    tail_token_count: int = Field(gt=0)
    digit_complete_lengths: list[int] = Field(min_length=1)
    max_segment_ms: int = Field(gt=0)
    partial_timeout_ms: int = Field(gt=0)


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    barge_in_frames: int = Field(gt=0)
    playback_ack_timeout_ms: int = Field(gt=0)
    max_session_ms: int = Field(gt=0)
    torch_threads: int = Field(gt=0)
    max_pending_segments: int = Field(gt=0)
    transcript_cache_entries: int = Field(gt=0)


def load_config(path: str | Path) -> dict[str, Any]:
    """Validate the literal spec schema, then cross-field engine invariants.

    Draft Darija text is permitted here; the audio-bank recorder must refuse it.
    Runtime settings are an extension allowed by the supplied JSON Schema.
    """
    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        schema = json.loads((ROOT / "configs/schema.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot load config/schema: {exc}") from exc

    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(config), key=lambda error: str(error.path))
    if errors:
        details = "; ".join(
            f"{'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ConfigError(f"{path.name}: {details}")

    slots = config["slots"]
    if not re.fullmatch(r'[a-z][a-z0-9_-]*',config['domain_id']):
        raise ConfigError('domain_id must be a lowercase safe identifier.')
    identifiers = [slot["id"] for slot in slots]
    if len(identifiers) != len(set(identifiers)):
        raise ConfigError("Slot ids must be unique.")
    for slot in slots:
        if type(slot.get('ask_order')) is not int:
            raise ConfigError(f"{slot['id']}: ask_order is required.")
        if slot["type"] in {"enum", "enum_list"} and not slot.get("values"):
            raise ConfigError(f"{slot['id']}: enum slots require nonempty values.")
        if "min" in slot and "max" in slot and slot["min"] > slot["max"]:
            raise ConfigError(f"{slot['id']}: min must not exceed max.")

    endpointing = config["endpointing"]
    if not 0 <= endpointing["vad_threshold"] <= 1:
        raise ConfigError("endpointing.vad_threshold must be between 0 and 1.")
    if any(value < 0 for key, value in endpointing.items() if key.endswith("_ms")):
        raise ConfigError("Endpointing durations must be nonnegative.")
    if endpointing["max_wait_ms"] < endpointing["base_silence_ms"]:
        raise ConfigError("max_wait_ms must be at least base_silence_ms.")
    overlap = config["overlap"]
    if not 0 <= overlap["confidence_floor"] <= 1:
        raise ConfigError("overlap.confidence_floor must be between 0 and 1.")
    if any(overlap[key] < 0 for key in ("min_tokens_per_second", "rms_std_ceiling", "max_clarifies")):
        raise ConfigError("Overlap thresholds must be nonnegative.")
    try:
        runtime = RuntimeConfig.model_validate(config.get("runtime", {}))
        EndpointConfig.model_validate(config.get('endpointing',{}))
        EngineConfig.model_validate(config.get('engine',{}))
        if runtime.max_recording_ms < runtime.frame_ms:
            raise ConfigError("max_recording_ms must allow at least one frame.")
    except ValidationError as exc:
        raise ConfigError(f"Invalid runtime settings: {exc}") from exc
    for section in ('stt','router','normalization','audio_output','benchmark'):
        if not isinstance(config.get(section),dict): raise ConfigError(f'Missing {section} settings.')
    positive={
        'stt':['timeout_ms','max_upload_bytes'],
        'router':['timeout_ms','max_tokens'],
        'audio_output':['number_max','tts_timeout_ms','recording_poll_ms'],
        'benchmark':['expected_scenarios','selection_max_mel_ms','plot_width','plot_height','plot_margin','tail_silence_ms','drain_timeout_ms'],
    }
    for section,keys in positive.items():
        for key in keys:
            value=config[section].get(key)
            if type(value) is not int or value<=0: raise ConfigError(f'{section}.{key} must be a positive integer.')
    if type(config['audio_output'].get('gap_ms')) is not int or config['audio_output']['gap_ms']<0: raise ConfigError('audio_output.gap_ms must be nonnegative.')
    if config['router'].get('retries')!=1: raise ConfigError('Router retries must equal one, as specified.')
    if config['stt'].get('moulsot_protocol') not in {'json','gradio'}: raise ConfigError('MoulSot protocol must be json or gradio.')
    try:
        configured_context(config['stt'])
    except ValueError as exc:
        raise ConfigError('Invalid stt.moulsot_context: ' + str(exc)) from exc
    if config['overlap'].get('required_signals')!=2: raise ConfigError('Overlap requires exactly two signals.')
    if not config['stt'].get('limits'): raise ConfigError('Groq quota windows are required.')
    for limit in config['stt']['limits']:
        if not isinstance(limit,dict) or not {'window_seconds'} <= limit.keys() or not any(key in limit for key in ('requests','audio_seconds')):
            raise ConfigError('Invalid Groq quota window.')
        if any(type(value) is not int or value<=0 for value in limit.values()): raise ConfigError('Quota limits must be positive integers.')
    if config['benchmark']['tail_silence_ms'] % runtime.frame_ms: raise ConfigError('Benchmark tail silence must contain whole frames.')
    if os.getenv('GROQ_ROUTER_MODEL', '').strip():
        config['router']['model'] = os.environ['GROQ_ROUTER_MODEL'].strip()
    return config
