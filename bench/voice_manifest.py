"""Validate explicitly supplied audio-turn diagnostics without provider access."""

import hashlib
import json
import math
from pathlib import Path
import re

from engine.config import ROOT, load_config
from engine.demo import demo_config


class ManifestError(ValueError):
    pass


def load_demo_domain(domain):
    """Validate a configured demo without allocating speech/network resources."""
    if not isinstance(domain, str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', domain):
        raise ManifestError('Domain must be a lowercase safe configured task ID.')
    try:
        config = load_config(ROOT / 'configs' / f'{domain}.json')
        if config['domain_id'] != domain:
            raise ManifestError('Domain filename must match its configured ID.')
        return demo_config(config)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ManifestError(f'Invalid demo domain {domain}: {exc}') from exc


def _reject_nonfinite(value):
    raise ManifestError('Non-finite JSON numbers are not allowed.')


def validate_turns(turns, timeout_seconds):
    """Also validate programmatic runner inputs before opening a connection."""
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or
            not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 180):
        raise ManifestError('Session timeout must be greater than zero and at most 180 seconds.')
    if not isinstance(turns, list) or not 1 <= len(turns) <= 5:
        raise ManifestError('Supply between 1 and 5 WAV turns.')
    total = 0
    for turn in turns:
        if (not isinstance(turn, dict) or not isinstance(turn.get('source_kind'), str) or
                turn['source_kind'] not in {'recorded', 'synthetic_diagnostic'}):
            raise ManifestError('Each turn requires source_kind recorded or synthetic_diagnostic.')
        pcm = turn.get('pcm')
        if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
            raise ManifestError('Each turn requires nonempty complete PCM16 samples.')
        seconds = len(pcm) / 32000
        if seconds > 20:
            raise ManifestError('Each WAV must be at most 20 seconds.')
        total += seconds
        if 'expected_action' in turn and (not isinstance(turn['expected_action'], str) or not turn['expected_action'].strip()):
            raise ManifestError('expected_action must be a nonempty action name.')
        if 'expected_slots' in turn and not isinstance(turn['expected_slots'], dict):
            raise ManifestError('expected_slots must be an object compared exactly with committed state.')
        if 'expected_slots' in turn:
            try:
                json.dumps(turn['expected_slots'], allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ManifestError('expected_slots must contain finite JSON values only.') from exc
    if total > 60:
        raise ManifestError('Combined source audio must be at most 60 seconds.')
    if total >= timeout_seconds:
        raise ManifestError('Source audio must be shorter than the session timeout.')
    return total


def load_manifest(path, *, timeout_seconds=180):
    """Resolve WAVs relative to the manifest and validate all before execution."""
    from bench.voice_transport import verified_wav

    path = Path(path).resolve(strict=True)
    if path.stat().st_size > 65536:
        raise ManifestError('Manifest must be at most 64 KiB.')
    try:
        data = json.loads(path.read_text(encoding='utf-8'), parse_constant=_reject_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError('Manifest must contain valid UTF-8 JSON.') from exc
    if not isinstance(data, dict) or set(data) != {'domain', 'turns'}:
        raise ManifestError('Manifest requires only domain and turns.')
    load_demo_domain(data['domain'])
    raw_turns = data['turns']
    if not isinstance(raw_turns, list) or not 1 <= len(raw_turns) <= 5:
        raise ManifestError('Supply between 1 and 5 WAV turns.')
    turns = []
    for item in raw_turns:
        if (not isinstance(item, dict) or not {'audio', 'source_kind'} <= set(item) or
                set(item) - {'audio', 'source_kind', 'expected_action', 'expected_slots'}):
            raise ManifestError('Each turn requires audio/source_kind and optional expected_action/expected_slots only.')
        if not isinstance(item['audio'], str) or not item['audio'].strip():
            raise ManifestError('audio must name an existing WAV file.')
        source = (path.parent / item['audio']).resolve(strict=True)
        if not source.is_file() or source.stat().st_size > 16 * 1024 * 1024:
            raise ManifestError('Source must be a WAV file at most 16 MiB.')
        raw = source.read_bytes()
        pcm, audio = verified_wav(raw)
        turn = {key: value for key, value in item.items() if key != 'audio'}
        turn.update(source=str(source), source_sha256=hashlib.sha256(raw).hexdigest(),
                    pcm=pcm, input_audio=audio)
        turns.append(turn)
    validate_turns(turns, timeout_seconds)
    return {'domain': data['domain'], 'turns': turns, 'timeout_seconds': timeout_seconds}
