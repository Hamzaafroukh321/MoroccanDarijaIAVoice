"""Transcribe one real kitchen recording with MoulSot only; no bank or gold labels."""

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import wave
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from engine.config import ROOT, load_config
from engine.stt import SpeechToText


def read_audio(path):
    with wave.open(str(path), 'rb') as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (16000, 1, 2, 'NONE'):
            raise ValueError('Use a 16 kHz mono PCM16 WAV.')
        frames = audio.getnframes()
        pcm = audio.readframes(frames)
        if frames == 0 or len(pcm) != frames * 2:
            raise ValueError('Recording is empty or truncated.')
        return pcm, frames / 16000


async def main():
    load_dotenv(ROOT/'.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path)
    parser.add_argument('--endpoint', default=os.getenv('MOULSOT_ENDPOINT', ''), help='Your duplicated Space URL.')
    parser.add_argument('--timeout-seconds', type=int, default=600)
    parser.add_argument('--check', action='store_true', help='Check file and setup without uploading audio.')
    args = parser.parse_args()
    try:
        if not args.endpoint.startswith('https://'):
            raise ValueError('Set MOULSOT_ENDPOINT to your HTTPS Space URL, or pass --endpoint.')
        if args.timeout_seconds <= 0:
            raise ValueError('Timeout must be positive.')
        pcm, seconds = read_audio(args.audio)
        config = load_config(ROOT/'configs/pizza.json')
        if len(pcm) + 44 > config['stt']['max_upload_bytes']:
            raise ValueError('Recording exceeds the configured upload limit.')
    except (ValueError, OSError, wave.Error, EOFError) as exc:
        print('NOT RUN:', exc)
        return 2
    print(f'Input: {seconds:.1f}s. Provider: MoulSot. Groq fallback: disabled.', flush=True)
    if args.check:
        return 0
    config['stt']['timeout_ms'] = args.timeout_seconds * 1000
    stt = SpeechToText(config, ROOT, primary='moulsot', endpoint=args.endpoint, fallback=False)
    output = ROOT/'bench/results'/f'moulsot_smoke_{uuid4().hex}.json'
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'audio_sha256': hashlib.sha256(args.audio.read_bytes()).hexdigest(),
              'audio_seconds': seconds, 'provider': 'moulsot', 'fallback': False, 'endpoint': args.endpoint,
              'human_verdict': 'pending', 'kind': 'qualitative_smoke_not_accuracy',
              'stt_config': config['stt'], 'adapter_sha256': hashlib.sha256((ROOT/'engine/stt.py').read_bytes()).hexdigest()}
    started = time.monotonic()
    try:
        result = await stt.transcribe(pcm)
        report.update(status='transcribed', result=asdict(result))
        print(result.text)
    except (RuntimeError, ValueError, OSError) as exc:
        report.update(status='failed', error=str(exc))
        print('MoulSot failed:', exc)
    finally:
        report.update(elapsed_ms=(time.monotonic()-started)*1000, calls=stt.calls)
        await stt.close()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print('Saved:', output)
    return 0 if report['status'] == 'transcribed' else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
