"""Measure supplied PCM WAVs and optionally replay local VAD, without providers."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bench.voice_transport import verified_wav
from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.endpointing import EndpointDetector, SileroVAD


def _dbfs(value):
    return 20 * math.log10(value) if value > 0 else None


def signal_metrics(pcm, sample_rate=16000, frame_ms=32):
    if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
        raise ValueError('Supply nonempty complete PCM16 samples.')
    if type(sample_rate) is not int or sample_rate <= 0 or type(frame_ms) is not int or frame_ms <= 0:
        raise ValueError('Sample rate and frame duration must be positive integers.')
    frame_samples = sample_rate * frame_ms // 1000
    if frame_samples < 1:
        raise ValueError('The measurement frame must include at least one sample.')
    signed = np.frombuffer(pcm, dtype='<i2')
    audio = signed.astype(np.float64) / 32768
    rms = float(np.sqrt(np.mean(audio ** 2)))
    frames = np.array([np.sqrt(np.mean(audio[index:index + frame_samples] ** 2))
                       for index in range(0, len(audio), frame_samples)])
    return {'samples': len(audio), 'duration_ms': len(audio) * 1000 / sample_rate,
            'rms_dbfs': _dbfs(rms), 'peak_dbfs': _dbfs(float(np.max(np.abs(audio)))),
            'exact_zero_fraction': float(np.mean(signed == 0)),
            'rail_fraction': float(np.mean((signed == -32768) | (signed == 32767))),
            'frame_rms_dbfs': {f'p{quantile}': _dbfs(float(np.percentile(frames, quantile)))
                               for quantile in (10, 50, 90)}}


def segment_audio(pcm, config, vad):
    signal_metrics(pcm)
    detector = EndpointDetector(config)
    frame_bytes = detector.frame_bytes
    vad.reset()
    segments = []
    speech_frames = 0

    def feed(frame, source):
        nonlocal speech_frames
        probability = vad(frame)
        if source and probability > config['endpointing']['vad_threshold']:
            speech_frames += 1
        segment = detector.feed(frame, probability)
        if segment is not None:
            segments.append({key: getattr(segment, key) for key in
                             ('start_ms', 'speech_end_ms', 'endpoint_ms', 'speech_ms', 'forced')})
            segments[-1]['uploaded_samples'] = len(segment.pcm) // config['runtime']['sample_width_bytes']

    for index in range(0, len(pcm), frame_bytes):
        feed(pcm[index:index + frame_bytes].ljust(frame_bytes, b'\0'), True)
    tail_frames = math.ceil(max(config['endpointing']['max_wait_ms'],
                               config['endpointing']['base_silence_ms']) / detector.frame_ms) + 2
    for _ in range(tail_frames):
        feed(bytes(frame_bytes), False)
    return {'segments': segments, 'source_end_ms': len(pcm) / 2 * 1000 / config['runtime']['sample_rate_hz'],
            'vad_speech_frames': speech_frames, 'appended_silence_ms': tail_frames * detector.frame_ms,
            'final_frame_padding_samples': (-len(pcm) % frame_bytes) // 2,
            'unfinished_buffer_ms': len(detector.buffer) / frame_bytes * detector.frame_ms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', nargs='+', type=Path, help='One to five mono16kHz PCM16 WAVs, total at most180seconds.')
    parser.add_argument('--vad', action='store_true', help='Run installed local Silero and demo endpointing; no ASR/router/TTS.')
    parser.add_argument('--domain', choices=['pizza', 'clinic'], default='pizza')
    parser.add_argument('--output', type=Path, help='Optional JSON report path; original WAVs are never changed.')
    args = parser.parse_args()
    try:
        if not 1 <= len(args.audio) <= 5:
            raise ValueError('Supply one to five recordings.')
        inputs = []
        for path in args.audio:
            path = path.resolve(strict=True)
            if path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError('Each input must be at most16MiB.')
            raw = path.read_bytes()
            pcm, metadata = verified_wav(raw)
            inputs.append((path, pcm, metadata, hashlib.sha256(raw).hexdigest()))
        if sum(item[2]['duration_seconds'] for item in inputs) > 180:
            raise ValueError('Combined source duration must be at most180seconds.')
        if args.output and args.output.resolve() in {item[0] for item in inputs}:
            raise ValueError('The output report must not overwrite an input WAV.')
        config = demo_config(load_config(ROOT / 'configs' / f'{args.domain}.json')) if args.vad else None
        vad = SileroVAD(config) if args.vad else None
        report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'domain': args.domain,
                  'purpose': 'offline signal and endpoint diagnostic', 'evaluation_eligible': False,
                  'provider_requests': 0, 'speech_recognition_used': False, 'local_vad_used': args.vad,
                  'method': 'PCM16 full scale32768; exact signed rails;32ms frame RMS quantiles;null dBFS means zero.',
                  'limits': 'No intelligibility, native accuracy, calibrated microphone level or optimal endpoint claim.',
                  'files': []}
        for path, pcm, metadata, digest in inputs:
            item = {'source': str(path), 'source_sha256': digest, 'audio': metadata,
                    'signal': signal_metrics(pcm)}
            if vad is not None:
                item['endpointing'] = segment_audio(pcm, config, vad)
            report['files'].append(item)
        if config:
            report['endpoint_settings'] = config['endpointing']
        encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding='utf-8')
            print(json.dumps({'status': 'inspected', 'files': len(inputs), 'provider_requests': 0,
                              'local_vad_used': args.vad, 'report': str(args.output.resolve())}))
        else:
            print(encoded)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'NOT INSPECTED: {exc}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
