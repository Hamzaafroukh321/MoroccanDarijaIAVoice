"""Frozen local MoulSot vocabulary experiment; dry by default.

No downloads, generation of input audio, Groq, TTS, restarts or provider changes.
Execution uses two fixed synthetic WAVs and at most four sequential loopback ASR
requests. A is today's no-system production framing; B adds a symmetric system
vocabulary. That framing change is part of the intervention, not hidden context.
This diagnostic is not native-reviewed evaluation and never rewrites output.
"""
import argparse
import asyncio
import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import time
from uuid import uuid4
import wave

import httpx

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = 'http://127.0.0.1:8011'
MODEL = 'moulsot.v0.3.Q4_K_M.gguf'
BUILD = 'b10809'
CONTEXT = 'الطبيب ألف، الطبيب باء'
LABELS = ('الطبيب ألف', 'الطبيب باء')
SENTINEL = 'CONTEXT_FORMAT_ONLY_67D89C'
PREFIX = 'language Arabic<asr_text>'
MAX_RESPONSE = 65536
SOURCES = {
    'positive': dict(path='bench/results/clinic_audio_turn_20260905_223813_synthetic_input.wav',
        sha256='632b20f0deebe121b97321b02df18f9539ab958c3f5364d1d820cbb05a9f8341',
        text='بغيت الطبيب ألف مع العشرة ولا الطبيب باء مع الحداش.',
        evidence='bench/results/clinic_audio_turn_20260905_223813.json'),
    'negative': dict(path='bench/results/clinic_sequence_20260905/turn_2_synthetic.wav',
        sha256='3bd55e3191f458d4aca45adc753e9b434658e4bd45b042ba9e6e169e7bac1ad7',
        text='لا، بدل الساعة، خليها مع الحداش.',
        evidence='bench/results/clinic_sequence_20260905/provenance.json'),
}


def frozen_audio(root, source):
    path = root / source['path']
    if not path.is_file() or path.stat().st_size > 20 * 32000 + 65536:
        raise ValueError('Frozen input is missing or exceeds the WAV limit.')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != source['sha256']:
        raise ValueError('Frozen input SHA256 does not match.')
    if (len(raw) < 44 or raw[:4] != b'RIFF' or raw[8:12] != b'WAVE' or
            int.from_bytes(raw[4:8], 'little') + 8 != len(raw)):
        raise ValueError('Expected a complete RIFF WAV input.')
    with wave.open(io.BytesIO(raw), 'rb') as audio:
        if (audio.getnchannels(), audio.getframerate(), audio.getsampwidth(), audio.getcomptype()) != (1, 16000, 2, 'NONE'):
            raise ValueError('Expected mono 16000 Hz PCM16 audio.')
        frames = audio.getnframes()
        if not 0 < frames <= 320000 or len(audio.readframes(frames)) != frames * 2:
            raise ValueError('Frozen input has invalid duration or incomplete samples.')
    return raw, frames / 16000


def audio_payload(raw, context=None):
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'a'},
        {'type': 'input_audio', 'input_audio': {
            'data': base64.b64encode(raw).decode('ascii'), 'format': 'wav'}}]}]
    if context is not None:
        messages.insert(0, {'role': 'system', 'content': context})
    return dict(messages=messages, max_tokens=512, temperature=0, cache_prompt=False, stream=False)


async def bounded_json(client, method, path, payload=None):
    async with asyncio.timeout(30):
        async with client.stream(method, path, json=payload) as response:
            # No raw error body/headers: successful inference bodies alone are
            # evidence. Redirects remain disabled and no credentials are loaded.
            if response.status_code != 200:
                raise ValueError(f'Loopback request failed with HTTP {response.status_code}.')
            raw = bytearray()
            async for chunk in response.aiter_bytes():
                if len(raw) + len(chunk) > MAX_RESPONSE:
                    raise ValueError('Loopback response exceeds the byte limit.')
                raw.extend(chunk)
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError('Expected a loopback JSON object.')
    return result


async def format_gate(client):
    props = await bounded_json(client, 'GET', '/props')
    if (not isinstance(props.get('build_info'), str) or
            re.match(r'^b10809(?:[- (]|$)', props['build_info']) is None or
            str(props.get('model_path', '')).replace('\\', '/').rsplit('/', 1)[-1] != MODEL):
        raise ValueError('Unexpected loopback model or runtime identity.')
    user = {'role': 'user', 'content': 'a'}
    baseline = await bounded_json(client, 'POST', '/apply-template', {'messages': [user]})
    candidate = await bounded_json(client, 'POST', '/apply-template', {
        'messages': [{'role': 'system', 'content': SENTINEL}, user]})
    prefix = '<|im_start|>system\n' + SENTINEL + '<|im_end|>\n'
    prompt = baseline.get('prompt')
    if (not isinstance(prompt, str) or '<|im_start|>user\na<|im_end|>' not in prompt or
            not prompt.endswith('<|im_start|>assistant\n') or
            candidate.get('prompt') != prefix + prompt):
        raise ValueError('System-context formatting changed or omitted the expected ChatML framing.')
    return dict(build_info=props['build_info'], model_path=props['model_path'],
        chat_template=props.get('chat_template'), baseline_prompt=prompt,
        sentinel_prompt=candidate['prompt'], inference_performed=False,
        limit='Text-only framing gate; audio framing remains the unchanged production payload.')


def transcription(payload):
    if str(payload.get('model', '')).replace('\\', '/').rsplit('/', 1)[-1] != MODEL:
        raise ValueError('Inference response model identity changed.')
    choices = payload.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError('Expected one transcription choice.')
    choice = choices[0]
    message = choice.get('message')
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Transcription was truncated or did not finish normally.')
    if (not isinstance(message, dict) or message.get('role') != 'assistant' or
            message.get('tool_calls') or message.get('function_call')):
        raise ValueError('Expected an assistant transcription response.')
    raw = message.get('content')
    if not isinstance(raw, str) or not raw.startswith(PREFIX):
        raise ValueError('Expected the exact Arabic ASR output prefix.')
    text = raw[len(PREFIX):]
    if re.search(r'[<>\x00-\x08\x0b\x0c\x0e-\x1f]', text):
        raise ValueError('Unexpected model marker or control character.')
    tokens = text.split()
    if any(tokens[i:i+4] == tokens[i+4:i+8] == tokens[i+8:i+12]
           for i in range(max(0, len(tokens) - 11))):
        raise ValueError('Repeated output detected; stop without another arm.')
    if any(text.count(label) > 1 for label in LABELS):
        raise ValueError('Repeated vocabulary labels detected.')
    return text


def improves_labels(baseline, candidate):
    """Gate only; exact literal label presence is not an accuracy score."""
    return all(label in candidate for label in LABELS) and not all(label in baseline for label in LABELS)


async def run_probe(client, audio, report):
    report['format_gate'] = await format_gate(client)
    for source_name in ('positive', 'negative'):
        results = []
        for arm, context in (('A', None), ('B', CONTEXT)):
            if len(report['calls']) >= 4:
                raise ValueError('ASR call ceiling reached.')
            record = dict(source=source_name, arm=arm, context=context,
                context_sha256=hashlib.sha256((context or '').encode()).hexdigest(),
                status='started')
            report['calls'].append(record)
            started = time.perf_counter()
            try:
                payload = await bounded_json(client, 'POST', '/v1/chat/completions',
                    audio_payload(audio[source_name], context))
                record['raw_response'] = payload
                record['text'] = transcription(payload)
                record['usage'] = payload.get('usage')
                record['status'] = 'completed'
                results.append(record['text'])
            except Exception as exc:
                record.update(status='error', error_type=type(exc).__name__)
                raise
            finally:
                record['elapsed_ms'] = round((time.perf_counter() - started) * 1000, 3)
        if source_name == 'positive':
            report['positive_exact_label_improvement'] = improves_labels(*results)
            if not report['positive_exact_label_improvement']:
                report['status'] = 'stopped_no_exact_label_improvement'
                return
        else:
            report['negative_output_unchanged'] = results[0] == results[1]
            report['negative_vocabulary_inserted'] = any(
                term in text for text in results for term in (*LABELS, 'الطبيب', 'ألف', 'باء'))
    report['status'] = 'completed_needs_listening_review'


async def main(execute=False):
    report = dict(purpose=__doc__, evaluation_eligible=False, native_review='pending',
        context=CONTEXT, sources=deepcopy(SOURCES), endpoint=BASE_URL,
        framing_intervention='A: production no-system; B: same payload plus symmetric system vocabulary.',
        max_asr_calls=4, max_seconds_per_request=30, groq_calls=0, tts_calls=0,
        calls=[], status='dry')
    audio = {}
    for name, source in SOURCES.items():
        try:
            audio[name], seconds = frozen_audio(ROOT, source)
            report['sources'][name].update(available=True, seconds=seconds)
        except (ValueError, OSError, wave.Error) as exc:
            report['sources'][name].update(available=False, error_type=type(exc).__name__)
    if not execute:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    path = ROOT / 'bench/results' / ('moulsot_context_' + uuid4().hex + '.json')
    try:
        if len(audio) != 2:
            raise ValueError('Both frozen source files must pass offline validation before any HTTP call.')
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30, follow_redirects=False,
                                     trust_env=False) as client:
            await run_probe(client, audio, report)
    except Exception as exc:
        report.update(status='error', error_type=type(exc).__name__,
                      error=str(exc) if isinstance(exc, ValueError) else 'Bounded loopback diagnostic failed.')
    finally:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(dict(report=str(path), status=report['status'], asr_calls=len(report['calls']))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Run the frozen bounded loopback experiment.')
    asyncio.run(main(parser.parse_args().execute))
