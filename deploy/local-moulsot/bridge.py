"""Optional loopback bridge for an already running, verified MoulSot GGUF server.

Run explicitly with uvicorn bridge:app --host 127.0.0.1 --port 8012 from this
directory. This module neither starts llama.cpp nor changes the engine provider.
"""

import asyncio
import base64
from contextlib import asynccontextmanager
from email import policy
from email.parser import BytesParser
import io
import json
import os
from pathlib import Path
import re
import sys
import wave

from fastapi import FastAPI, HTTPException, Request
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.moulsot_context import context_sha256, validate_context_text
from engine.asr_provenance import ENV_VAR as RUNTIME_SNAPSHOT_ENV, decode_snapshot, runtime_ack


UPSTREAM_URL = 'http://127.0.0.1:8011/v1/chat/completions'
EXPECTED_MODEL = 'moulsot.v0.3.Q4_K_M.gguf'
MODEL_VARIANT = 'moulsot.v0.3-Q4_K_M-mmproj-Q8_0'
MAX_AUDIO_SECONDS = 20
MAX_WAV_BYTES = MAX_AUDIO_SECONDS * 32000 + 65536
MAX_BODY_BYTES = MAX_WAV_BYTES + 65536
MAX_RESPONSE_BYTES = 65536
MAX_OUTPUT_TOKENS = 512  # Up to 20 s of Arabic can exceed a short 128-token probe.
REQUEST_TIMEOUT_SECONDS = 30
PREFIX = 'language Arabic<asr_text>'


def validate_wav(raw):
    if len(raw) > MAX_WAV_BYTES:
        raise HTTPException(413, 'WAV exceeds the byte limit.')
    if (len(raw) < 44 or raw[:4] != b'RIFF' or raw[8:12] != b'WAVE' or
            int.from_bytes(raw[4:8], 'little') + 8 != len(raw)):
        raise HTTPException(400, 'Expected a complete RIFF WAV file.')
    offset, data_chunks = 12, 0
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise HTTPException(400, 'Incomplete WAV chunk header.')
        size = int.from_bytes(raw[offset + 4:offset + 8], 'little')
        if raw[offset:offset + 4] == b'data':
            data_chunks += 1
            if size % 2:
                raise HTTPException(400, 'WAV contains a partial PCM16 sample.')
        offset += 8 + size + (size % 2)
        if offset > len(raw):
            raise HTTPException(400, 'Incomplete WAV chunk data.')
    if data_chunks != 1:
        raise HTTPException(400, 'Expected exactly one WAV audio data chunk.')
    try:
        with wave.open(io.BytesIO(raw), 'rb') as audio:
            if (audio.getnchannels(), audio.getframerate(), audio.getsampwidth(), audio.getcomptype()) != (1, 16000, 2, 'NONE'):
                raise HTTPException(400, 'WAV must be mono 16000 Hz uncompressed PCM16.')
            samples = audio.getnframes()
            if samples > MAX_AUDIO_SECONDS * 16000:
                raise HTTPException(413, 'WAV exceeds the 20-second duration limit.')
            if samples == 0 or len(audio.readframes(samples)) != samples * 2:
                raise HTTPException(400, 'WAV must contain complete, nonempty PCM16 samples.')
    except (wave.Error, EOFError, ValueError) as exc:
        raise HTTPException(400, 'Malformed WAV file.') from exc
    return raw


async def read_upload(request):
    content_type = request.headers.get('content-type', '')
    if not content_type.lower().startswith('multipart/form-data;') or any(c in content_type for c in '\r\n'):
        raise HTTPException(415, 'Send a multipart/form-data WAV field named file.')
    length = request.headers.get('content-length')
    if length is not None:
        try:
            length = int(length)
        except ValueError as exc:
            raise HTTPException(400, 'Invalid Content-Length.') from exc
        if length < 0:
            raise HTTPException(400, 'Invalid Content-Length.')
        if length > MAX_BODY_BYTES:
            raise HTTPException(413, 'Upload exceeds the byte limit.')
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_BODY_BYTES:
            raise HTTPException(413, 'Upload exceeds the byte limit.')
        body.extend(chunk)
    try:
        header = ('Content-Type: ' + content_type + '\r\nMIME-Version: 1.0\r\n\r\n').encode('ascii')
        message = BytesParser(policy=policy.default).parsebytes(header + body)
        if message.defects or not message.is_multipart():
            raise ValueError('Invalid multipart envelope')
        parts = list(message.iter_parts())
        if not 1 <= len(parts) <= 2:
            raise ValueError('Expected one file and at most one context part')
        fields = {}
        for part in parts:
            name = part.get_param('name', header='content-disposition')
            if (part.defects or part.is_multipart() or part.get_content_disposition() != 'form-data' or
                    name not in {'file', 'context'} or name in fields or
                    part.get('content-transfer-encoding', '').lower() not in {'', 'binary', '8bit'}):
                raise ValueError('Invalid or repeated multipart part')
            if ((name == 'file' and not part.get_filename()) or
                    (name == 'context' and part.get_filename() is not None)):
                raise ValueError('Invalid multipart filename')
            fields[name] = part.get_payload(decode=True)
            if not isinstance(fields[name], bytes):
                raise ValueError('Invalid multipart body')
        if 'file' not in fields:
            raise ValueError('Missing audio file')
        raw = fields['file']
        context = validate_context_text(fields.get('context', b'').decode('utf-8'))
    except (ValueError, UnicodeError, TypeError) as exc:
        raise HTTPException(400, 'Expected one complete WAV file and at most one valid bounded context field.') from exc
    return validate_wav(raw), context


def parse_transcription(payload):
    if not isinstance(payload, dict):
        raise ValueError('Expected an upstream response object.')
    model = payload.get('model')
    if not isinstance(model, str) or model.replace('\\', '/').rsplit('/', 1)[-1] != EXPECTED_MODEL:
        raise ValueError('Upstream did not identify the expected MoulSot model.')
    choices = payload.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError('Expected exactly one transcription choice.')
    choice = choices[0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Transcription was truncated or did not finish normally.')
    message = choice.get('message')
    if (not isinstance(message, dict) or message.get('role') != 'assistant' or
            message.get('tool_calls') or message.get('function_call')):
        raise ValueError('Expected an assistant transcription message.')
    content = message.get('content')
    if not isinstance(content, str) or not content.startswith(PREFIX):
        raise ValueError('Expected the exact Arabic ASR prefix.')
    text = content[len(PREFIX):]
    if re.search(r'[<>\x00-\x08\x0b\x0c\x0e-\x1f]', text):
        raise ValueError('Transcription contains unexpected model markers or control characters.')
    return {'text': text if text.strip() else '', 'confidence': None,
            'model_variant': MODEL_VARIANT,
            'provenance': {'source_model': 'atlasia/moulsot.v0.3',
                           'upstream_model': model, 'quantized': True}}


def create_app(client=None):
    # Capture optional launch metadata once. Standalone or malformed metadata
    # changes no inference behavior and creates no verified provenance claim.
    snapshot, _ = decode_snapshot(os.getenv(RUNTIME_SNAPSHOT_ENV))
    acknowledgment = runtime_ack(snapshot) if snapshot is not None else None
    @asynccontextmanager
    async def lifespan(application):
        if client is None:
            application.state.client = httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False, follow_redirects=False)
        try:
            yield
        finally:
            if client is None:
                await application.state.client.aclose()
                application.state.client = None

    application = FastAPI(title='Optional local MoulSot bridge', lifespan=lifespan)
    application.state.client = client
    application.state.busy = False

    @application.get('/health')
    async def health():
        return {'status': 'adapter_ready', 'upstream_checked': False,
                'model_variant': MODEL_VARIANT, 'busy': application.state.busy}

    @application.post('/transcribe')
    async def transcribe(request: Request):
        if application.state.busy:
            raise HTTPException(429, 'Local MoulSot is busy; no inference was queued.')
        # There is no await between inspection and reservation on this event loop.
        application.state.busy = True
        try:
            async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
                audio, context = await read_upload(request)
                upstream = application.state.client
                if upstream is None:
                    raise HTTPException(503, 'Bridge lifespan has not started.')
                payload = {'messages': [{'role': 'user', 'content': [
                    {'type': 'text', 'text': 'a'},
                    {'type': 'input_audio', 'input_audio': {
                        'data': base64.b64encode(audio).decode('ascii'), 'format': 'wav'}}]}],
                    'max_tokens': MAX_OUTPUT_TOKENS, 'temperature': 0, 'cache_prompt': False, 'stream': False}
                if context:
                    payload['messages'].insert(0, {'role': 'system', 'content': context})
                async with upstream.stream('POST', UPSTREAM_URL, json=payload) as response:
                    if response.status_code != 200:
                        raise HTTPException(502, 'Local MoulSot upstream returned an error.')
                    raw = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(raw) + len(chunk) > MAX_RESPONSE_BYTES:
                            raise HTTPException(502, 'Local MoulSot response exceeds the byte limit.')
                        raw.extend(chunk)
                try:
                    result = parse_transcription(json.loads(raw))
                    if acknowledgment is not None:
                        result['runtime_snapshot'] = dict(acknowledgment)
                    if context:
                        # Acknowledge the exact forwarded context, not a claim
                        # that the model obeyed it or improved recognition.
                        result['context_sha256'] = context_sha256(context)
                    return result
                except (ValueError, UnicodeError, TypeError) as exc:
                    raise HTTPException(502, 'Invalid local MoulSot transcription: ' + str(exc)) from exc
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise HTTPException(504, 'Local MoulSot request timed out.') from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, 'Local MoulSot upstream is unavailable.') from exc
        finally:
            application.state.busy = False

    return application


app = create_app()
