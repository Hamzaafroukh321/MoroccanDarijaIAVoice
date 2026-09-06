"""MoulSot / Groq adapters, explicit confidence provenance, persistent quotas."""

import argparse
import asyncio
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import io
import ipaddress
import json
import math
import os
import re
from pathlib import Path
import time
import wave

import httpx

from engine.moulsot_context import configured_context, context_sha256, validate_context_target
from engine.asr_provenance import (ENV_VAR as RUNTIME_SNAPSHOT_ENV, call_provenance,
    decode_snapshot, exact_local_endpoint, record_response_provenance)


class STTError(RuntimeError):
    pass


class RateLimitError(STTError):
    pass


class BudgetExhaustedError(RateLimitError):
    """A verified request or audio total exceeded a configured budget."""


class MoulSotQuotaError(STTError):
    """A recognized provider limit with an application-owned safe message."""


def recognition_failure(exc, provider, *, local_json=False):
    """Classify failures without reflecting response bodies, URLs or headers."""
    label = 'Local MoulSot' if local_json else 'MoulSot' if provider == 'moulsot' else 'Groq'
    status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
    if isinstance(exc, MoulSotQuotaError):
        return 'quota', str(exc), status
    if status == 429:
        return 'busy_or_rate_limited', f'{label} recognition is busy or rate-limited. Wait briefly, then start a new session.', status
    if status == 504 or isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError)):
        return 'timeout', f'{label} recognition timed out. Start a new session when the service is available; check the ASR service logs if this repeats.', status
    if status in {401, 403}:
        return 'authentication', f'{label} recognition access was denied. Check the provider credentials and permissions.', status
    if status is not None and status >= 500:
        return 'service_failure', f'{label} recognition service failed. Check the ASR service logs before starting a new session.', status
    if status is not None:
        return 'request_rejected', f'{label} recognition request was rejected. Check the ASR service configuration and input format.', status
    if isinstance(exc, BudgetExhaustedError):
        return 'local_budget', f'{label} recognition request budget is exhausted. Wait for the configured budget window to reset.', status
    if isinstance(exc, RateLimitError):
        return 'limiter_unavailable', f'{label} recognition budget could not be checked. Inspect the local quota ledger and service logs before retrying.', status
    if isinstance(exc, httpx.HTTPError):
        return 'connection', f'{label} recognition could not reach the service. Check that the ASR service is running and reachable.', status
    return 'invalid_response_or_configuration', f'{label} recognition failed. Check the ASR service configuration and logs.', status


def is_loopback_json(endpoint, protocol):
    if protocol != 'json':
        return False
    try:
        url = httpx.URL(endpoint)
        if url.scheme not in {'http', 'https'}:
            return False
        return url.host == 'localhost' or ipaddress.ip_address(url.host).is_loopback
    except (ValueError, httpx.InvalidURL):
        return False


@contextmanager
def timed_phase(record, name):
    """Measure a request-local phase even when its await fails or is cancelled."""
    started = time.perf_counter()
    try:
        yield
    finally:
        if record is not None:
            record['phases_ms'][name] = (time.perf_counter() - started) * 1000


def moulsot_inference_error(payload, authenticated):
    detail=payload.get('error','') if isinstance(payload,dict) else payload
    if not isinstance(detail,str): detail=''
    if 'zerogpu' in detail.casefold() and any(word in detail.casefold() for word in ('quota','runs limit')):
        reason='runs limit' if 'runs limit' in detail.casefold() else 'GPU time quota'
        message=f'MoulSot: Hugging Face ZeroGPU {reason} exceeded. '
        message+=('Your authenticated account has reached its limit. Wait for quota to reset.' if authenticated else
            'These server requests are anonymous. Set HF_TOKEN in the local .env file and restart the server to use your Hugging Face account quota, or wait for the limit to reset. Browser sign-in does not authenticate the server.')
        retry=re.search(r'Try again in (\d{1,3}:\d{2}:\d{2})',detail)
        if retry: message+=' Provider retry time: '+retry[1]+'.'
        return MoulSotQuotaError(message)
    return STTError('MoulSot inference failed. Check the Space container logs.')


def moulsot_error_envelope(data):
    """Recognize the owned Space's legacy errors stringified by one Textbox.

    Only exact tuple/status framing is recognized, never arbitrary HTML or an
    occurrence of "Error" in speech. Error details are neither parsed nor
    reflected. Other Gradio layouts and genuine empty transcripts keep their
    existing behavior.
    """
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], str):
        return False
    value = data[0]
    for status in ("<div style='color:red'>Unexpected result format.</div>",
                   "<div style='color:#666'>Please upload an audio file first.</div>"):
        if value == str(('', '', None, status)):
            return True
    # Python repr uses double quotes for the usual status, or single quotes
    # with escaped style quotes when the exception contains double quotes.
    return ((value.startswith("('', '', None, \"<div style='color:red'>Error: ") and
             value.endswith('</div>")')) or
            (value.startswith("('', '', None, '<div style=\\'color:red\\'>Error: ") and
             value.endswith("</div>')")))


@dataclass(frozen=True)
class Transcript:
    text: str
    confidence: float | None
    provider: str
    confidence_kind: str = 'unavailable'


def wav_bytes(pcm, runtime):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as writer:
        writer.setnchannels(runtime['channels'])
        writer.setsampwidth(runtime['sample_width_bytes'])
        writer.setframerate(runtime['sample_rate_hz'])
        writer.writeframes(pcm)
    return buffer.getvalue()


@contextmanager
def locked_file(path):
    """Fail closed if another process is reserving quota; never race counters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a+b') as lock:
        lock.seek(0, 2)
        if not lock.tell(): lock.write(b'\0'); lock.flush()
        lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RateLimitError('Quota ledger is busy; retry the request.') from exc
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == 'nt': msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(lock, fcntl.LOCK_UN)


class RateLimiter:
    def __init__(self, config, root, clock=time.time):
        self.settings = config['stt']
        self.path = Path(root) / self.settings['rate_limit_file']
        self.clock = clock

    def reserve(self, audio_seconds=0):
        now = self.clock()
        if not math.isfinite(audio_seconds) or audio_seconds < 0:
            raise RateLimitError('Invalid audio duration.')
        with locked_file(self.path):
            try:
                entries = json.loads(self.path.read_text()) if self.path.exists() else []
                if not isinstance(entries, list): raise ValueError('not a list')
                for item in entries:
                    if not all(isinstance(item[key], (int,float)) and math.isfinite(item[key]) for key in ('time','seconds')):
                        raise ValueError('invalid quota entry')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise RateLimitError('Quota ledger is unreadable; inspect it before continuing.') from exc
            longest = max(limit['window_seconds'] for limit in self.settings['limits'])
            entries = [item for item in entries if now - item['time'] < longest]
            for limit in self.settings['limits']:
                window = [item for item in entries if now - item['time'] < limit['window_seconds']]
                if 'requests' in limit and len(window) + 1 > limit['requests']:
                    raise BudgetExhaustedError(f"Groq request budget exhausted for {limit['window_seconds']} seconds.")
                if 'audio_seconds' in limit and sum(item['seconds'] for item in window) + audio_seconds > limit['audio_seconds']:
                    raise BudgetExhaustedError(f"Groq audio budget exhausted for {limit['window_seconds']} seconds.")
            entries.append({'time':now,'seconds':audio_seconds})
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps(entries), encoding='utf-8')
            temporary.replace(self.path)


class SpeechToText:
    def __init__(self, config, root, *, primary=None, endpoint=None, api_key=None, client=None, fallback=True, hf_token=None):
        self.config = config
        self.settings = deepcopy(config['stt'])
        protocol = os.getenv('MOULSOT_PROTOCOL', '').strip() or self.settings['moulsot_protocol']
        if protocol not in {'json', 'gradio'}:
            raise STTError('MOULSOT_PROTOCOL must be json or gradio; leave it blank to use the domain configuration.')
        self.settings['moulsot_protocol'] = protocol
        self.runtime = config['runtime']
        self.primary = primary or os.getenv('STT_PRIMARY', 'moulsot')
        if self.primary not in {'moulsot','groq'}: raise STTError('STT_PRIMARY must be moulsot or groq.')
        self.endpoint = endpoint if endpoint is not None else os.getenv('MOULSOT_ENDPOINT', '')
        self._runtime_snapshot, self._runtime_snapshot_status = decode_snapshot(os.getenv(RUNTIME_SNAPSHOT_ENV))
        try:
            self.moulsot_context = configured_context(self.settings)
        except ValueError:
            raise STTError('Invalid experimental MoulSot vocabulary configuration.') from None
        try:
            validate_context_target(self.moulsot_context, demo=config.get('demo'),
                primary=self.primary, fallback=fallback, protocol=protocol, endpoint=self.endpoint)
        except ValueError as exc:
            raise STTError(str(exc)) from None
        self.moulsot_context_hash = context_sha256(self.moulsot_context) if self.moulsot_context else None
        self.api_key = api_key if api_key is not None else os.getenv('GROQ_API_KEY','')
        self.hf_token = (hf_token if hf_token is not None else os.getenv('HF_TOKEN','')).strip()
        self.limiter = RateLimiter(config, root)
        client_options = {'timeout': self.settings['timeout_ms']/1000}
        if self.moulsot_context:
            client_options['trust_env'] = False
        self.client = client or httpx.AsyncClient(**client_options)
        self.owns_client = client is None
        self.fallback = fallback
        self.calls = []

    async def close(self):
        if self.owns_client: await self.client.aclose()

    async def transcribe(self, pcm):
        if not pcm: return Transcript('', None, self.primary)
        if len(pcm) % (self.runtime['channels'] * self.runtime['sample_width_bytes']):
            raise STTError('PCM contains an incomplete sample.')
        audio = wav_bytes(pcm, self.runtime)
        if len(audio) > self.settings['max_upload_bytes']: raise STTError('Audio exceeds upload limit.')
        providers = [self.primary]
        if self.primary == 'moulsot' and self.fallback: providers.append('groq')
        last_failure = None
        quota_error = None
        for provider in providers:
            started = time.perf_counter()
            record = {'provider': provider, 'ok': False, 'elapsed_clock': 'perf_counter',
                      'pcm_bytes': len(pcm), 'audio_duration_ms': len(pcm) * 1000 /
                      (self.runtime['sample_rate_hz'] * self.runtime['sample_width_bytes'] * self.runtime['channels']),
                      'phases_ms': {}}
            record['runtime_provenance'] = call_provenance(self._runtime_snapshot, self._runtime_snapshot_status,
                provider=provider, protocol=self.settings['moulsot_protocol'], endpoint=self.endpoint)
            if self.moulsot_context:
                record['moulsot_context'] = {'experimental': True,
                    'terms': deepcopy(self.settings['moulsot_context']['terms']),
                    'sha256': self.moulsot_context_hash, 'applied': False}
            try:
                async with asyncio.timeout(self.settings['timeout_ms']/1000):
                    result = await (self._moulsot(audio, timing=record) if provider == 'moulsot' else self._groq(audio, len(pcm)))
                record.update(ok=True, confidence_kind=result.confidence_kind)
                return result
            except asyncio.CancelledError:
                record.update(error='CancelledError', cancelled=True)
                raise
            except (httpx.HTTPError, STTError, ValueError, KeyError, TypeError, asyncio.TimeoutError) as exc:
                # Do not log provider bodies/headers, which can contain user text or secrets.
                record['error'] = type(exc).__name__
                kind, last_failure, status = recognition_failure(exc, provider,
                    local_json=provider == 'moulsot' and is_loopback_json(self.endpoint, self.settings['moulsot_protocol']))
                record['failure_kind'] = kind
                if status is not None:
                    record['http_status'] = status
                if isinstance(exc,MoulSotQuotaError):
                    quota_error=exc
                    record.update(message=str(exc),authenticated=bool(self.hf_token))
            finally:
                record['elapsed_ms'] = (time.perf_counter() - started) * 1000
                self.calls.append(record)
        if quota_error: raise quota_error
        raise STTError(last_failure) from None

    async def _moulsot(self, audio, *, timing=None):
        if not self.endpoint:
            raise STTError('Set MOULSOT_ENDPOINT to a hosted MoulSot Space, or use an isolated local JSON service with MOULSOT_PROTOCOL=json. Direct model loading is incompatible with the pinned runtime.')
        if self.settings['moulsot_protocol'] == 'json':
            request_options = {}
            if self.moulsot_context:
                request_options = {'data': {'context': self.moulsot_context}, 'follow_redirects': False}
            with timed_phase(timing, 'json_request'):
                response = await self.client.post(self.endpoint, files={'file':('segment.wav',audio,'audio/wav')},
                                                  **request_options)
                response.raise_for_status()
                payload = response.json()
            if timing is not None and 'runtime_provenance' in timing:
                eligible = exact_local_endpoint(self.endpoint, self.settings['moulsot_protocol'])
                try:
                    eligible = eligible and not response.history and str(response.url) == self.endpoint
                except RuntimeError:
                    eligible = False
                record_response_provenance(timing['runtime_provenance'], self._runtime_snapshot, payload,
                    eligible=eligible)
            if self.moulsot_context:
                if not isinstance(payload, dict) or payload.get('context_sha256') != self.moulsot_context_hash:
                    raise STTError('Local MoulSot did not acknowledge the configured vocabulary.')
                if timing is not None and 'moulsot_context' in timing:
                    timing['moulsot_context']['applied'] = True
            text = payload['text']
            confidence = payload.get('confidence')
            if not isinstance(text,str): raise STTError('MoulSot text must be a string.')
            if confidence is not None and (type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<=confidence<=1):
                raise STTError('Invalid MoulSot confidence.')
            return Transcript(text.strip(),confidence,'moulsot','provider_reported' if confidence is not None else 'unavailable')
        base = self.endpoint.rstrip('/') + '/gradio_api'
        # Never send the Hugging Face credential to custom JSON services or Groq.
        endpoint_url=httpx.URL(base)
        hf_host=endpoint_url.host.endswith('.hf.space')
        headers={'Authorization':'Bearer '+self.hf_token} if self.hf_token and hf_host and endpoint_url.scheme=='https' else {}
        with timed_phase(timing, 'upload'):
            upload = await self.client.post(base+'/upload',headers=headers,files={'files':('segment.wav',audio,'audio/wav')})
            upload.raise_for_status()
            paths = upload.json()
            if not isinstance(paths,list) or not paths or not isinstance(paths[0],str): raise STTError('Invalid MoulSot upload response.')
        call = base + '/call/' + self.settings['moulsot_api_name']
        with timed_phase(timing, 'submission'):
            response = await self.client.post(call,headers=headers,json={'data':[{'path':paths[0],'orig_name':'segment.wav','meta':{'_type':'gradio.FileData'}}, self.settings['language']]})
            response.raise_for_status()
            event_id = response.json()['event_id']
            if not isinstance(event_id,str) or not event_id.isalnum(): raise STTError('Invalid inference event id.')
        event = ''
        if timing is not None:
            timing['result_wait_basis'] = 'Combined stream wait; queue, network and inference are not separated.'
        with timed_phase(timing, 'result_wait'):
            async with self.client.stream('GET',call+'/'+event_id,headers=headers) as stream:
                stream.raise_for_status()
                async for line in stream.aiter_lines():
                    if line.startswith('event:'): event=line.partition(':')[2].strip()
                    elif line.startswith('data:') and event == 'error':
                        try: payload=json.loads(line.partition(':')[2])
                        except ValueError: payload=None
                        raise moulsot_inference_error(payload,bool(headers))
                    elif line.startswith('data:') and event == 'complete':
                        data=json.loads(line.partition(':')[2])
                        if not isinstance(data,list) or not data or not isinstance(data[0],str): raise STTError('Invalid MoulSot transcription.')
                        if moulsot_error_envelope(data):
                            raise STTError('MoulSot inference failed. Check the Space container logs.')
                        return Transcript(data[0].strip(),None,'moulsot')
        raise STTError('MoulSot stream ended without a result.')

    async def _groq(self, audio, pcm_size):
        if not self.api_key: raise STTError('GROQ_API_KEY is missing.')
        duration = pcm_size / (self.runtime['sample_rate_hz'] * self.runtime['sample_width_bytes'] * self.runtime['channels'])
        self.limiter.reserve(duration)
        response=await self.client.post(self.settings['groq_url'],headers={'Authorization':'Bearer '+self.api_key},files={'file':('segment.wav',audio,'audio/wav')},data={'model':self.settings['groq_model'],'response_format':'verbose_json','temperature':str(self.settings['temperature'])})
        response.raise_for_status()
        data=response.json()
        if not isinstance(data.get('text'),str): raise STTError('Invalid Groq transcription.')
        weighted=[]
        for segment in data.get('segments',[]):
            logprob=segment.get('avg_logprob')
            count=len(segment.get('tokens',[]))
            if type(logprob) in (int,float) and math.isfinite(logprob) and logprob<=0 and count:
                weighted.append((logprob,count))
        confidence=math.exp(sum(value*count for value,count in weighted)/sum(count for _,count in weighted)) if weighted else None
        return Transcript(data['text'].strip(),confidence,'groq','geometric_mean_token_probability' if weighted else 'unavailable')


async def _main():
    from dotenv import load_dotenv
    from engine.config import ROOT, load_config
    load_dotenv(ROOT/'.env')
    parser=argparse.ArgumentParser(description='Transcribe a local PCM16 mono WAV with the configured provider.')
    parser.add_argument('audio',type=Path)
    parser.add_argument('--domain',default='pizza')
    args=parser.parse_args()
    config=load_config(ROOT/'configs'/f'{args.domain}.json')
    with wave.open(str(args.audio),'rb') as audio:
        if (audio.getframerate(),audio.getnchannels(),audio.getsampwidth()) != (16000,1,2): raise STTError('Expected 16 kHz mono PCM16 WAV.')
        pcm=audio.readframes(audio.getnframes())
    stt=SpeechToText(config,ROOT)
    try: print((await stt.transcribe(pcm)).text)
    finally: await stt.close()


if __name__ == '__main__':
    asyncio.run(_main())
