"""Opt-in synthetic domain previews using the shared task/voice engine.

Draft language and synthetic output are kept separate from the reviewed research bank.
"""

import audioop
import asyncio
from copy import deepcopy
from datetime import date
import hashlib
import io
import json
import os
import re
import struct
import time
import tempfile
from pathlib import Path
import wave
from urllib.parse import quote
from xml.sax.saxutils import escape, quoteattr

import httpx

from engine.config import ROOT
from engine.demo_validation import DemoConfigError, validate_demo_config
from engine.responder import RenderedAudio, AudioBankError
from engine.stt import RateLimiter, wav_bytes
from engine.state import Action, TaskState


DEMO_PROFILES = {'pizza': 'voice.json', 'clinic': 'clinic.json'}
DEMO_TTS_PROVIDERS = frozenset({'groq','elevenlabs','azure','darija_xtts'})


def demo_supported(config):
    return demo_profile_path(config) is not None


def demo_profile_path(config):
    domain = config.get('domain_id')
    if not isinstance(domain, str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', domain):
        return None
    path = ROOT / 'configs/demo' / DEMO_PROFILES.get(domain, domain + '.json')
    return path if path.is_file() else None


def demo_config(config):
    profile_path = demo_profile_path(config)
    if profile_path is None:
        raise ValueError('No synthetic preview is registered for this domain.')
    result = deepcopy(config)
    shared = json.loads((ROOT/'configs/demo/voice.json').read_text(encoding='utf-8'))
    if config['domain_id'] == 'pizza':
        settings = shared
        settings.setdefault('state_kind', 'collection_scoped')
        settings.setdefault('task_scope', 'pizza pickup; no address or phone')
        settings.setdefault('ui', {
            'title': 'Let’s order a pizza.',
            'note': 'Pizza pickup demo · Draft Darija. Ask for a pizza, change a detail, then confirm the summary. Pause briefly after each turn. This demo does not place an order.',
            'state_title': 'Your order',
            'state_empty': 'Each pizza keeps its quantity, size and toppings. Drinks belong to the order.',
            'completion_text': 'You confirmed the demo order. No real order was placed.',
        })
    else:
        # Share transport/voice settings, not pizza fields, language or policies.
        common = {'elevenlabs','azure','darija_xtts','cache_dir','base_silence_ms',
                  'max_segment_ms','max_session_ms','max_repairs','router_max_tokens','number_words'}
        settings = {key:deepcopy(value) for key,value in shared.items()
                    if key.startswith('tts_') or key in common}
        settings.update(json.loads(profile_path.read_text(encoding='utf-8')))
        if settings.get('state_kind') not in {'flat_scoped', 'configured_collection_scoped'}:
            raise DemoConfigError('Additional task previews require flat_scoped or configured_collection_scoped.')
        if settings.get('order_schema_version') is not None:
            raise DemoConfigError('A flat task preview cannot select the pizza order schema adapter.')
    provider=os.getenv('DEMO_TTS_PROVIDER','groq').strip().lower()
    if provider not in DEMO_TTS_PROVIDERS: raise ValueError('DEMO_TTS_PROVIDER must be groq, elevenlabs, azure or darija_xtts.')
    settings['tts_provider']=provider
    if provider in {'elevenlabs','azure','darija_xtts'}: settings.update(settings[provider])
    result['demo'] = settings
    # Multi-item operations need more output room than the original flat router.
    result['router']['max_tokens'] = settings['router_max_tokens']
    if result['router']['model'] in {'openai/gpt-oss-120b','openai/gpt-oss-20b'}:
        result['router']['reasoning_effort'] = 'low'
    result['display_name'] = settings['label']
    if not isinstance(settings.get('questions'), dict):
        raise DemoConfigError('demo.questions must be an object.')
    settings.setdefault('required_slots', list(settings['questions']))
    fields = settings.get('fields', list(settings['questions']))
    if (not isinstance(fields, list) or not fields or
            any(not isinstance(field, str) or not field for field in fields) or
            len(fields) != len(set(fields))):
        raise ValueError('Preview fields must be a nonempty list of unique slot IDs.')
    result['slots'] = [s for s in result['slots'] if s['id'] in fields]
    if (len(result['slots']) != len(fields) or
            {slot['id'] for slot in result['slots']} != set(fields)):
        raise ValueError('Preview fields must exist exactly once in the selected domain.')
    overrides = settings.get('slot_overrides', {})
    if not isinstance(overrides, dict) or not set(overrides).issubset(fields):
        raise ValueError('Preview slot_overrides must map selected slot IDs to overrides.')
    for slot_id, override in overrides.items():
        if not isinstance(override, dict):
            raise ValueError(f'Preview slot override for {slot_id} must be an object.')
        if 'id' in override and override['id'] != slot_id:
            raise ValueError(f'Preview slot override cannot change the identity of {slot_id}.')
    for slot in result['slots']:
        slot.update(deepcopy(overrides.get(slot['id'],{})))
    validate_demo_config(result)
    for slot in result['slots']:
        if slot['id'] in settings.get('required_slots',[]):
            slot['required'] = True
    settings['menu_options']={slot['id']:[settings['values'].get(value,value) for value in slot.get('values',[])] for slot in result['slots']}
    # One MoulSot call per finished turn. No speculative ASR calls against ZeroGPU.
    result['endpointing'].update(base_silence_ms=settings['base_silence_ms'],
        hesitation_hold_ms=0, continuation_hold_ms=0, digit_hold_ms=0,
        max_segment_ms=settings['max_segment_ms'])
    result['engine']['max_session_ms'] = settings['max_session_ms']
    result['stt']['timeout_ms'] = 120000
    result['normalization']['darija_numbers'] = settings['number_words']
    return result


def _readback_value(key, value, settings):
    kind = settings.get('readback_formats',{}).get(key)
    if kind == 'day_month_year':
        parsed = date.fromisoformat(value)
        return f'{parsed.day:02d} / {parsed.month:02d} / {parsed.year:04d}'
    if kind == '24_hour':
        return str(value).replace(':', ' : ')
    entries = value if isinstance(value,list) else [value]
    return '، '.join(settings['values'].get(str(entry),str(entry)) for entry in entries)


def readback_parts(values, settings):
    """Complete semantic phrases, shared by text and optional cached grouping."""
    pieces = [settings['readback_prefix']]
    collection = settings.get('state_kind') == 'collection_scoped' or (
        'state_kind' not in settings and settings.get('order_schema_version') == 2)
    if settings.get('state_kind') == 'configured_collection_scoped':
        schema = settings['transaction_schema']
        order = settings['readback_order']
        for index, item in enumerate(values.get(schema['collection'], []), 1):
            pieces.append(settings['collection_label'] + ' ' + str(index) + '.')
            for key in order:
                if key in schema['item_slots'] and key in item:
                    pieces.append(settings['labels'][key] + ': ' + _readback_value(key, item[key], settings) + '.')
        for key in order:
            if key in schema['root_slots'] and key in values:
                pieces.append(settings['labels'][key] + ': ' + _readback_value(key, values[key], settings) + '.')
    elif collection and 'items' in values:
        for index, item in enumerate(values['items']):
            pieces.append(settings['item_label'] + ' ' + str(index + 1) + '.')
            for key in ('quantity', 'size', 'toppings'):
                value = item[key]
                entries = value if isinstance(value, list) else [value]
                rendered = '، '.join(settings['values'].get(str(entry), str(entry)) for entry in entries)
                if key == 'toppings' and not entries:
                    rendered = settings['plain_toppings']
                pieces.append(settings['labels'][key] + ': ' + rendered + '.')
        drinks = '، '.join(settings['values'][entry] for entry in values['drink'])
        pieces.append(settings['labels']['drink'] + ': ' + drinks + '.')
    else:
        for key in settings.get('readback_order', settings['labels']):
            if key in values:
                rendered = _readback_value(key, values[key], settings)
                pieces.append(settings['labels'][key] + ': ' + rendered + '.')
    pieces.append(settings['readback_question'])
    return pieces


def reply_text(action, values, settings):
    if settings.get('state_kind') == 'configured_collection_scoped':
        schema = settings['transaction_schema']
        rows = values.get(schema['collection'], [])
        prefix = ''
        if action.item_id is not None:
            index = next((i for i, row in enumerate(rows, 1) if row['id'] == action.item_id), len(rows) + 1)
            prefix = settings['collection_label'] + ' ' + str(index) + '. '
        if action.kind in {'item_reference', 'ambiguous'}:
            return settings['responses']['item_reference']
        if action.slot and action.kind in {'ask', 'repair', 'ambiguous_value', 'unintelligible'}:
            return prefix + settings['questions'][action.slot]
        if action.kind == 'unsupported_value' and action.slot:
            options = settings.get('menu_options', {}).get(action.slot, [])
            return prefix + settings['responses']['unsupported_option'] + ' ' + (
                '، '.join(options) + '.' if options else settings['questions'][action.slot])
    if action.kind == 'unsupported_value' and action.slot:
        options = settings.get('menu_options',{}).get(action.slot,[])
        return settings['responses']['unsupported_option']+' '+(
            '، '.join(options)+'.' if options else settings['questions'][action.slot])
    if action.kind == 'ambiguous_value' and action.slot:
        return settings['questions'][action.slot]
    if action.kind == 'item_reference':
        action=Action('ambiguous',action.slot,action.item_id)
    if action.kind == 'unsupported_menu' and action.slot:
        options=settings.get('menu_options',{}).get(action.slot,[])
        return settings['responses']['unsupported_option']+' '+('، '.join(options)+'.' if options else settings['questions'][action.slot])
    if action.kind == 'unintelligible' and action.slot:
        action=Action('repair',action.slot,action.item_id)
    collection = settings.get('state_kind') == 'collection_scoped' or (
        'state_kind' not in settings and settings.get('order_schema_version') == 2)
    if collection and 'items' in values:
        items = values['items']
        if action.kind == 'ambiguous' and not items:
            return settings['responses']['ambiguous_new_order']
        def item_name(index):
            return settings['item_label'] + ' ' + str(index + 1)
        if action.kind in {'ask','repair'} and action.slot:
            prefix = ''
            if action.item_id is not None:
                index = next((i for i,item in enumerate(items) if item['id']==action.item_id),len(items))
                prefix = item_name(index) + '. '
            return prefix + settings['questions'][action.slot]
    if action.kind == 'repair':
        return settings['questions'][action.slot] if action.slot else settings['responses']['listen']
    if action.kind == 'ask':
        return settings['questions'][action.slot]
    if action.kind == 'readback':
        return ' '.join(readback_parts(values, settings))
    return settings['responses'][action.kind]


class DemoTaskState(TaskState):
    """Bound recovery and preserve the reviewed engine's state semantics."""
    def __init__(self, config):
        super().__init__(config)
        self.repair_count = 0
        self.unsupported_items = False

    def consume(self, response):
        intent = response.get('intent', 'task')
        if intent == 'multiple_items':
            self.unsupported_items = True
            self.invalidate_confirmation()
            self.repair_count += 1
            return Action('handoff' if self.repair_count > self.config['demo']['max_repairs'] else 'multiple_items')
        if self.unsupported_items:
            # Never accept an old flattened order after a rejected multi-item request.
            return Action('handoff')
        if response['unclear'] or intent != 'task':
            self.invalidate_confirmation()
            if intent != 'greeting': self.repair_count += 1
            if self.repair_count > self.config['demo']['max_repairs']:
                return Action('handoff')
            pending = self.next_action()
            return Action('repair', pending.slot if pending.kind=='ask' else None)
        before = (self.version,self.readback_version,self.awaiting_correction,self.confirmed_version)
        action = super().consume(response)
        after = (self.version,self.readback_version,self.awaiting_correction,self.confirmed_version)
        if after != before: self.repair_count = 0
        return action


def pcm16(data, runtime):
    try:
        # Groq returns a complete HTTP body with FFmpeg's unknown-length WAV
        # sentinels. Finalize only those lengths, preserving finite-size checks.
        if data[:4]==b'RIFF' and data[4:8]==b'\xff'*4 and data[8:12]==b'WAVE':
            fixed=bytearray(data)
            offset=12
            while offset+8<=len(fixed):
                size=struct.unpack_from('<I',fixed,offset+4)[0]
                if fixed[offset:offset+4]==b'data' and size==0xffffffff:
                    struct.pack_into('<I',fixed,offset+4,len(fixed)-offset-8)
                    struct.pack_into('<I',fixed,4,len(fixed)-8)
                    data=bytes(fixed)
                    break
                offset+=8+size+(size%2)
        with wave.open(io.BytesIO(data), 'rb') as audio:
            if audio.getcomptype() != 'NONE' or audio.getnchannels() not in (1, 2):
                raise AudioBankError('Unsupported demo speech WAV.')
            width = audio.getsampwidth()
            pcm = audio.readframes(audio.getnframes())
            if not pcm or len(pcm) != audio.getnframes()*width*audio.getnchannels():
                raise AudioBankError('Demo speech WAV is empty or truncated.')
            if width == 1:
                pcm = audioop.bias(pcm, 1, -128)
            if width != 2:
                pcm = audioop.lin2lin(pcm, width, 2)
            if audio.getnchannels() == 2:
                pcm = audioop.tomono(pcm, 2, .5, .5)
            if audio.getframerate() != runtime['sample_rate_hz']:
                pcm, _ = audioop.ratecv(pcm, 2, 1, audio.getframerate(), runtime['sample_rate_hz'], None)
            return pcm
    except (wave.Error, EOFError, audioop.error) as exc:
        raise AudioBankError('The demo speech provider returned invalid audio.') from exc


def chunks(text, maximum):
    words = text.split()
    current = ''
    for word in words:
        if len(word) > maximum:
            raise AudioBankError('A speech token exceeds the provider limit.')
        candidate = (current + ' ' + word).strip()
        if len(candidate) > maximum:
            yield current
            current = word
        else:
            current = candidate
    if current:
        yield current


def readback_groups(values, settings, maximum):
    """Pack complete fields in order; only oversized fields use word splitting."""
    groups = []
    current = ''
    for piece in readback_parts(values, settings):
        for part in chunks(piece, maximum):
            candidate = (current + ' ' + part).strip()
            if current and len(candidate) > maximum:
                groups.append(current)
                current = part
            else:
                current = candidate
    if current:
        groups.append(current)
    return groups


def azure_speech_url():
    region = os.getenv('AZURE_SPEECH_REGION', '').strip().lower()
    if not re.fullmatch(r'[a-z][a-z0-9]{1,39}', region):
        raise AudioBankError('Set AZURE_SPEECH_REGION to the region of your Speech resource, for example westeurope.')
    return f'https://{region}.tts.speech.microsoft.com/cognitiveservices/v1'


def speech_cache_path(settings, root, text):
    provider = settings.get('tts_provider', 'groq')
    identity = [provider, settings['tts_model'], settings['tts_voice'], text]
    if provider == 'darija_xtts':
        # Preserve existing default-speaker clips. Nondefault API/revision gets
        # a separate key; remote checkpoint freshness is not inferred from a URL.
        identity.extend([settings['tts_url'], settings['tts_temperature']])
        if settings.get('tts_api_name') != 'infer_EGTTS':
            identity.append({'api_name': settings.get('tts_api_name')})
    elif provider in {'groq', 'elevenlabs'}:
        identity.extend([settings['tts_url'], settings.get('tts_output_format')])
    if settings.get('tts_cache_revision'):
        identity.append({'cache_revision': settings['tts_cache_revision']})
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
    return Path(root)/settings['cache_dir']/(digest+'.wav')


def read_speech_cache(path, runtime):
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError:
        return None
    try:
        pcm16(raw, runtime)
    except AudioBankError:
        return None
    return raw


class DemoVoice:
    def __init__(self, config, root, *, client=None):
        self.config = config
        self.settings = config['demo']
        self.provider = self.settings.get('tts_provider','groq')
        self.root = Path(root)
        self.limiter = RateLimiter(config, root)
        self.client = client or httpx.AsyncClient(timeout=self.settings['tts_timeout_ms']/1000)
        self.owns_client = client is None
        self.xtts_reference = None
        self._xtts_references = {}
        self.calls = []
        self._inflight = {}
        self._closed = False
        self._close_task = None

    async def close(self):
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_resources())
            self._close_task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        await asyncio.shield(self._close_task)

    async def _close_resources(self):
        tasks = list(self._inflight.values())
        for task in tasks:
            task.cancel()
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            if self.owns_client:
                await self.client.aclose()

    async def _darija_xtts(self, text, settings=None):
        """Use the author's hosted checkpoint and default speaker through Gradio."""
        settings = settings or self.settings
        endpoint = settings['tts_url'].rstrip('/')
        url = httpx.URL(endpoint)
        if url.scheme != 'https' or not url.host.endswith('.hf.space'):
            raise AudioBankError('Darija XTTS requires an HTTPS Hugging Face Space endpoint.')
        token = os.getenv('HF_TOKEN', '')
        headers = {'Authorization': 'Bearer '+token} if token else {}
        try:
            async with asyncio.timeout(settings['tts_timeout_ms']/1000):
                reference = self._xtts_references.get(endpoint)
                if reference is None:
                    response = await self.client.get(endpoint+'/config', headers=headers)
                    response.raise_for_status()
                    components = response.json()['components']
                    reference = next(c['props']['value'] for c in components
                        if c['type']=='audio' and c['props'].get('label')=='Speaker reference')
                    if not isinstance(reference, dict) or not isinstance(reference.get('path'), str):
                        raise ValueError('Missing reference')
                    reference = {'path':reference['path'], 'meta':{'_type':'gradio.FileData'}}
                    self._xtts_references[endpoint] = reference
                    self.xtts_reference = reference
                call = endpoint+'/gradio_api/call/'+settings['tts_api_name']
                response = await self.client.post(call, headers=headers,
                    json={'data':[text, reference, settings['tts_temperature']]})
                response.raise_for_status()
                event_id = response.json()['event_id']
                if not isinstance(event_id, str) or not event_id.isalnum():
                    raise ValueError('Invalid event id')
                event = ''
                async with self.client.stream('GET', call+'/'+event_id, headers=headers) as stream:
                    stream.raise_for_status()
                    async for line in stream.aiter_lines():
                        if line.startswith('event:'): event=line.partition(':')[2].strip()
                        elif line.startswith('data:') and event=='error':
                            detail=line.partition(':')[2].lower()
                            if any(word in detail for word in ('quota','limit exceeded','gpu duration','too many requests')):
                                raise AudioBankError('Darija XTTS shared GPU quota is exhausted. Check HF_TOKEN or wait for the quota reset.')
                            raise AudioBankError('Darija XTTS generation failed. Its hosted Space may be busy or unavailable.')
                        elif line.startswith('data:') and event=='complete':
                            result=json.loads(line.partition(':')[2])
                            path=result[0]['path']
                            if not isinstance(path,str) or not path.startswith('/tmp/gradio/') or '..' in path.split('/'):
                                raise ValueError('Invalid audio path')
                            # The Space sometimes returns a malformed /gradi-prefixed URL.
                            # Use its same-origin canonical file route, never a returned host.
                            response=await self.client.get(endpoint+'/gradio_api/file='+quote(path,safe='/'),headers=headers)
                            response.raise_for_status()
                            return response
                raise AudioBankError('Darija XTTS stream ended without audio.')
        except TimeoutError:
            raise AudioBankError('Darija XTTS timed out waiting for its shared GPU. Try again later.') from None
        except (KeyError,IndexError,TypeError,ValueError,StopIteration):
            raise AudioBankError('Darija XTTS returned an unexpected response. Check the hosted Space configuration.') from None

    async def render(self, action, values):
        if self._closed:
            raise AudioBankError('The demo voice is closed.')
        started = time.perf_counter()
        record = {'provider': self.provider, 'ok': False, 'cache_hits': 0, 'cache_misses': 0,
                  'coalesced_waits': 0, 'elapsed_clock': 'perf_counter'}
        try:
            rendered = await self._render_audio(action, values, record)
            record['ok'] = True
            return rendered
        finally:
            record['elapsed_ms'] = (time.perf_counter()-started)*1000
            self.calls.append(record)

    async def _render_audio(self, action, values, record):
        settings = deepcopy(self.settings)
        text = reply_text(action, values, settings)
        record['characters'] = len(text)
        parts = list(chunks(text, settings['tts_max_chars']))
        record['render_mode'] = 'whole'
        if action.kind == 'readback' and settings.get('tts_readback_mode', 'whole') == 'grouped':
            # A complete existing render is cheaper and preserves its prosody.
            # Never store assembled groups under its full-text cache identity.
            legacy_ready = all(read_speech_cache(speech_cache_path(settings, self.root, part),
                self.config['runtime']) is not None for part in parts)
            if not legacy_ready:
                maximum = settings.get('tts_readback_group_chars', min(96, settings['tts_max_chars']))
                try:
                    grouped = readback_groups(values, settings, maximum)
                except AudioBankError:
                    # The provider-sized legacy plan was validated above. A
                    # grouping target must not reject an otherwise valid token.
                    grouped = parts
                if len(grouped) <= 16 and grouped != parts:
                    parts = grouped
                    record['render_mode'] = 'grouped'
        record['parts'] = len(parts)
        pieces = []
        for part in parts:
            if self._closed:
                raise AudioBankError('The demo voice is closed.')
            path = speech_cache_path(settings, self.root, part)
            raw = read_speech_cache(path, self.config['runtime'])
            if raw is not None:
                record['cache_hits'] += 1
            else:
                task = self._inflight.get(path)
                if task is None:
                    record['cache_misses'] += 1
                    task = asyncio.create_task(self._produce_part(part, path, settings))
                    self._inflight[path] = task
                    def retire(done, key=path):
                        if self._inflight.get(key) is done:
                            self._inflight.pop(key, None)
                        if not done.cancelled():
                            done.exception()  # Retrieve failures even if every waiter left.
                    task.add_done_callback(retire)
                else:
                    record['coalesced_waits'] += 1
                raw = await asyncio.shield(task)
            pieces.append(pcm16(raw, self.config['runtime']))
        pcm = b''.join(pieces)
        return RenderedAudio(wav_bytes(pcm, self.config['runtime']), len(pcm)//2,
            len(pcm)//2, ['synthetic_demo:'+settings['tts_model']], text=text)

    async def _produce_part(self, part, path, settings):
        key_name={'elevenlabs':'ELEVENLABS_API_KEY','azure':'AZURE_SPEECH_KEY','groq':'GROQ_API_KEY','darija_xtts':'HF_TOKEN'}[self.provider]
        key = os.getenv(key_name, '')
        if not key and self.provider!='darija_xtts':
            raise AudioBankError(f'Set {key_name} for the demo voice.')
        if self.provider=='groq': self.limiter.reserve()
        try:
            if self.provider=='darija_xtts':
                response = await self._darija_xtts(part, settings)
            elif self.provider=='azure':
                ssml = ('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="ar-MA">'
                    f'<voice name={quoteattr(settings["tts_voice"])}>{escape(part)}</voice></speak>')
                response=await self.client.post(azure_speech_url(),
                    headers={'Ocp-Apim-Subscription-Key':key,'Content-Type':'application/ssml+xml',
                        'X-Microsoft-OutputFormat':'riff-16khz-16bit-mono-pcm','User-Agent':'DarijaVoice'},
                    content=ssml.encode('utf-8'))
            elif self.provider=='elevenlabs':
                response=await self.client.post(settings['tts_url']+settings['tts_voice'],
                    params={'output_format':settings['tts_output_format']},
                    headers={'xi-api-key':key},json={'model_id':settings['tts_model'],'text':part})
            else:
                response = await self.client.post(settings['tts_url'],
                    headers={'Authorization':'Bearer '+key}, json={'model':settings['tts_model'],
                        'voice':settings['tts_voice'], 'input':part, 'response_format':'wav'})
            response.raise_for_status()
            raw = response.content
            if self.provider=='elevenlabs':
                if not raw or len(raw)%2: raise AudioBankError('ElevenLabs returned empty or incomplete PCM audio.')
                raw=wav_bytes(raw,self.config['runtime'])
            pcm16(raw, self.config['runtime'])
        except httpx.HTTPStatusError as exc:
            if self.provider=='darija_xtts':
                raise AudioBankError(f'Darija XTTS returned HTTP {exc.response.status_code}. Check the hosted Space availability and Hugging Face quota.') from None
            if self.provider=='azure':
                advice = ('Azure Speech is throttled or its quota is exhausted. Wait before trying again; check your F0 usage.'
                    if exc.response.status_code==429 else
                    f'Azure Speech returned HTTP {exc.response.status_code}. Check the Speech key, matching region and voice availability.')
                raise AudioBankError(advice) from None
            if self.provider=='elevenlabs' and exc.response.status_code==402:
                raise AudioBankError('ElevenLabs requires a paid plan to use this library voice through the API. The website preview can work while API access is blocked. Review your ElevenLabs subscription.') from None
            try: code=exc.response.json().get('error',{}).get('code')
            except (ValueError,AttributeError): code=None
            if code=='model_terms_required':
                raise AudioBankError('Accept the Arabic voice model terms in your Groq playground, then start again: https://console.groq.com/playground?model=canopylabs%2Forpheus-arabic-saudi') from None
            provider_name='ElevenLabs' if self.provider=='elevenlabs' else 'Groq'
            raise AudioBankError(f'{provider_name} speech returned HTTP {exc.response.status_code}. Check your API key, voice access, plan and credits.') from None
        except httpx.HTTPError:
            raise AudioBankError('The demo voice service could not be reached.') from None
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem+'.', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return raw
