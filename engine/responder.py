"""Reviewed human WAV bank, deterministic readback and restricted TTS fallback."""

from dataclasses import dataclass
import io
from pathlib import Path
import wave

import httpx

from engine.stt import wav_bytes


class AudioBankError(RuntimeError):
    pass


def review_issues(value, path='config'):
    issues=[]
    if isinstance(value,dict):
        for key,item in value.items():
            if '[[' in key: issues.append(path+'.<placeholder key>')
            if key.casefold()=='needs_human_review' and item is True: issues.append(path+'.'+key)
            issues.extend(review_issues(item,path+'.'+key))
    elif isinstance(value,list):
        for index,item in enumerate(value): issues.extend(review_issues(item,f'{path}[{index}]'))
    elif isinstance(value,str) and '[[' in value: issues.append(path)
    return issues


def bank_manifest(config):
    """Filename -> reviewed phrase; also used by the microphone-bank recorder."""
    manifest={}
    for response in config['responses'].values(): manifest[response['audio']]=response['text_ary']
    for slot in config['slots']:
        manifest[slot['prompt_audio']]=slot['prompt_text_ary']
        manifest[slot['readback_audio']]=slot.get('readback_text_ary',f"[[TODO_DARIJA: readback label for {slot['id']}]]")
    manifest.update({key:value['text_ary'] for key,value in config['audio_output']['fragments'].items()})
    settings=config['audio_output']
    required=[settings['connector_audio']]
    required += [settings['digit_pattern'].format(value=value) for value in range(10)]
    required += [settings['number_pattern'].format(value=value) for value in range(settings['number_max']+1)]
    for slot in config['slots']:
        required += [settings['enum_pattern'].format(value=value) for value in slot.get('values',[])]
        if slot['type'] in {'date','time'}: required.append(settings[slot['type']+'_separator_audio'])
    for name in required:
        if name not in manifest: manifest[name]=f'[[TODO_DARIJA: record {name}]]'
    return manifest


@dataclass
class RenderedAudio:
    wav: bytes
    total_samples: int
    synthetic_samples: int
    fragments: list[str]
    text: str | None = None


class AudioBank:
    def __init__(self, config, root, *, client=None):
        self.config=config
        self.settings=config['audio_output']
        self.runtime=config['runtime']
        self.directory=Path(root)/self.settings['audio_dir']/config['domain_id']
        self.client=client

    def path(self, filename):
        if not isinstance(filename,str) or Path(filename).name!=filename or not filename.endswith('.wav') or any(char in filename for char in '/\\:'):
            raise AudioBankError('Audio filenames must be plain .wav filenames.')
        return self.directory/filename

    def decode(self, data):
        try:
            with wave.open(io.BytesIO(data),'rb') as audio:
                expected=(self.runtime['sample_rate_hz'],self.runtime['channels'],self.runtime['sample_width_bytes'])
                if (audio.getframerate(),audio.getnchannels(),audio.getsampwidth())!=expected or audio.getcomptype()!='NONE':
                    raise AudioBankError('Audio must be 16 kHz mono PCM16 WAV.')
                pcm=audio.readframes(audio.getnframes())
                if not pcm or len(pcm)!=audio.getnframes()*self.runtime['sample_width_bytes']:
                    raise AudioBankError('Audio is empty or truncated.')
                return pcm
        except (wave.Error,EOFError) as exc: raise AudioBankError('Invalid WAV file.') from exc

    def preflight(self):
        issues=review_issues(self.config)+review_issues(bank_manifest(self.config),'manifest')
        if issues: raise AudioBankError(f'Language review is incomplete ({len(issues)} fields). Run scripts/check_config.py --ready.')
        missing=[name for name in bank_manifest(self.config) if not self.path(name).is_file()]
        if missing: raise AudioBankError(f'Audio bank is missing {len(missing)} WAV files. Run scripts/record_audio_bank.py.')
        for name in bank_manifest(self.config): self.decode(self.path(name).read_bytes())
        dynamic=[slot for slot in self.config['slots'] if slot['type'] in {'text','address'}]
        if dynamic and not (self.settings['tts_enabled'] and self.settings['tts_endpoint']):
            raise AudioBankError('Free-text readback requires the configured restricted TTS endpoint.')

    def plan(self, action, values):
        if action.kind=='listen': return []
        if action.kind=='ask':
            slot=next(slot for slot in self.config['slots'] if slot['id']==action.slot)
            return [slot['prompt_audio']]
        if action.kind!='readback': return [self.config['responses'][action.kind]['audio']]
        plan=[self.config['responses']['confirm_prefix']['audio']]
        filled=[slot for slot in sorted(self.config['slots'],key=lambda s:s['ask_order']) if slot['id'] in values]
        for index,slot in enumerate(filled):
            if index: plan.append(self.settings['connector_audio'])
            plan.append(slot['readback_audio'])
            value=values[slot['id']]; kind=slot['type']
            if kind in {'enum','enum_list'}:
                items=value if isinstance(value,list) else [value]
                for item_index,item in enumerate(items):
                    if item_index: plan.append(self.settings['connector_audio'])
                    plan.append(self.settings['enum_pattern'].format(value=item))
            elif kind=='integer':
                if 0<=value<=self.settings['number_max']: plan.append(self.settings['number_pattern'].format(value=value))
                else: raise AudioBankError('Integer is outside the recorded number bank.')
            elif kind in {'phone','date','time'}:
                # Critical phone values are always read digit by digit. Dates
                # use canonical ISO order, with separately reviewed separators.
                for char in value:
                    if char.isdecimal(): plan.append(self.settings['digit_pattern'].format(value=char))
                    elif kind in {'date','time'}: plan.append(self.settings[f'{kind}_separator_audio'])
            elif kind in {'text','address'} and slot.get('tts_allowed'):
                plan.append({'text':value,'spell':kind=='address','slot':slot['id']})
            else: raise AudioBankError(f"No approved readback for slot {slot['id']}.")
        plan.append(self.config['responses']['confirm_question']['audio'])
        return plan

    async def _tts(self, descriptor):
        if not self.settings['tts_enabled'] or not self.settings['tts_endpoint']:
            raise AudioBankError('Restricted TTS readback has not been configured.')
        payload={**descriptor,'language':self.config['language']}
        async def request(client):
            response=await client.post(self.settings['tts_endpoint'],json=payload,timeout=self.settings['tts_timeout_ms']/1000)
            response.raise_for_status()
            return self.decode(response.content)
        try:
            if self.client: return await request(self.client)
            async with httpx.AsyncClient() as client: return await request(client)
        except httpx.HTTPError as exc: raise AudioBankError('Restricted TTS provider failed.') from exc

    async def render(self, action, values):
        if review_issues(self.config) or review_issues(bank_manifest(self.config)): raise AudioBankError('Unreviewed language cannot be played.')
        pieces=[]; synthetic=0; names=[]
        gap=bytes(self.runtime['sample_rate_hz']*self.settings['gap_ms']//1000*self.runtime['sample_width_bytes'])
        for item in self.plan(action,values):
            if pieces: pieces.append(gap)
            if isinstance(item,dict):
                pcm=await self._tts(item); synthetic+=len(pcm)//self.runtime['sample_width_bytes']; names.append('tts:'+item['slot'])
            else:
                try: pcm=self.decode(self.path(item).read_bytes())
                except OSError as exc: raise AudioBankError(f'Missing audio: {item}') from exc
                names.append(item)
            pieces.append(pcm)
        pcm=b''.join(pieces)
        return RenderedAudio(wav_bytes(pcm,self.runtime),len(pcm)//self.runtime['sample_width_bytes'],synthetic,names)

    def save_recording(self, filename, source, *, overwrite=False):
        if review_issues(self.config) or review_issues(bank_manifest(self.config)): raise AudioBankError('Review all language before recording the bank.')
        if filename not in bank_manifest(self.config): raise AudioBankError('Unknown audio-bank filename.')
        data=Path(source).read_bytes()
        self.decode(data)
        self.directory.mkdir(parents=True,exist_ok=True)
        destination=self.path(filename)
        with destination.open('wb' if overwrite else 'xb') as output: output.write(data)
        return destination
