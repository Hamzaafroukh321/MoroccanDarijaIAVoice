"""Bounded opt-in comparison using two saved draft clinic states, no ASR calls.

Defaults to a dry plan. --execute allows at most three new XTTS generations;
stops on the first failure. Isolated cache preserves the existing full renders.
"""
import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from bench.voice_transport import verified_wav
from engine.config import load_config
from engine.demo import DemoVoice, demo_config, readback_groups, reply_text, speech_cache_path
from engine.state import Action


class BoundedVoice(DemoVoice):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated = 0

    async def _produce_part(self, part, path, settings):
        if self.generated >= 3:
            raise RuntimeError('Three-generation experiment limit reached.')
        self.generated += 1
        return await super()._produce_part(part, path, settings)


async def compare(execute):
    load_dotenv(ROOT / '.env')
    config = demo_config(load_config(ROOT / 'configs/clinic.json'))
    if config['demo']['tts_provider'] != 'darija_xtts':
        raise ValueError('This experiment requires the selected Darija XTTS provider.')
    config['demo']['tts_readback_mode'] = 'grouped'
    config['demo']['tts_readback_group_chars'] = 96
    states = [dict(doctor='doctor_a', date='2026-09-08', time=t) for t in ('10:30', '11:00')]
    plans = [readback_groups(state, config['demo'], 96) for state in states]
    unique = list(dict.fromkeys(part for plan in plans for part in plan))
    report = dict(source_kind='synthetic_diagnostic', evaluation_eligible=False,
        native_review='not reviewed', microphone_capture=False, real_time_playback=False,
        purpose='Measure full response assembly and reuse after one field changes.',
        default_changed=False, max_generations=3, group_chars=96, plans=plans,
        unique_characters=sum(map(len, unique)), unique_generations=len(unique), renders=[],
        historical_full_render_ms=8315.011,
        historical_comparison='Different request time/load; not a controlled average.',
        status='dry_run')
    if len(unique) > 3 or sum(map(len, unique)) > 160:
        raise ValueError('Unexpected plan exceeds the experiment character/call bounds.')
    if not execute:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    folder = ROOT / 'bench/results' / ('readback_groups_' + uuid.uuid4().hex)
    folder.mkdir()
    config['demo']['cache_dir'] = str(folder / 'cache')
    voice = BoundedVoice(config, ROOT)
    try:
        for index, state in enumerate(states):
            rendered = await voice.render(Action('readback'), deepcopy(state))
            _, metadata = verified_wav(rendered.wav)
            expected = reply_text(Action('readback'), state, config['demo'])
            if rendered.text != expected:
                raise ValueError('Readback text changed during grouping.')
            path = folder / f'readback_{index}.wav'
            path.write_bytes(rendered.wav)
            report['renders'].append(dict(state=state, text=rendered.text,
                audio=metadata, path=str(path), sha256=hashlib.sha256(rendered.wav).hexdigest(),
                calls=deepcopy(voice.calls[-1]),
                full_text_cache_created=speech_cache_path(config['demo'], ROOT, expected).exists()))
            print(json.dumps(report['renders'][-1], ensure_ascii=False), flush=True)
        # Cache-only reuse is measured separately, never described as inference.
        rendered = await voice.render(Action('readback'), deepcopy(states[1]))
        report['repeat_cached'] = deepcopy(voice.calls[-1])
        report['status'] = 'completed'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        await voice.close()
        report['generations_started'] = voice.generated
        (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(str(folder / 'report.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(compare(args.execute), timeout=150))
