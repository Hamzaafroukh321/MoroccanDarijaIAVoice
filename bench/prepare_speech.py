"""Prepare a bounded set of exact runtime demo replies; dry-run by default.

No readback combinations, ASR calls, or router inference are generated here.
"""

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import ROOT, load_config
from engine.demo import DemoVoice, chunks, demo_config, reply_text
from engine.state import Action


MAX_MISSES = 20
MAX_CHARACTERS = 3000


def fixed_replies(config):
    """Use the renderer itself as the source of text, retaining action provenance."""
    settings = config['demo']
    collection = settings.get('state_kind') == 'collection_scoped'
    values = {'items': []} if collection else {}
    required = [slot for slot in config['slots'] if slot['required']]
    actions = [Action('greeting')]
    kinds = ('ask', 'repair', 'unintelligible')
    if not collection:
        kinds += ('ambiguous_value',)
    for kind in kinds:
        for slot in required:
            # Drinks are order-level; only the first pizza item is pre-rendered.
            item_id = 1 if collection and slot['id'] != 'drink' else None
            actions.append(Action(kind, slot['id'], item_id))
    if 'unsupported_option' in settings['responses']:
        for slot in config['slots']:
            if settings.get('menu_options', {}).get(slot['id']):
                actions.append(Action('unsupported_menu' if collection else 'unsupported_value', slot['id']))
    for kind in ('drink_size', 'unsupported_drink', 'listen', 'accepted', 'handoff'):
        if kind in settings['responses']:
            actions.append(Action(kind))
    unique = {}
    for action in actions:
        text = reply_text(action, values, settings)
        if text not in unique:
            unique[text] = {'text': text, 'actions': [], 'values': values.copy()}
        unique[text]['actions'].append(asdict(action))
    return list(unique.values())


def inspect_reply(reply, config, root):
    # These shared helpers preserve runtime identity and WAV validation.
    from engine.demo import read_speech_cache, speech_cache_path
    fragments = []
    for part in chunks(reply['text'], config['demo']['tts_max_chars']):
        path = speech_cache_path(config['demo'], root, part)
        valid = read_speech_cache(path, config['runtime']) is not None
        fragments.append({'text': part, 'characters': len(part), 'path': str(path),
                          'cache': 'valid' if valid else ('corrupt' if path.exists() else 'absent')})
    return fragments


async def prepare_speech(config, root, *, execute=False, max_misses=3,
                         max_characters=500, voice_factory=DemoVoice):
    """Reserve full replies before rendering; budgets count potential new fragments.

    A failed reply may have cached earlier fragments. It still consumes its full
    reservation and stops this run. Provider exception details are never exported.
    """
    if type(max_misses) is not int or not 0 <= max_misses <= MAX_MISSES:
        raise ValueError(f'max_misses must be between 0 and {MAX_MISSES}.')
    if type(max_characters) is not int or not 0 <= max_characters <= MAX_CHARACTERS:
        raise ValueError(f'max_characters must be between 0 and {MAX_CHARACTERS}.')
    root = Path(root).resolve()
    report = {'domain': config['domain_id'], 'mode': 'execute' if execute else 'dry_run',
              'provider': config['demo']['tts_provider'], 'model': config['demo']['tts_model'],
              'voice': config['demo']['tts_voice'],
              'provenance': 'Exact engine.demo.reply_text output from fixed Action inputs; draft synthetic demo language, not native-reviewed speech or an accuracy evaluation.',
              'limits': {'max_misses': max_misses, 'max_characters': max_characters},
              'reserved_misses': 0, 'reserved_characters': 0,
              'stopped_on_failure': False, 'replies': []}
    voice = None
    planned_paths = set()
    try:
        for reply in fixed_replies(config):
            fragments = inspect_reply(reply, config, root)
            entry = {**reply, 'fragments_before': fragments,
                     'missing_fragments': sum(f['cache'] != 'valid' for f in fragments),
                     'missing_characters': sum(f['characters'] for f in fragments if f['cache'] != 'valid'),
                     'cache_hit': all(f['cache'] == 'valid' for f in fragments)}
            report['replies'].append(entry)
            if entry['cache_hit']:
                entry['status'] = 'cached'
                continue
            if report['stopped_on_failure']:
                entry['status'] = 'skipped_after_failure'
                continue
            # Shared fragments are one reservation, even across different replies.
            missing = {f['path']: f for f in fragments if f['cache'] != 'valid'
                       and (execute or f['path'] not in planned_paths)}
            count, characters = len(missing), sum(f['characters'] for f in missing.values())
            if (report['reserved_misses'] + count > max_misses or
                    report['reserved_characters'] + characters > max_characters):
                entry['status'] = 'skipped_budget'
                continue
            report['reserved_misses'] += count
            report['reserved_characters'] += characters
            planned_paths.update(missing)
            if not execute:
                entry['status'] = 'would_prepare'
                continue
            try:
                if voice is None:
                    voice = voice_factory(config, root)
                await voice.render(Action(**reply['actions'][0]), reply['values'])
                if any(f['cache'] != 'valid' for f in inspect_reply(reply, config, root)):
                    raise RuntimeError('Runtime render did not produce complete valid cache entries.')
                entry['status'] = 'prepared'
            except Exception as exc:
                entry['status'] = 'failed'
                entry['error_type'] = type(exc).__name__
                report['stopped_on_failure'] = True
    finally:
        if voice is not None:
            await voice.close()
            report['tts_calls'] = list(getattr(voice, 'calls', []))
    for entry in report['replies']:
        entry['fragments_after'] = inspect_reply(entry, config, root)
        entry['available'] = all(f['cache'] == 'valid' for f in entry['fragments_after'])
    report['available_texts'] = [entry['text'] for entry in report['replies'] if entry['available']]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=('pizza', 'clinic'), required=True)
    parser.add_argument('--execute', action='store_true', help='Generate only complete replies within both budgets.')
    parser.add_argument('--max-misses', type=int, default=3, help='Maximum missing fragments to reserve (0–20).')
    parser.add_argument('--max-characters', type=int, default=500, help='Maximum missing-fragment characters (0–3000).')
    parser.add_argument('--output', type=Path, help='JSON report path (default: timestamped bench/results file).')
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    config = demo_config(load_config(ROOT / 'configs' / f'{args.domain}.json'))
    try:
        report = asyncio.run(prepare_speech(config, ROOT, execute=args.execute,
                                           max_misses=args.max_misses, max_characters=args.max_characters))
    except ValueError as exc:
        parser.error(str(exc))
    report['created_at'] = datetime.now(timezone.utc).isoformat()
    output = args.output or ROOT / 'bench/results' / (
        'speech_preparation_' + args.domain + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(output), 'mode': report['mode'],
                      'available_replies': len(report['available_texts']),
                      'reserved_misses': report['reserved_misses'],
                      'reserved_characters': report['reserved_characters'],
                      'stopped_on_failure': report['stopped_on_failure']}, ensure_ascii=False))
    return 1 if report['stopped_on_failure'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
