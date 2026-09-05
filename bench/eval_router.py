"""Text-only live router evaluation; owner-reviewed gold data, no audio required."""

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
from engine.config import ROOT, load_config
from engine.normalize import canonical_state, normalize
from engine.router import Router, check_models, parse_response
from engine.state import TaskState

MODELS = ['openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'qwen/qwen3.6-27b']
FLAGS = ('unclear', 'is_affirmation', 'is_negation')


def validate_cases(data, config):
    cases = data.get('cases', [])
    if len(cases) != 30 or Counter(c.get('category') for c in cases) != {
        'straightforward': 10, 'correction': 10, 'messy': 10,
    }:
        raise ValueError('Expected 30 cases: 10 straightforward, 10 correction, 10 messy.')
    ids = [case['id'] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError('Case IDs must be unique.')
    for case in cases:
        if case.get('reviewed') is not True or not isinstance(case.get('transcript'), str) or not case['transcript'].strip():
            raise ValueError(f"{case['id']}: supply a transcript and review its expected operations/flags first.")
        canonical_state(case['state'], config)
        parse_response(json.dumps({**case['expected'], 'confidence': 1.0}), config)
    return cases


def matches(result, expected):
    # Exact operation order and values; confidence is not a gold label.
    return all(result[key] == expected[key] for key in ('ops', *FLAGS))


def summarize(rows):
    valid = [row for row in rows if row['valid']]
    latencies = sorted(row['elapsed_ms'] for row in valid)
    return {
        'cases': len(rows), 'exact_ops_and_flags_percent': 100 * sum(row['passed'] for row in rows) / len(rows),
        'valid_response_percent': 100 * len(valid) / len(rows),
        'provider_failure_cases': sum(row.get('provider_failure', False) for row in rows),
        'retries': sum(max(0, len(row.get('attempts', [])) - 1) for row in rows),
        'successful_latency_p50_ms': statistics.median(latencies) if latencies else None,
        'successful_latency_p95_ms': latencies[math.ceil(.95 * len(latencies)) - 1] if latencies else None,
        'by_category': {category: {'passed': sum(r['passed'] for r in rows if r['category'] == category),
                                  'total': sum(r['category'] == category for r in rows)}
                        for category in sorted({r['category'] for r in rows})},
    }


async def evaluate(cases, config, models, output, *, probe=False, interval=3.2):
    report = {'kind': 'protocol_smoke_not_accuracy' if probe else 'owner_reviewed_text_router_eval',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'dataset_sha256': hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
              'engine_sha256': {name: hashlib.sha256((ROOT/'engine'/name).read_bytes()).hexdigest()
                                for name in ('router.py', 'normalize.py', 'state.py', 'lexicon.py')},
              'comparison': 'exact ordered ops and three flags; confidence excluded; failed calls count as failures',
              'models': {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        temporary.replace(output)
    for model in models:
        selected = deepcopy(config)
        selected['router']['model'] = model  # Explicit eval candidate overrides the loaded env default.
        entry = report['models'][model] = {'response_format': 'json_schema', 'strict': True,
                                          'config': selected, 'status': 'running', 'rows': []}
        router = Router(selected, ROOT)
        try:
            await check_models([model], router.api_key)
            for case in cases:
                before = len(router.calls)
                started = time.monotonic()
                row = {'id': case['id'], 'category': case['category'], 'passed': False, 'valid': False}
                try:
                    transcript = normalize(case['transcript'], selected)
                    result = (await router.route(transcript, case['state'])).model_dump()
                    attempts = router.calls[before:]
                    row.update({'normalized_transcript': transcript, 'result': result,
                                'expected': case['expected'], 'valid': bool(attempts and attempts[-1]['ok'])})
                    row['passed'] = row['valid'] and matches(result, case['expected'])
                    if row['valid']:
                        state = TaskState(selected)
                        state.apply([{'op': 'set', 'slot': k, 'value': v} for k, v in case['state'].items()])
                        state.apply(result['ops'])
                        row['resulting_state'] = state.values
                except (ValueError, RuntimeError, OSError) as exc:
                    row['error_type'] = type(exc).__name__
                row['elapsed_ms'] = (time.monotonic() - started) * 1000
                row['attempts'] = router.calls[before:]
                row['provider_failure'] = any(attempt['provider_failure'] for attempt in row['attempts'])
                entry['rows'].append(row)
                save()
                print(model, case['id'], 'PASS' if row['passed'] else 'FAIL', flush=True)
                # Respect the shared quota ledger; never reset it for evaluation.
                if any(a.get('http_status') == 400 for a in row['attempts']):
                    entry['status'] = 'blocked_http_400_check_request_compatibility'
                    break
                await asyncio.sleep(interval)
            else:
                entry['status'] = 'complete'
            if not probe and entry['status'] == 'complete':
                entry['summary'] = summarize(entry['rows'])
        except (RuntimeError, ValueError, OSError) as exc:
            entry.update(status='blocked', error=str(exc))
        finally:
            await router.close()
            save()
    return report


async def main():
    load_dotenv(ROOT/'.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=ROOT/'bench/router_cases/pizza.json')
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--check', action='store_true', help='Validate gold cases without API calls.')
    parser.add_argument('--probe', action='store_true', help='One English protocol check per model; NOT a Darija accuracy eval.')
    args = parser.parse_args()
    config = load_config(ROOT/'configs/pizza.json')
    try:
        cases = [{'id': 'protocol', 'category': 'protocol', 'state': {}, 'transcript': 'I want a large pizza.',
                  'expected': {'ops': [{'op': 'set', 'slot': 'size', 'value': 'large'}],
                               'unclear': False, 'is_affirmation': False, 'is_negation': False}}] if args.probe else validate_cases(
                                   json.loads(args.cases.read_text(encoding='utf-8')), config)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print('NOT RUN:', exc)
        return 2
    if args.check:
        print('READY:', len(cases), 'cases')
        return 0
    output = ROOT/'bench/results'/f"router_{'probe' if args.probe else 'eval'}_{uuid4().hex}.json"
    report = await evaluate(cases, config, args.models, output, probe=args.probe)
    print('Saved:', output)
    return 0 if all(m['status'] == 'complete' and all(r['passed'] for r in m['rows']) for m in report['models'].values()) else 1


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
