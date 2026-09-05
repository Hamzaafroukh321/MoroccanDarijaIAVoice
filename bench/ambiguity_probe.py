"""Bounded English engineering diagnostics, not the owner's Darija gold eval.

Dry by default. --execute sends at most seven Groq requests, no ASR/TTS.
Each case starts from a fresh state. No accuracy percentage is computed.
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
from engine.config import load_config
from engine.demo import demo_config
from engine.normalize import normalize
from engine.router import Router
from engine.scoped_task import ScopedTaskState

CASES = [
    dict(id='dependent_pairs', text='Doctor A at 10:00 or doctor B at 11:00.', contract='dependent', state={}),
    dict(id='bare_time_alternatives', text='10:00 or 11:00', contract='time_question',
         state=dict(doctor='doctor_a', date='2026-09-15')),
    dict(id='time_correction', text='Not 10:00; change the time to 11:00.', contract='correction',
         state=dict(doctor='doctor_a', date='2026-09-15', time='10:00')),
    dict(id='independent_date_uncertain_time', text='Doctor A, date 2026-09-15, time 10:00 or 11:00.',
         contract='hold_except_time', state={}),
    dict(id='independent_time_uncertain_doctor', text='Doctor A or doctor B. Date 2026-09-15. Time 14:00.',
         contract='hold_except_choice', state={}),
    dict(id='custom_dependent_pairs', text='Counter A at 10:00 or counter B at 11:00.',
         contract='dependent', state={}, custom=True),
    dict(id='custom_independent_facts', text='Counter A or counter B. Date 2026-09-15. Time 14:00.',
         contract='hold_except_choice', state={}, custom=True),
]


def check(case, state, response, action):
    contract = case['contract']
    choice = 'counter' if case.get('custom') else 'doctor'
    body = response.model_dump()
    scope = body.get('clarification') or {}
    held = (state.pending_proposal or {}).get('state', {})
    if contract == 'correction':
        return action.kind == 'readback' and state.values == {**case['state'], 'time': '11:00'} and not state.confirmed
    if state.values != case['state'] or body['ops'] or state.confirmed or body['intent'] != 'ambiguous':
        return False
    if contract == 'dependent':
        return not body['proposed_ops'] and scope.get('slot') in {choice, 'time'}
    if contract == 'time_question':
        return scope.get('slot') == 'time' and not body['proposed_ops']
    if contract == 'hold_except_time':
        return scope.get('slot') == 'time' and held == {choice: choice+'_a', 'date': '2026-09-15'}
    return scope.get('slot') == choice and held == {'date': '2026-09-15', 'time': '14:00'}


async def probe(execute, selected_ids=None):
    cases = [case for case in CASES if selected_ids is None or case['id'] in selected_ids]
    if not execute:
        print(json.dumps(dict(kind='unreviewed_English_engineering_cases', cases=cases,
                             max_requests=len(cases), evaluation_eligible=False), indent=2))
        return
    load_dotenv(ROOT / '.env')
    base = demo_config(load_config(ROOT / 'configs/clinic.json'))
    base['router']['retries'] = 0
    output = ROOT / 'bench/results' / ('ambiguity_probe_' + uuid.uuid4().hex + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(kind='unreviewed_English_engineering_cases', evaluation_eligible=False,
        asr_calls=0, tts_calls=0, max_requests=len(cases), rows=[], status='started',
        source_hashes={name: hashlib.sha256((ROOT/'engine'/name).read_bytes()).hexdigest()
                       for name in ('router.py', 'enum_grounding.py', 'temporal_grounding.py', 'dialogue.py', 'scoped_task.py')})
    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    try:
        for case in cases:
            config = deepcopy(base)
            if case.get('custom'):
                config = json.loads(json.dumps(config).replace('doctor', 'counter'))
                config['domain_id'] = 'counter_diagnostic'
                config['demo']['task_scope'] = 'fictional counter, preferred date and time only; no reservation'
            state = ScopedTaskState(config)
            if case['state']:
                state.apply([dict(op='set', slot=k, value=v) for k,v in case['state'].items()])
            router = Router(config, ROOT)
            row = dict(case=case, status='started')
            try:
                response = await router.route(normalize(case['text'], config), state.router_context())
                action = state.consume(response.model_dump())
                row.update(response=response.model_dump(), state=state.router_context(), action=action.kind,
                           contract_satisfied=check(case, state, response, action), status='routed')
            except Exception as exc:
                row.update(status='failed', error_type=type(exc).__name__, contract_satisfied=False)
            finally:
                row['calls'] = router.calls
                await router.close()
                report['rows'].append(row)
                save()
            print(json.dumps(dict(id=case['id'], status=row['status'], contract_satisfied=row['contract_satisfied']),
                             ensure_ascii=False), flush=True)
            if any(call.get('provider_failure') or call.get('http_status') for call in row['calls']):
                report['status'] = 'stopped_provider_failure'
                break
        else:
            report['status'] = 'completed'
    finally:
        save()
        print(str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--case', action='append', choices=[case['id'] for case in CASES],
                        help='Select only unresolved cases; repeat to select more than one.')
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(probe(args.execute, args.case), timeout=150))
