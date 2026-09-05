"""At most three Groq text calls for linked choices; dry by default, no ASR/TTS.

English engineering dialogue with configured draft assistant text. No audio is
played and no native accuracy, availability or booking is evaluated.
"""
import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from engine.config import load_config
from engine.demo import demo_config, reply_text
from engine.normalize import normalize
from engine.router import Router
from engine.scoped_task import ScopedTaskState


INITIAL = {'doctor': 'doctor_a', 'date': '2026-09-15', 'time': '10:00'}
UTTERANCES = ['Doctor A at 10:00 or doctor B at 11:00.', 'Doctor B.', '11:00.']


async def probe(execute):
    if not execute:
        print(json.dumps(dict(initial=INITIAL, utterances=UTTERANCES, max_requests=3,
                              evaluation_eligible=False), indent=2))
        return
    load_dotenv(ROOT / '.env')
    config = demo_config(load_config(ROOT / 'configs/clinic.json'))
    config['router']['retries'] = 0
    task = ScopedTaskState(config)
    task.apply([dict(op='set', slot=key, value=value) for key, value in INITIAL.items()])
    report = dict(kind='English engineering linked-choice text diagnostic',
                  evaluation_eligible=False, asr_calls=0, tts_calls=0, max_requests=3,
                  initial=INITIAL, turns=[], passed=False,
                  source_hashes={name: hashlib.sha256((ROOT/'engine'/name).read_bytes()).hexdigest()
                                 for name in ('router.py', 'scoped_task.py', 'dialogue.py', 'pipeline.py')})
    output = ROOT/'bench/results'/('coupled_choice_' + uuid4().hex + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    router = Router(config, ROOT)
    pending_request, last_action, last_text = None, None, None
    try:
        for index, text in enumerate(UTTERANCES):
            context = task.router_context()
            context.update(pending_request=deepcopy(pending_request), last_assistant_action=last_action,
                           last_assistant_text=last_text, readback_complete=False)
            response = await router.route(normalize(text, config), context)
            action = task.consume(response.model_dump())
            report['turns'].append(dict(text=text, response=response.model_dump(), action=action.kind,
                                       context_after=task.router_context(), confirmed=task.confirmed))
            if index == 0:
                proposal = task.pending_proposal or {}
                if (task.values != INITIAL or set(proposal.get('remaining_slots', [])) != {'doctor', 'time'}):
                    report['failure'] = 'Initial dependent fields were not explicitly tracked.'
                    break
                pending_request = dict(id='request_1', text=text, truncated=False)
            elif task.pending_clarification is None:
                report['passed'] = (task.values == {**INITIAL, 'doctor': 'doctor_b', 'time': '11:00'}
                                    and action.kind == 'readback' and not task.confirmed)
                if not report['passed']:
                    report['failure'] = 'Resolution lost or changed a linked value.'
                break
            elif (task.values != INITIAL or task.confirmed or task.pending_clarification['slot'] != 'time'):
                report['failure'] = 'Partial answer changed committed values or lost the next question.'
                break
            last_action = dict(kind=action.kind, slot=action.slot, item_id=action.item_id)
            last_text = reply_text(action, task.values, config['demo'])
    except Exception as exc:
        report['error_type'] = type(exc).__name__
    finally:
        report['calls'] = router.calls
        await router.close()
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(dict(passed=report['passed'], calls=len(router.calls),
                              failure=report.get('failure'), error_type=report.get('error_type'))))
        print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(probe(args.execute), timeout=120))
