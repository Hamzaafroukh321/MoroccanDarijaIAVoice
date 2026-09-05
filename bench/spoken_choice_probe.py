"""One new synthetic spoken answer after a saved coupled-choice question.

Dry by default. Execution makes at most one MoulSot call, one Groq call, one
input synthesis and one reply render (normally cached). This is a whole-file
diagnostic, not a microphone conversation or native accuracy evaluation.

Execution requires the local SOURCE report below and its original audio, plus
configured providers and the MoulSot bridge on port 8012. Those private diagnostic
artifacts are intentionally excluded from Git; a fresh checkout supports dry mode.
"""
import argparse
import asyncio
from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from bench.smoke_moulsot import read_audio
from engine.config import load_config
from engine.demo import DemoVoice, demo_config
from engine.normalize import normalize
from engine.router import Router, parse_response
from engine.scoped_task import ScopedTaskState
from engine.state import Action
from engine.stt import SpeechToText


SOURCE = ROOT/'bench/results/clinic_audio_turn_20260905_223944.json'
ANSWER = 'الطبيب باء'


async def probe(execute):
    if not execute:
        print(json.dumps(dict(source=str(SOURCE), draft_synthetic_answer=ANSWER,
            max_asr_calls=1, max_router_calls=1, max_input_renders=1, max_reply_renders=1,
            evaluation_eligible=False), ensure_ascii=False, indent=2))
        return
    load_dotenv(ROOT/'.env')
    os.environ['DEMO_TTS_PROVIDER'] = 'darija_xtts'
    os.environ['MOULSOT_PROTOCOL'] = 'json'
    config = demo_config(load_config(ROOT/'configs/clinic.json'))
    config['router']['retries'] = 0
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    if source.get('domain') != 'clinic' or source.get('evaluation_eligible') is not False:
        raise ValueError('Expected the saved unreviewed clinic diagnostic.')
    original_audio = Path(source['source'])
    if hashlib.sha256(original_audio.read_bytes()).hexdigest() != source['source_sha256']:
        raise ValueError('Saved question input has changed.')
    task = ScopedTaskState(config)
    task.consume(parse_response(json.dumps(source['router']), config).model_dump())
    if (task.values or not task.pending_proposal or
            set(task.pending_proposal.get('remaining_slots', [])) != {'doctor', 'time'}):
        raise ValueError('Saved reply did not open the expected coupled question.')
    context = task.router_context()
    context.update(pending_request=dict(id='request_1', text=normalize(source['transcript']['text'], config),
        truncated=False), last_assistant_action=dict(kind='ambiguous_value', slot='doctor', item_id=None),
        last_assistant_text=source['spoken_text'], readback_complete=False)
    directory = ROOT/'bench/results'/('spoken_choice_' + uuid4().hex)
    directory.mkdir(parents=True)
    report = dict(kind=__doc__, evaluation_eligible=False, source_report=str(SOURCE),
                  source_report_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  synthetic_input_text=ANSWER, passed=False, context_before=deepcopy(context))
    input_config = deepcopy(config)
    input_config['demo']['responses']['diagnostic_input'] = ANSWER
    input_voice, reply_voice = DemoVoice(input_config, ROOT), DemoVoice(config, ROOT)
    stt = SpeechToText(config, ROOT, primary='moulsot', fallback=False,
                       endpoint='http://127.0.0.1:8012/transcribe')
    router = Router(config, ROOT)
    try:
        audio = await input_voice.render(Action('diagnostic_input'), {})
        path = directory/'synthetic_answer.wav'
        path.write_bytes(audio.wav)
        pcm, seconds = read_audio(path)
        report.update(input_seconds=seconds, input_sha256=hashlib.sha256(audio.wav).hexdigest())
        transcript = await stt.transcribe(pcm)
        report['transcript'] = asdict(transcript)
        response = await router.route(normalize(transcript.text, config), context)
        action = task.consume(response.model_dump())
        report.update(response=response.model_dump(), action=asdict(action), context_after=task.router_context())
        report['passed'] = (task.values == {} and not task.confirmed and task.pending_proposal is not None
            and task.pending_proposal['state'].get('doctor') == 'doctor_b'
            and task.pending_proposal.get('remaining_slots') == ['time']
            and task.pending_clarification['id'] != context['pending_clarification']['id'])
        reply = await reply_voice.render(action, task.values)
        (directory/'reply.wav').write_bytes(reply.wav)
        report['spoken_text'] = reply.text
    except Exception as exc:
        report['error_type'] = type(exc).__name__
    finally:
        await input_voice.close(); await reply_voice.close(); await stt.close(); await router.close()
        report.update(input_tts_calls=input_voice.calls, reply_tts_calls=reply_voice.calls,
                      stt_calls=stt.calls, router_calls=router.calls)
        (directory/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(dict(passed=report['passed'], transcript=report.get('transcript'),
                              spoken_text=report.get('spoken_text'), error_type=report.get('error_type')),
                         ensure_ascii=False))
        print(directory/'report.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(probe(args.execute), timeout=240))
