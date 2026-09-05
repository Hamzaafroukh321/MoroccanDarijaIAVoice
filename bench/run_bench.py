"""Replay owner-annotated audio through the actual voice pipeline."""

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import wave

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from engine.config import ROOT, load_config
from engine.endpointing import SileroVAD
from engine.normalize import canonical_state
from engine.pipeline import VoiceSession
from engine.responder import AudioBank, AudioBankError, review_issues
from engine.router import Router
from engine.stt import SpeechToText
from engine import lexicon
from bench.metrics import compute


def load_scenarios(config, root=ROOT):
    paths=sorted((root/'bench/scenarios'/config['domain_id']).glob('*.json'))
    scenarios=[]; issues=[]
    if len(paths)!=config['benchmark']['expected_scenarios']:
        issues.append(f"Expected {config['benchmark']['expected_scenarios']} scenario files; found {len(paths)}.")
    ids=set()
    for path in paths:
        try:
            scenario=json.loads(path.read_text(encoding='utf-8'))
            if scenario['id'] in ids: raise ValueError('duplicate scenario id')
            ids.add(scenario['id'])
            if scenario['domain']!=config['domain_id']: raise ValueError('wrong domain')
            if not scenario.get('annotations_reviewed') or review_issues(scenario): raise ValueError('owner annotation/review is pending')
            audio=(root/'bench'/scenario['audio']).resolve()
            if not audio.is_relative_to((root/'bench/recordings').resolve()): raise ValueError('audio must be under bench/recordings')
            with wave.open(str(audio),'rb') as recording:
                if (recording.getframerate(),recording.getnchannels(),recording.getsampwidth())!=(16000,1,2): raise ValueError('expected mono PCM16 at 16 kHz')
                duration=recording.getnframes()*1000/recording.getframerate()
            gold=canonical_state(scenario['gold_state'],config)
            if any(slot['required'] and gold.get(slot['id']) in (None,'',[]) for slot in config['slots']): raise ValueError('gold state lacks a required slot')
            utterances=scenario['utterances']
            if not utterances: raise ValueError('no annotated utterances')
            previous=-1
            for utterance in utterances:
                start=utterance['start_ms']; end=utterance['end_ms']
                if type(start) not in (int,float) or type(end) not in (int,float) or not 0<=start<end<=duration: raise ValueError('invalid utterance timestamps')
                if start<previous: raise ValueError('utterances must be ordered by start time')
                previous=start
            corrections=scenario['corrections']
            if any(item['is_correction'] for item in utterances) and not corrections: raise ValueError('correction flags require correction events')
            for correction in corrections:
                index=correction['utterance_index']
                if type(index) is not int or not 0<=index<len(utterances) or not utterances[index]['is_correction']: raise ValueError('correction points to an unmarked utterance')
                canonical_state({correction['slot']:correction['value']},config)
            scenario['_audio_path']=str(audio)
            # Required so relative date expressions never change between runs.
            if not isinstance(scenario.get('reference_date'),str): raise ValueError('reference_date (YYYY-MM-DD) is required')
            datetime.strptime(scenario['reference_date'],'%Y-%m-%d')
            scenarios.append(scenario)
        except (OSError,ValueError,KeyError,TypeError,wave.Error) as exc:
            issues.append(f'{path.name}: {exc}')
    return scenarios,issues


def provenance(config,scenarios,root=ROOT):
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    return {'config':deepcopy(config),'engine_sha256':{path.name:digest(path) for path in sorted((root/'engine').glob('*.py'))},'audio_sha256':{scenario['id']:digest(Path(scenario['_audio_path'])) for scenario in scenarios},'annotations_sha256':{scenario['id']:hashlib.sha256(json.dumps({k:v for k,v in scenario.items() if not k.startswith('_')},sort_keys=True).encode()).hexdigest() for scenario in scenarios},'created_at':datetime.now(timezone.utc).isoformat()}


async def run_scenario(scenario,config,stt,router,root=ROOT):
    config=deepcopy(config)
    config['normalization']['reference_date']=scenario['reference_date']
    bank=AudioBank(config,root); chunks=[]; timers=set(); current_playback=None
    async def emit(event):
        nonlocal current_playback
        if event['type']=='audio_start': chunks.clear(); current_playback=event['playback_id']
        elif event['type']=='audio_stop': current_playback=None
        elif event['type']=='audio_end':
            with wave.open(io.BytesIO(b''.join(chunks)),'rb') as audio: duration=audio.getnframes()/audio.getframerate()
            async def acknowledge(playback_id):
                await asyncio.sleep(duration)
                if current_playback==playback_id: await session.playback_finished(playback_id)
            task=asyncio.create_task(acknowledge(event['playback_id']));timers.add(task);task.add_done_callback(timers.discard)
    async def emit_audio(data): chunks.append(data)
    session=VoiceSession(config,root,stt,router,bank,SileroVAD(config),emit,emit_audio)
    try:
        await session.start()
        # Recordings start after the greeting; playback during replay is timed.
        if session.playback_task: await session.playback_task
        with wave.open(scenario['_audio_path'],'rb') as audio: pcm=audio.readframes(audio.getnframes())
        size=session.detector.frame_bytes
        pcm+=bytes((-len(pcm))%size)
        pcm+=bytes(config['runtime']['sample_rate_hz']*config['benchmark']['tail_silence_ms']//1000*config['runtime']['sample_width_bytes'])
        started=time.monotonic()
        for index in range(0,len(pcm),size):
            if session.done.is_set(): break
            await session.feed(pcm[index:index+size])
            target=started+(index//size+1)*config['runtime']['frame_ms']/1000
            await asyncio.sleep(max(0,target-time.monotonic()))
        async with asyncio.timeout(config['benchmark']['drain_timeout_ms']/1000):
            await session.queue.join()
            if session.playback_task: await session.playback_task
    finally:
        await session.close(status='incomplete')
        for task in timers: task.cancel()
        await asyncio.gather(*timers,return_exceptions=True)
    result=session.save();result['scenario_id']=scenario['id']
    return result


async def run_suite(config,scenarios,*,primary='moulsot',root=ROOT):
    if lexicon.NEEDS_HUMAN_REVIEW: raise RuntimeError('Review the Darija lexicon before running a measured benchmark.')
    AudioBank(config,root).preflight()
    stt=SpeechToText(config,root,primary=primary,fallback=False)
    router=Router(config,root)
    results=[]
    try:
        for scenario in scenarios:
            print(f"Running {config['domain_id']}/{scenario['id']} [{primary}]",flush=True)
            result=await run_scenario(scenario,config,stt,router,root)
            results.append(result)
    finally:
        await stt.close();await router.close()
    if any(not call['ok'] for result in results for call in result.get('stt_calls',[])):
        raise RuntimeError('Provider failures occurred; session logs were saved but no comparison score will be published.')
    if any(call['provider_failure'] for result in results for call in result.get('router_calls',[])):
        raise RuntimeError('Router transport failures occurred; session logs were saved but no comparison score will be published.')
    return {'metrics':compute(scenarios,results,config),'results':results,'provenance':provenance(config,scenarios,root),'primary':primary}


def print_metrics(metrics):
    for key in ('TSR','FCR','CSR','CEA','MEL'):
        value=metrics.get(key)
        print(f'{key}: '+(f'{value:.3f}' if value is not None else 'N/A — not measured or no applicable cases'))


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain',choices=['pizza','clinic'],default='pizza')
    parser.add_argument('--check',action='store_true',help='Validate owner annotations and audio without network calls.')
    parser.add_argument('--primary',choices=['moulsot','groq'],default='moulsot')
    args=parser.parse_args();load_dotenv(ROOT/'.env')
    config=load_config(ROOT/'configs'/f'{args.domain}.json')
    scenarios,issues=load_scenarios(config)
    try: AudioBank(config,ROOT).preflight()
    except (AudioBankError,OSError) as exc: issues.append(str(exc))
    if issues:
        print('NOT RUN — required research inputs are missing.')
        print_metrics({})
        for issue in issues: print(' - '+issue)
        return 2
    if args.check: print(f'Ready: {len(scenarios)} validated scenarios.');return 0
    try: report=await run_suite(config,scenarios,primary=args.primary)
    except (RuntimeError,OSError) as exc: print(f'NOT SCORED: {exc}');return 2
    filename=f"{args.domain}_{args.primary}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}.json"
    (ROOT/'bench/results'/filename).write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print_metrics(report['metrics'])
    print('Results: bench/results/'+filename)
    return 0


if __name__=='__main__': raise SystemExit(asyncio.run(main()))
