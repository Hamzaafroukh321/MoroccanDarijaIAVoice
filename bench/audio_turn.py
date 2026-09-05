"""One recorded or explicitly synthetic input through actual speech providers.

Whole-file adapter diagnostic: not browser endpointing, microphone capture,
device playback, a complete order, or a native-reviewed accuracy evaluation.
"""
import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import time
import wave

from dotenv import load_dotenv

from bench.smoke_moulsot import read_audio
from engine.config import ROOT, load_config
from engine.demo import DemoVoice, demo_config
from engine.demo_order import DemoOrderState
from engine.scoped_task import ScopedTaskState
from engine.state import Action
from engine.normalize import normalize
from engine.router import Router
from engine.stt import SpeechToText, wav_bytes
from engine.stt import Transcript


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain',choices=['pizza','clinic'],default='pizza')
    inputs=parser.add_mutually_exclusive_group()
    inputs.add_argument('--audio',help='Existing 16kHz mono PCM WAV input.')
    inputs.add_argument('--synthetic-text',help='Draft text synthesized as diagnostic input; never a human speech evaluation.')
    inputs.add_argument('--transcript-report',help='Reuse an earlier report and its audio/transcript; makes no new ASR call.')
    args=parser.parse_args()
    load_dotenv(ROOT/'.env')
    config=demo_config(load_config(ROOT/'configs'/f'{args.domain}.json'))
    config['router']['retries']=0
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    prefix=ROOT/'bench/results'/(args.domain+'_audio_turn_'+stamp)
    report={'purpose':__doc__,'evaluation_eligible':False,'domain':args.domain,
            'source_kind':'synthetic_diagnostic' if args.synthetic_text else 'existing_recording',
            'status':'started','timings_ms':{},'device_playback_tested':False,'native_review':'pending'}
    def save():
        prefix.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    stt=SpeechToText(config,ROOT,primary='moulsot',fallback=False)
    router=Router(config,ROOT)
    voice=DemoVoice(config,ROOT)
    state=ScopedTaskState(config) if config['demo'].get('state_kind')=='flat_scoped' else DemoOrderState(config)
    try:
        save()
        previous=None
        if args.transcript_report:
            previous_path=ROOT/args.transcript_report
            previous=json.loads(previous_path.read_text(encoding='utf-8'))
            if previous.get('domain')!=args.domain:
                raise ValueError('Transcript report domain does not match.')
            source=ROOT/previous['source']
            if hashlib.sha256(source.read_bytes()).hexdigest()!=previous['source_sha256']:
                raise ValueError('Original audio changed since the transcript was recorded.')
            report.update(source_kind=previous['source_kind'],reused_transcript_from=str(previous_path),
                          new_asr_call=False)
        elif args.synthetic_text:
            source=prefix.with_name(prefix.name+'_synthetic_input.wav')
            input_config=deepcopy(config)
            input_config['demo']['responses']['diagnostic_input']=args.synthetic_text
            input_voice=DemoVoice(input_config,ROOT)
            started=time.monotonic()
            try: generated=await input_voice.render(Action('diagnostic_input'),{})
            finally: await input_voice.close()
            source.write_bytes(generated.wav)
            report['timings_ms']['synthetic_input_tts']=(time.monotonic()-started)*1000
            report['synthetic_input_text']=args.synthetic_text
        else:
            source=ROOT/args.audio if args.audio else ROOT/'bench/recordings/capture_13867675871d4e1bb19950bbe4eb7d51.wav'
        pcm,seconds=read_audio(source)
        report.update(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input_seconds=seconds)
        save()
        started=time.monotonic()
        if previous:
            transcript=Transcript(**previous['transcript'])
        else:
            transcript=await stt.transcribe(pcm)
            report['timings_ms']['asr']=(time.monotonic()-started)*1000
            report['new_asr_call']=True
        report['transcript']=asdict(transcript);report['status']='transcribed';save()
        print('MoulSot:',transcript.text,flush=True)
        started=time.monotonic()
        response=await router.route(normalize(transcript.text,config),state.router_context())
        report['timings_ms']['router']=(time.monotonic()-started)*1000
        action=state.consume(response.model_dump())
        report.update(router=response.model_dump(),action=asdict(action),state=state.values,
                      pending_proposal=state.pending_proposal,status='routed');save()
        print('Reply action:',action.kind,flush=True)
        started=time.monotonic()
        rendered=await voice.render(action,state.values)
        report['timings_ms']['tts_including_cache']=(time.monotonic()-started)*1000
        with wave.open(io.BytesIO(rendered.wav),'rb') as audio:
            reply_pcm=audio.readframes(audio.getnframes())
            if audio.getframerate()!=16000 or audio.getnchannels()!=1 or audio.getsampwidth()!=2 or len(reply_pcm)!=audio.getnframes()*2:
                raise ValueError('Reply WAV format or length mismatch.')
        reply=prefix.with_name(prefix.name+'_reply.wav')
        combined=prefix.with_name(prefix.name+'_conversation.wav')
        reply.write_bytes(rendered.wav)
        combined.write_bytes(wav_bytes(pcm+bytes(16000)+reply_pcm,config['runtime']))
        report.update(status='reply_generated',spoken_text=rendered.text,reply_path=str(reply),
                      conversation_path=str(combined),reply_seconds=len(reply_pcm)/32000,
                      voice=config['demo']['tts_model'],synthetic_reply=True)
        print(json.dumps({key:report[key] for key in ('status','spoken_text','timings_ms','reply_path','conversation_path')},ensure_ascii=False),flush=True)
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        print(json.dumps({'status':'failed','error_type':type(exc).__name__,'error':str(exc)},ensure_ascii=False),flush=True)
    finally:
        report['stt_calls']=stt.calls;report['router_calls']=router.calls
        report['tts_calls']=voice.calls
        await stt.close();await router.close();await voice.close()
        save()
        print('Report:',prefix.with_suffix('.json'),flush=True)


if __name__=='__main__': asyncio.run(main())
