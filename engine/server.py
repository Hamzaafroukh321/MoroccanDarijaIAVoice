"""FastAPI application and session lifecycle; audio belongs in the pipeline."""

import asyncio
import json
import logging
import os
import re
from urllib.parse import urlsplit
from contextlib import AsyncExitStack, asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from engine.config import ROOT, load_config
from engine.pipeline import AudioCapture, CaptureError, VoiceSession
from engine.endpointing import SileroVAD
from engine.responder import AudioBank, AudioBankError, bank_manifest, review_issues
from engine.router import Router, check_models
from engine.stt import SpeechToText
from engine import lexicon
from engine.demo import DemoVoice, demo_config, demo_supported, azure_speech_url, DEMO_TTS_PROVIDERS
from engine.demo_validation import DemoConfigError
from engine.recovery import RecoveryStore, RecoveryError, RECOVERY_ERROR
from engine.moulsot_context import configured_context, validate_context_target


load_dotenv(ROOT / ".env")
config = load_config(ROOT / "configs/pizza.json")
runtime = config["runtime"]
logger = logging.getLogger(__name__)
recovery_store = RecoveryStore()
@asynccontextmanager
async def lifespan(app):
    key = os.getenv('GROQ_API_KEY', '')
    if key:
        models = {domain_config(domain['id'])['router']['model'] for domain in available_domains()}
        await check_models(models, key)
        logger.info('Groq startup check passed: %s', ', '.join(sorted(models)))
    else:
        logger.warning('GROQ_API_KEY is missing; only microphone capture is available.')
    yield


app = FastAPI(title="Darija Voice", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


def domain_config(domain):
    if not re.fullmatch(r'[a-z][a-z0-9_-]*',domain): raise ValueError('Invalid domain id.')
    selected=load_config(ROOT/'configs'/f'{domain}.json')
    if selected['domain_id'] != domain:
        raise ValueError('Domain filename must match its configured ID.')
    return selected


def available_domains():
    """Discover valid local task configs without allocating speech providers."""
    domains=[]
    for path in (ROOT/'configs').glob('*.json'):
        if path.stem == 'schema' or not re.fullmatch(r'[a-z][a-z0-9_-]*', path.stem):
            continue
        try:
            selected=load_config(path)
        except (ValueError, OSError):
            continue
        if selected['domain_id'] != path.stem:
            continue
        title=selected.get('display_name', path.stem)
        if not isinstance(title, str) or not title.strip():
            continue
        domains.append({'id':path.stem, 'title':title})
    defaults={'pizza':0, 'clinic':1}
    return sorted(domains, key=lambda item:(defaults.get(item['id'],2), item['id']))


def voice_issues(selected):
    issues=[]
    if configured_context(selected['stt']):
        issues.append('Disable experimental MoulSot vocabulary for reviewed research sessions; it is supported only in the local demo.')
    if lexicon.NEEDS_HUMAN_REVIEW: issues.append('Review the supplied Darija markers in engine/lexicon.py.')
    if not os.getenv('GROQ_API_KEY'): issues.append('Set GROQ_API_KEY for the slot router.')
    if os.getenv('STT_PRIMARY','moulsot')=='moulsot' and not os.getenv('MOULSOT_ENDPOINT'):
        issues.append('Set MOULSOT_ENDPOINT to hosted MoulSot or the isolated local JSON service.')
    try: AudioBank(selected,ROOT).preflight()
    except (AudioBankError,OSError) as exc: issues.append(str(exc))
    return issues


def demo_issues(selected):
    issues=[]
    if not demo_supported(selected): issues.append('No synthetic preview is registered for this domain.')
    if not os.getenv('GROQ_API_KEY'): issues.append('Set GROQ_API_KEY for the slot router.')
    if not os.getenv('MOULSOT_ENDPOINT'): issues.append('Set MOULSOT_ENDPOINT for speech recognition.')
    protocol = os.getenv('MOULSOT_PROTOCOL', '').strip() or selected['stt']['moulsot_protocol']
    if protocol not in {'json', 'gradio'}:
        issues.append('Set MOULSOT_PROTOCOL to json or gradio.')
    try:
        validate_context_target(configured_context(selected['stt']), demo=True,
            primary='moulsot', fallback=False, protocol=protocol, endpoint=os.getenv('MOULSOT_ENDPOINT', ''))
    except ValueError as exc:
        issues.append(str(exc))
    provider=os.getenv('DEMO_TTS_PROVIDER','groq').strip().lower()
    if provider not in DEMO_TTS_PROVIDERS:
        issues.append('Set DEMO_TTS_PROVIDER to darija_xtts, azure, elevenlabs or groq.')
    if provider=='elevenlabs' and not os.getenv('ELEVENLABS_API_KEY'):
        issues.append('Set ELEVENLABS_API_KEY for the Moroccan demo voice.')
    if provider=='azure':
        if not os.getenv('AZURE_SPEECH_KEY'): issues.append('Set AZURE_SPEECH_KEY for Mouna’s voice.')
        try: azure_speech_url()
        except AudioBankError as exc: issues.append(str(exc))
    return issues


def demo_asr_label(selected):
    protocol = os.getenv('MOULSOT_PROTOCOL', '').strip() or selected['stt']['moulsot_protocol']
    try:
        host = urlsplit(os.getenv('MOULSOT_ENDPOINT', '')).hostname
    except ValueError:
        host = None
    label = 'MoulSot on this computer' if protocol == 'json' and host in {'127.0.0.1', 'localhost', '::1'} else 'Hosted MoulSot'
    return label + (' · experimental vocabulary' if configured_context(selected['stt']) else '')


@app.get('/')
async def index(domain: str='pizza', bank: str='', overwrite: bool=False, mode: str='capture') -> HTMLResponse:
    try: selected=domain_config(domain)
    except (ValueError,OSError): return HTMLResponse('Unknown or invalid domain.',status_code=400)
    page=(ROOT/'web/index.html').read_text(encoding='utf-8')
    public={key:value for key,value in selected['runtime'].items() if key not in {'host','port','recordings_dir','results_dir'}}
    public.update({'domain_id':domain,'domain_title':selected['display_name'],'domains':available_domains(),'voice_issues':voice_issues(selected),'bank_target':bank,'bank_overwrite':overwrite,'bank_text':bank_manifest(selected).get(bank,''),'max_session_ms':selected['engine']['max_session_ms']})
    public.update(demo_issues=demo_issues(selected), demo_supported=demo_supported(selected),
                  demo_asr_label=demo_asr_label(selected),
                  demo_ui={}, demo_labels={}, demo_values={}, demo_state_kind=None, demo_collection=None,
                  initial_mode=mode if mode in {'capture','demo','voice'} else 'capture')
    if demo_supported(selected):
        try:
            preview=demo_config(selected)
            public['demo_max_session_ms']=preview['engine']['max_session_ms']
            public['demo_tts_provider']=preview['demo']['tts_provider']
            public['demo_tts_voice_label']=preview['demo'].get('tts_voice_label',preview['demo']['tts_voice'])
            public['demo_ui']=preview['demo']['ui']
            public['demo_labels']=preview['demo']['labels']
            public['demo_values']=preview['demo']['values']
            public['demo_state_kind']=preview['demo']['state_kind']
            if preview['demo']['state_kind'] == 'configured_collection_scoped':
                settings = preview['demo']
                schema = settings['transaction_schema']
                public['demo_collection'] = dict(name=schema['collection'], label=settings['collection_label'],
                    root_slots=schema['root_slots'], item_slots=schema['item_slots'],
                    min_items=settings['collection_min_items'], max_items=settings['collection_max_items'])
        except DemoConfigError as exc:
            public['demo_issues'].append(str(exc))
        except (ValueError,OSError,KeyError,TypeError) as exc:
            logger.warning('Demo preview configuration failed: %s',type(exc).__name__)
            if not any('DEMO_TTS_PROVIDER' in issue for issue in public['demo_issues']):
                public['demo_issues'].append('The demo configuration is invalid. Check configs/demo and DEMO_TTS_PROVIDER.')
    payload=json.dumps(public).replace('<','\\u003c')
    return HTMLResponse(page.replace('{{CAPTURE_CONFIG}}',payload))


async def finish_voice_cleanup(resources):
    """Finish all registered closers even if the connection task is cancelled."""
    cleanup=asyncio.create_task(resources.aclose())
    cancelled=False
    while True:
        try:
            await asyncio.shield(cleanup)
            break
        except asyncio.CancelledError:
            if cleanup.cancelled():
                raise
            cancelled=True
        except Exception as exc:
            # AsyncExitStack already attempted every closer. Never expose a
            # cleanup exception body through the client or skip another owner.
            logger.error('Voice resource cleanup failed: %s',type(exc).__name__)
            break
    if cancelled:
        raise asyncio.CancelledError()


async def voice_connection(websocket, selected, demo=False):
    issues=demo_issues(selected) if demo else voice_issues(selected)
    if issues:
        await websocket.send_json({'type':'error','message':' '.join(issues)})
        await websocket.close(code=1008)
        return
    resources=AsyncExitStack()
    voice=None
    recovery_offer=None
    prepared_resume_token=None
    resume_consumed=False

    async def emit(event):
        nonlocal recovery_offer
        if (event.get('type') == 'error' and prepared_resume_token is not None and
                not resume_consumed):
            try:
                recovery_store.prepare(prepared_resume_token, selected)
            except RecoveryError:
                pass
            else:
                # Authoritative status only; keep the existing client offer.
                # Do not spend/reissue the token or echo it into another field.
                event = {**event, 'resume_available': True}
        if demo and voice is not None and event.get('type') == 'error':
            # Freeze the old conversation before advertising its committed copy.
            # Do not cancel the task currently delivering the error to the peer.
            voice.recovery_frozen = True
            current_task = asyncio.current_task()
            for task in (voice.worker, voice.playback_task, voice.partial_task):
                if task is not None and task is not current_task and not task.done():
                    task.cancel()
            if recovery_offer is None:
                recovery_offer = recovery_store.offer(selected, voice.task, voice.session_id,
                                                       voice.pending_request)
            if recovery_offer is not None:
                event = {**event, 'recovery': recovery_offer}
        await websocket.send_json(event)

    try:
        if demo: selected=demo_config(selected)
        await websocket.send_json({'type':'ready'})
        first=await asyncio.wait_for(websocket.receive_json(),selected['runtime']['idle_timeout_ms']/1000)
        if not isinstance(first,dict) or first.get('type')!='start': raise CaptureError('Start the voice session first.')
        recovered = None
        if 'resume_token' in first:
            if not demo: raise RecoveryError(RECOVERY_ERROR)
            recovered = recovery_store.prepare(first['resume_token'], selected)
            prepared_resume_token = first['resume_token']
        stt=SpeechToText(selected,ROOT,primary='moulsot',fallback=False) if demo else SpeechToText(selected,ROOT)
        resources.push_async_callback(stt.close)
        router=Router(selected,ROOT)
        resources.push_async_callback(router.close)
        bank=DemoVoice(selected,ROOT) if demo else AudioBank(selected,ROOT)
        if demo: resources.push_async_callback(bank.close)
        vad=await asyncio.to_thread(SileroVAD,selected)
        voice=VoiceSession(selected,ROOT,stt,router,bank,vad,emit,websocket.send_bytes)
        resources.push_async_callback(voice.close)
        if recovered is not None:
            # Allocation failure leaves the original token available. Recheck
            # expiry/one-use status after awaits before activating this task.
            recovered = recovery_store.restore(first['resume_token'], selected)
            resume_consumed = True
            voice.task = recovered.task
            voice.resumed_from = recovered.source_session_id
            voice.discarded_pending = recovered.discarded_pending
        await websocket.send_json({'type':'started','session_id':voice.session_id,
                                   'resumed':recovered is not None,
                                   'discarded_pending':bool(recovered and recovered.discarded_pending)})
        await voice.start()
        while not voice.done.is_set():
            receive=asyncio.create_task(websocket.receive())
            done=asyncio.create_task(voice.done.wait())
            try:
                completed,_=await asyncio.wait({receive,done},timeout=selected['runtime']['idle_timeout_ms']/1000,return_when=asyncio.FIRST_COMPLETED)
                if not completed: raise CaptureError('Voice connection timed out.')
                if done in completed: break
                message=receive.result()
            finally:
                for task in (receive,done):
                    if not task.done(): task.cancel()
                await asyncio.gather(receive,done,return_exceptions=True)
            # The peer has already closed the ASGI connection. Sending another
            # close frame can raise RuntimeError and mislabel a normal disconnect.
            if message['type']=='websocket.disconnect': return
            if message.get('bytes') is not None:
                await voice.feed(message['bytes'])
            else:
                control=json.loads(message.get('text',''))
                if not isinstance(control,dict): raise CaptureError('Expected a control object.')
                if control.get('type')=='playback_finished': await voice.playback_finished(control.get('playback_id'))
                elif control.get('type')=='stop':
                    await voice.close()
                    await websocket.send_json({'type':'done','status':'interrupted','slots':voice.task.values,'session_id':voice.session_id})
                    break
                else: raise CaptureError('Unknown voice-session command.')
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        pass
    except RecoveryError:
        try:
            await emit({'type':'error','message':RECOVERY_ERROR})
            await websocket.close(code=1008)
        except (WebSocketDisconnect,RuntimeError): pass
    except DemoConfigError as exc:
        try:
            await emit({'type':'error','message':str(exc)})
            await websocket.close(code=1008)
        except (WebSocketDisconnect,RuntimeError): pass
    except (ValueError,RuntimeError,OSError,asyncio.TimeoutError) as exc:
        logger.warning('Voice session failed: %s',type(exc).__name__)
        if voice: voice.status='error'
        try:
            await emit({'type':'error','message':str(exc) or 'Voice session timed out.'})
            await websocket.close(code=1011)
        except (WebSocketDisconnect,RuntimeError): pass
    finally:
        await finish_voice_cleanup(resources)


@app.websocket(runtime["websocket_path"])
async def session(websocket: WebSocket) -> None:
    origin = websocket.headers.get("origin")
    expected_origin = f"{'https' if websocket.url.scheme == 'wss' else 'http'}://{websocket.headers.get('host')}"
    if origin != expected_origin:
        await websocket.close(code=1008, reason="Open the recorder from this server.")
        return
    await websocket.accept()
    try: selected=domain_config(websocket.query_params.get('domain','pizza'))
    except ValueError:
        await websocket.send_json({'type':'error','message':'Unknown or invalid domain.'})
        await websocket.close(code=1008)
        return
    mode=websocket.query_params.get('mode','capture')
    if mode in {'voice','demo'}:
        await voice_connection(websocket,selected,demo=mode=='demo')
        return
    if mode not in {'capture','bank'}:
        await websocket.close(code=1008)
        return
    if mode=='bank' and review_issues(selected):
        await websocket.send_json({'type':'error','message':'Review every language field before recording the bank.'})
        await websocket.close(code=1008)
        return
    capture = AudioCapture(selected, ROOT)
    try:
        await websocket.send_json({"type": "ready"})
        while True:
            message = await asyncio.wait_for(websocket.receive(), timeout=runtime["idle_timeout_ms"] / 1000)
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                capture.receive(message["bytes"])
                continue
            try:
                control = json.loads(message.get("text", ""))
            except json.JSONDecodeError as exc:
                raise CaptureError("Expected a JSON control message.") from exc
            if not isinstance(control, dict):
                raise CaptureError("Expected a JSON object.")
            if control.get("type") == "start":
                if 'resume_token' in control:
                    raise CaptureError(RECOVERY_ERROR)
                await websocket.send_json(capture.start())
            elif control.get("type") == "stop":
                if "samples" not in control:
                    raise CaptureError("Stop must include the captured sample count.")
                result = capture.finish(control["samples"])
                if mode=='bank':
                    name=websocket.query_params.get('bank','')
                    destination=AudioBank(selected,ROOT).save_recording(name,ROOT/result['audio'],overwrite=websocket.query_params.get('overwrite')=='true')
                    result['bank_audio']=destination.relative_to(ROOT).as_posix()
                logger.info("Saved capture %s (%s ms)", result["audio"], result["duration_ms"])
                await websocket.send_json(result)
                await websocket.close(code=1000)
                break
            else:
                raise CaptureError("Unknown recording control message.")
    except (CaptureError, AudioBankError, asyncio.TimeoutError) as exc:
        detail = str(exc) or "Recording connection timed out. Please try again."
        try:
            await websocket.send_json({"type": "error", "message": detail})
            await websocket.close(code=1008)
        except (WebSocketDisconnect, RuntimeError):
            pass
    except WebSocketDisconnect:
        pass
    except OSError:
        logger.exception("Could not write microphone capture")
        try:
            await websocket.send_json({"type": "error", "message": "Could not save the recording to disk."})
            await websocket.close(code=1011)
        except (WebSocketDisconnect, RuntimeError):
            pass
    finally:
        if capture.writer is not None:
            try:
                result = capture.finish(status="interrupted")
                logger.info("Saved interrupted capture %s", result["audio"])
            except OSError:
                logger.exception("Could not finalize interrupted capture")


def run_server():
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
    uvicorn.run(
        app, host=runtime["host"], port=runtime["port"],
        ws_max_size=runtime["sample_rate_hz"] * runtime["frame_ms"] // 1000 * runtime["sample_width_bytes"],
        ws_max_queue=runtime["max_buffered_frames"],
        # Deflate can expand a 1024-byte PCM frame beyond the wire-size limit.
        # Keep the exact frame bound; raw local audio needs no compression.
        ws_per_message_deflate=False,
    )


if __name__ == "__main__":
    run_server()
