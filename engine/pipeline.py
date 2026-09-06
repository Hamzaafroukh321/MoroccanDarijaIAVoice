"""Local audio capture and orchestration of domain-independent voice sessions."""

from datetime import datetime, timezone
from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from uuid import uuid4
import wave


class CaptureError(ValueError):
    """Invalid session input that must not be saved as a successful capture."""


class AudioCapture:
    def __init__(self, config: dict[str, Any], root: Path):
        self.runtime = config["runtime"]
        self.domain = config["domain_id"]
        self.session_id = uuid4().hex
        self.root = root
        self.frame_samples = self.runtime["sample_rate_hz"] * self.runtime["frame_ms"] // 1000
        self.frame_bytes = self.frame_samples * self.runtime["sample_width_bytes"] * self.runtime["channels"]
        self.max_samples = self.runtime["sample_rate_hz"] * self.runtime["max_recording_ms"] // 1000
        self.path = root / self.runtime["recordings_dir"] / f"capture_{self.session_id}.wav"
        self.frames = 0
        self.pending = b""
        self.writer: wave.Wave_write | None = None
        self.result: dict[str, Any] | None = None

    def start(self) -> dict[str, Any]:
        if self.writer is not None or self.result is not None:
            raise CaptureError("This recording has already started.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = wave.open(str(self.path), "wb")
        self.writer.setnchannels(self.runtime["channels"])
        self.writer.setsampwidth(self.runtime["sample_width_bytes"])
        self.writer.setframerate(self.runtime["sample_rate_hz"])
        return {"type": "started", "session_id": self.session_id}

    def receive(self, frame: bytes) -> None:
        if self.writer is None or self.result is not None:
            raise CaptureError("Start the recording before sending audio.")
        if len(frame) != self.frame_bytes:
            raise CaptureError(f"Expected {self.frame_bytes} bytes per PCM frame.")
        if self.frames * self.frame_samples >= self.max_samples:
            raise CaptureError("The configured recording duration was exceeded.")
        # Hold the last frame so stop can remove only the worklet's zero padding.
        # This preserves the exact input length without buffering the recording.
        if self.pending:
            self.writer.writeframesraw(self.pending)
        self.pending = frame
        self.frames += 1

    def finish(self, samples: int | None = None, status: str = "complete") -> dict[str, Any]:
        if self.result is not None:
            return self.result
        if self.writer is None:
            raise CaptureError("No recording has started.")
        total = self.frames * self.frame_samples
        if samples is None:
            samples = min(total, self.max_samples)
        if type(samples) is not int or not 0 <= samples <= min(total, self.max_samples):
            raise CaptureError("Invalid sample count in stop message.")
        if self.frames and samples <= total - self.frame_samples:
            raise CaptureError("Stop may only trim padding from the final frame.")
        if status == "complete" and samples == 0:
            raise CaptureError("No microphone audio was received.")
        tail_samples = samples - (total - self.frame_samples) if self.frames else 0
        try:
            if self.pending:
                self.writer.writeframesraw(self.pending[:tail_samples * self.runtime["sample_width_bytes"]])
        finally:
            self.writer.close()
            self.writer = None
        self.pending = b""
        self.result = {
            "type": "saved",
            "purpose": "development_capture",
            "session_id": self.session_id,
            "domain": self.domain,
            "status": status,
            "audio": self.path.relative_to(self.root).as_posix(),
            "sample_rate_hz": self.runtime["sample_rate_hz"],
            "channels": self.runtime["channels"],
            "sample_width_bytes": self.runtime["sample_width_bytes"],
            "frames_received": self.frames,
            "samples": samples,
            "duration_ms": samples * 1000 / self.runtime["sample_rate_hz"],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        result_dir = self.root / self.runtime["results_dir"]
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / f"capture_{self.session_id}.json").write_text(
            json.dumps(self.result, indent=2) + "\n", encoding="utf-8"
        )
        return self.result


# Full voice pipeline. Adapters own their external I/O; this module coordinates.
import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from enum import Enum
import hashlib
import logging
import time

from engine.endpointing import EndpointDetector
from engine.normalize import normalize
from engine.overlap import assess
from engine.responder import AudioBankError
from engine.router import RouterError, RouterOutputError, RouterConfigurationError
from engine.state import Action, TaskState
from engine.stt import STTError

logger = logging.getLogger(__name__)


class TurnState(str, Enum):
    IDLE='IDLE'
    LISTENING='LISTENING'
    PROCESSING='PROCESSING'
    SPEAKING='SPEAKING'
    CLARIFYING='CLARIFYING'
    CONFIRMING='CONFIRMING'
    DONE='DONE'


class VoiceSession:
    def __init__(self, config, root, stt, router, bank, vad, emit, emit_audio):
        self.config=config; self.root=Path(root)
        self.stt=stt; self.router=router; self.bank=bank; self.vad=vad
        self.emit=emit; self.emit_audio=emit_audio
        self.session_id=uuid4().hex
        from engine.task_factory import make_task
        self.task=make_task(config)
        self.detector=EndpointDetector(config)
        self.mode=TurnState.IDLE
        self.audio_time_ms=0
        self.queue=asyncio.Queue(maxsize=config['engine']['max_pending_segments'])
        self.worker=None; self.playback_task=None; self.partial_task=None
        self.playback_id=None; self.playback_done=None
        self.cache=OrderedDict(); self.epoch=0; self.barge=[]
        self.closed=False; self.status='in_progress'
        self._close_task=None
        self.deferred_action=None
        self.clarify_count=0; self.endpoints=[]; self.turns=[]; self.outputs=[]
        self.transcript_requests=[]
        self.progress_id=None
        self.progress_events=[]
        self.started_perf=time.perf_counter()
        self.started_at=time.monotonic()
        self.stt_call_start=len(getattr(stt,'calls',[]))
        self.router_call_start=len(getattr(router,'calls',[]))
        self.local_router_call_start=len(getattr(router,'local_calls',[]))
        self.tts_call_start=len(getattr(bank,'calls',[]))
        self.done=asyncio.Event()
        self.last_assistant_action=None
        self.last_assistant_text=None
        self.pending_request=None
        self.next_request_id=1
        self.resumed_from=None
        self.discarded_pending=False
        self.recovery_frozen=False

    async def set_mode(self, mode):
        self.mode=mode
        proposal=getattr(self.task,'pending_proposal',None)
        await self.emit({'type':'state','state':mode.value,'slots':deepcopy(self.task.values),
                         'clarify_count':self.clarify_count,
                         'pending_clarification':deepcopy(getattr(self.task,'pending_clarification',None)),
                         'pending_proposal':self._proposal_preview(proposal)})

    @staticmethod
    def _proposal_preview(proposal):
        if proposal is None:
            return None
        return {key:deepcopy(proposal[key]) for key in
                ('state','coupled_slots','answered_slots','remaining_slots',
                 'linked_addresses','answered_addresses','remaining_addresses') if key in proposal}

    async def start(self):
        self.worker=asyncio.create_task(self._worker())
        await self._choose(self.task.next_action() if self.resumed_from is not None else Action('greeting'))

    @asynccontextmanager
    async def _progress(self, stage):
        if self.closed:
            raise asyncio.CancelledError()
        progress_id=uuid4().hex
        self.progress_id=progress_id
        event={'type':'progress','progress_id':progress_id,'stage':stage,'active':True}
        try:
            self.progress_events.append({**event,'at_ms':(time.perf_counter()-self.started_perf)*1000})
            await self.emit(event)
            yield
        finally:
            # Another stage may have superseded this one while awaiting I/O.
            # Its display must not be cleared by an obsolete completion.
            if self.progress_id==progress_id:
                self.progress_id=None
                finished={**event,'active':False}
                self.progress_events.append({**finished,'at_ms':(time.perf_counter()-self.started_perf)*1000,
                                             'suppressed_on_close':self.closed})
                if not self.closed:
                    await self.emit(finished)

    async def _transcript(self, pcm, *, purpose='final'):
        key=hashlib.sha256(pcm).hexdigest()
        reused=key in self.cache
        if not reused:
            self.cache[key]=asyncio.create_task(self.stt.transcribe(pcm))
        self.cache.move_to_end(key)
        # Evict only completed entries; pending calls must remain cancellable.
        for candidate in list(self.cache):
            if len(self.cache)<=self.config['engine']['transcript_cache_entries']: break
            if candidate!=key and self.cache[candidate].done():
                self.cache[candidate].exception() if not self.cache[candidate].cancelled() else None
                del self.cache[candidate]
        started=time.perf_counter()
        request={'purpose':purpose,'audio_sha256':key,'pcm_bytes':len(pcm),
                 'audio_seconds':len(pcm)/(self.config['runtime']['sample_rate_hz'] *
                     self.config['runtime']['sample_width_bytes'] * self.config['runtime']['channels']),
                 'reused_transcription':reused,'ok':False,'elapsed_clock':'perf_counter'}
        self.transcript_requests.append(request)
        try:
            result=await asyncio.shield(self.cache[key])
            request['ok']=True
            return result
        except asyncio.CancelledError:
            request['cancelled']=True
            raise
        except Exception as exc:
            request['error_type']=type(exc).__name__
            raise
        finally:
            request['wait_ms']=(time.perf_counter()-started)*1000

    def _begin_partial(self, pcm, speech_end):
        epoch=self.epoch
        async def update():
            try:
                transcript=await self._transcript(pcm,purpose='partial')
                if epoch==self.epoch and self.detector.speech_end_ms==speech_end:
                    self.detector.set_tail(normalize(transcript.text,self.config))
            except (STTError,ValueError):
                pass  # Final transcription reports provider failures visibly.
        self.partial_task=asyncio.create_task(update())
        # Keep all partial tasks for deterministic cancellation on close.
        self._partials.add(self.partial_task)
        self.partial_task.add_done_callback(self._partials.discard)

    async def _feed_listening(self, pcm, probability):
        speech=probability>self.config['endpointing']['vad_threshold']
        holds=any(self.config['endpointing'][key] for key in ('hesitation_hold_ms','continuation_hold_ms','digit_hold_ms'))
        if not speech and self.detector.was_speech and self.detector.speech_ms>=self.config['endpointing']['min_speech_duration_ms'] and holds:
            self._begin_partial(bytes(self.detector.buffer),self.detector.speech_end_ms)
        settings=self.config['endpointing']
        pending=self.partial_task is not None and not self.partial_task.done() and self.detector.silence_ms<min(settings['partial_timeout_ms'],settings['max_wait_ms'])
        segment=self.detector.feed(pcm,probability,tail_pending=pending)
        if segment:
            self.deferred_action=None
            self.epoch+=1
            self.partial_task=None
            self.endpoints.append({'start_ms':segment.start_ms,'speech_end_ms':segment.speech_end_ms,'endpoint_ms':segment.endpoint_ms,'forced':segment.forced})
            logger.info('SEGMENT start=%.0f speech_end=%.0f endpoint=%.0f',segment.start_ms,segment.speech_end_ms,segment.endpoint_ms)
            await self.emit({'type':'endpoint',**self.endpoints[-1]})
            try: self.queue.put_nowait(segment)
            except asyncio.QueueFull: raise RuntimeError('Speech processing queue is full; stop and retry.')
        elif not self.detector.active and self.deferred_action is not None and self.queue.empty():
            action=self.deferred_action; self.deferred_action=None
            await self._choose(action)

    async def feed(self, pcm):
        if self.closed or self.recovery_frozen or self.mode==TurnState.DONE: return
        if len(pcm)!=self.detector.frame_bytes: raise ValueError('Expected one 32 ms PCM frame.')
        self.audio_time_ms+=self.config['runtime']['frame_ms']
        if self.audio_time_ms>self.config['engine']['max_session_ms']:
            raise RuntimeError('Maximum voice session duration reached.')
        probability=self.vad(pcm)
        if self.playback_id is not None:
            if probability>self.config['endpointing']['vad_threshold']:
                self.barge.append((pcm,probability))
            else: self.barge.clear()
            if len(self.barge)>=self.config['engine']['barge_in_frames']:
                frames=list(self.barge)
                await self.interrupt()
                self.detector.clock_ms=self.audio_time_ms-len(frames)*self.config['runtime']['frame_ms']
                for frame,prob in frames: await self._feed_listening(frame,prob)
            else:
                self.detector.clock_ms=self.audio_time_ms
            return
        self.detector.clock_ms=self.audio_time_ms-self.config['runtime']['frame_ms']
        await self._feed_listening(pcm,probability)

    async def interrupt(self):
        playback_id=self.playback_id
        self.playback_id=None
        if self.playback_task:
            self.playback_task.cancel()
            await asyncio.gather(self.playback_task,return_exceptions=True)
            self.playback_task=None
        if playback_id: await self.emit({'type':'audio_stop','playback_id':playback_id})
        self.task.invalidate_confirmation()
        self.barge.clear(); self.epoch+=1; self.detector.reset()
        await self.set_mode(TurnState.LISTENING)

    async def playback_finished(self, playback_id):
        if playback_id==self.playback_id and self.playback_done is not None:
            self.playback_done.set()

    async def _choose(self, action):
        if self.closed or self.recovery_frozen: return
        if action.kind=='listen':
            await self.set_mode(TurnState.LISTENING)
            return
        if self.playback_task:
            self.playback_task.cancel()
            await asyncio.gather(self.playback_task,return_exceptions=True)
        playback_id=uuid4().hex
        self.barge.clear()
        self.playback_id=playback_id
        self.playback_done=asyncio.Event()
        version=self.task.version
        values=deepcopy(self.task.values)
        if (self.config.get('demo', {}).get('state_kind') == 'configured_collection_scoped' and
                action.kind in {'ask', 'repair', 'ambiguous_value', 'unsupported_value', 'unintelligible'} and
                self.task.pending_proposal is not None):
            # A question about the fourth proposed row must say "fourth", not
            # the next committed row number. Only question rendering sees this
            # preview; readback/confirmation always use committed values.
            values=deepcopy(self.task.pending_proposal['state'])
        mode=TurnState.CONFIRMING if action.kind=='readback' else TurnState.CLARIFYING if action.kind=='clarify_overlap' else TurnState.SPEAKING
        await self.set_mode(mode)
        self.playback_task=asyncio.create_task(self._play(action,values,version,playback_id))

    async def _play(self, action, values, version, playback_id):
        try:
            async with self._progress('synthesizing'):
                audio=await self.bank.render(action,values)
            if playback_id!=self.playback_id: return
            output={'action':action.kind,'samples':audio.total_samples,'synthetic_samples':audio.synthetic_samples,'fragments':audio.fragments,'completed':False}
            self.outputs.append(output)
            if getattr(audio, 'text', None):
                output['text'] = audio.text
                self.last_assistant_action={'kind':action.kind,'slot':action.slot,'item_id':action.item_id}
                self.last_assistant_text=audio.text[:400]
                await self.emit({'type':'assistant_text','text':audio.text,'action':action.kind,'playback_id':playback_id})
            await self.emit({'type':'audio_start','playback_id':playback_id,'bytes':len(audio.wav),'action':action.kind})
            for index in range(0,len(audio.wav),self.detector.frame_bytes):
                await self.emit_audio(audio.wav[index:index+self.detector.frame_bytes])
            await self.emit({'type':'audio_end','playback_id':playback_id})
            await asyncio.wait_for(self.playback_done.wait(),timeout=self.config['engine']['playback_ack_timeout_ms']/1000)
            if playback_id!=self.playback_id: return
            output['completed']=True
            self.playback_id=None
            self.barge.clear()
            if action.kind=='readback':
                self.task.begin_confirmation(version)
            proposal=getattr(self.task,'pending_proposal',None)
            await self.emit({'type':'playback_complete','playback_id':playback_id,
                'action':action.kind,'slots':deepcopy(self.task.values),
                'pending_clarification':deepcopy(getattr(self.task,'pending_clarification',None)),
                'pending_proposal':self._proposal_preview(proposal),
                'version':self.task.version,'spoken_version':version,
                'readback_complete':self.task.readback_version==self.task.version})
            if action.kind=='readback':
                await self.set_mode(TurnState.CONFIRMING)
            elif action.kind in {'accepted','handoff'}:
                self.status=('demo_completed' if self.config.get('demo') else 'completed') if action.kind=='accepted' else 'handoff'
                await self.set_mode(TurnState.DONE)
                self.save()
                await self.emit({'type':'done','status':self.status,'slots':deepcopy(self.task.values),'session_id':self.session_id})
                self.done.set()
            else: await self.set_mode(TurnState.LISTENING)
        except Exception as exc:
            self.status='error'; self.playback_id=None
            await self.emit({'type':'error','message':str(exc) or 'Audio playback was not acknowledged.'})
            self.done.set()

    async def _worker(self):
        while not self.recovery_frozen:
            segment=await self.queue.get()
            record=None
            try:
                await self.set_mode(TurnState.PROCESSING)
                async with self._progress('transcribing'):
                    transcript=await self._transcript(segment.pcm)
                unusable=assess(segment.pcm,transcript.text,transcript.confidence,segment.speech_ms,self.config)
                record={'transcript':transcript.text,'provider':transcript.provider,'confidence':transcript.confidence,'confidence_kind':transcript.confidence_kind,'overlap':unusable,'endpoint_ms':segment.endpoint_ms}
                self.turns.append(record)
                logger.info('TRANSCRIPT %s',transcript.text)
                await self.emit({'type':'transcript','text':transcript.text,'provider':transcript.provider})
                if unusable['unusable']:
                    self.clarify_count+=1
                    action=Action('handoff' if self.clarify_count>self.config['overlap']['max_clarifies'] else 'clarify_overlap')
                elif not transcript.text.strip():
                    action = self.task.consume({'ops':[], 'unclear':True, 'is_affirmation':False,
                        'is_negation':False, 'intent':'unclear'}) if self.config.get('demo') else Action('not_understood')
                else:
                    normalized=normalize(transcript.text,self.config)
                    context=self.task.router_context() if hasattr(self.task,'router_context') else deepcopy(self.task.values)
                    if hasattr(self.task,'router_context'):
                        context['last_assistant_action']=self.last_assistant_action
                        context['last_assistant_text']=self.last_assistant_text
                        context['readback_complete']=self.task.readback_version==self.task.version
                        context['pending_request']=deepcopy(self.pending_request)
                        record['router_context']=deepcopy(context)
                    async with self._progress('routing'):
                        response=await self.router.route(normalized,context)
                    payload=response.model_dump() if hasattr(response,'model_dump') else response
                    record['router']=payload
                    record['routing_source']=getattr(response, '_routing_source', 'groq')
                    had_proposal=getattr(self.task,'pending_proposal',None) is not None
                    had_clarification=getattr(self.task,'pending_clarification',None)
                    action=(self.task.consume(payload, retained_request=self.pending_request)
                            if payload.get('discard_request') is not None else self.task.consume(payload))
                    if (payload.get('discard_request') is not None or payload.get('discard_clarification') is not None or
                            (had_proposal and payload.get('resolves_clarification') is not None and
                             getattr(self.task,'pending_clarification',None) is None and
                             getattr(self.task,'pending_proposal',None) is None)):
                        self.pending_request=None
                    elif getattr(self.task,'pending_clarification',None) is not None:
                        if (self.pending_request is None or
                                (had_clarification is None and self.pending_request.get('clarification_resolved') is True)):
                            # Keep one bounded recovery exchange. A newly opened
                            # question owns the current utterance, not text from
                            # an earlier resolved question. Committed facts stay
                            # in task state; this is context supersession, not a
                            # user cancellation or rollback.
                            self.pending_request={'id':f'request_{self.next_request_id}',
                                                  'text':normalized[:1000],'truncated':len(normalized)>1000}
                            self.next_request_id+=1
                    elif self.task.ready:
                        self.pending_request=None
                    elif self.pending_request is not None:
                        # A provider may resolve only the queried field. Keep
                        # the bounded original while other order fields remain.
                        self.pending_request['clarification_resolved']=True
                    record['state']=deepcopy(self.task.values)
                record['action']={'kind':action.kind,'slot':action.slot,'item_id':action.item_id}
                if self.config.get('demo'):
                    record['repair_count']=self.task.repair_count
                    if action.kind=='multiple_items':
                        await self.emit({'type':'warning','message':'This demo cannot yet keep different pizzas separate. Your request was not merged into a single pizza or confirmed. Please start a new session for identical pizzas.'})
                # Never start an obsolete response over speech already in progress.
                if action.kind=='handoff' or (not self.detector.active and self.queue.empty()): await self._choose(action)
                else:
                    self.deferred_action=action
                    await self.set_mode(TurnState.LISTENING)
            except (STTError,RouterError,ValueError) as exc:
                if record is not None:
                    record['processing_error']={'type':type(exc).__name__,'message':str(exc)}
                if (isinstance(exc, RouterConfigurationError) or
                        (self.config.get('demo') and isinstance(exc, (STTError, RouterError))
                         and not isinstance(exc, RouterOutputError))):
                    self.status='error'
                    await self.emit({'type':'error','message':str(exc)})
                    self.done.set()
                    continue
                await self.emit({'type':'warning','message':str(exc)})
                recovery=self.task.consume({'ops':[], 'unclear':True, 'is_affirmation':False,
                    'is_negation':False, 'intent':'unclear'}) if self.config.get('demo') else Action('not_understood')
                if record is not None:
                    record['action']={'kind':recovery.kind,'slot':recovery.slot,'item_id':recovery.item_id}
                if recovery.kind=='handoff' or (not self.detector.active and self.queue.empty()): await self._choose(recovery)
                else:
                    self.deferred_action=recovery
                    await self.set_mode(TurnState.LISTENING)
            except Exception as exc:
                self.status='error'
                logger.exception('Voice processing failed')
                await self.emit({'type':'error','message':f'Voice processing failed ({type(exc).__name__}). See the server log.'})
                self.done.set()
            finally:
                self.queue.task_done()

    def save(self):
        directory=self.root/self.config['runtime']['results_dir']; directory.mkdir(parents=True,exist_ok=True)
        total=sum(item['samples'] for item in self.outputs)
        synthetic=sum(item['synthetic_samples'] for item in self.outputs)
        result={'session_id':self.session_id,'domain':self.config['domain_id'],'status':self.status,'state':deepcopy(self.task.values),'confirmed':self.task.confirmed,'clarify_count':self.clarify_count,'endpoints':self.endpoints,'turns':self.turns,'outputs':self.outputs,'synthetic_output_fraction':synthetic/total if total else 0.0,'synthetic_fraction_basis':'generated samples, including interrupted output','stt_calls':getattr(self.stt,'calls',[])[self.stt_call_start:],'router_calls':getattr(self.router,'calls',[])[self.router_call_start:],'saved_at':datetime.now(timezone.utc).isoformat()}
        result['transcript_requests']=deepcopy(self.transcript_requests)
        result['local_router_calls']=deepcopy(getattr(self.router,'local_calls',[])[self.local_router_call_start:])
        result['progress_events']=deepcopy(self.progress_events)
        if self.resumed_from is not None:
            result['resumed_from']=self.resumed_from
            result['recovery_scope']='committed_values_only'
            result['discarded_pending']=self.discarded_pending
        prefix = 'demo_session' if self.config.get('demo') else 'session'
        if self.config.get('demo'):
            result.update(is_demo=True, evaluation_eligible=False, response_voice=self.config['demo']['tts_model'],
                tts_calls=getattr(self.bank,'calls',[])[self.tts_call_start:],
                order_schema_version=self.config['demo'].get('order_schema_version',1),
                pending_clarification=deepcopy(getattr(self.task,'pending_clarification',None)),
                pending_proposal=deepcopy(getattr(self.task,'pending_proposal',None)),
                response_provider=self.config['demo'].get('tts_provider','groq'),response_voice_id=self.config['demo']['tts_voice'],
                language_review='draft', task_scope=self.config['demo'].get('task_scope','pizza pickup; no address or phone'), config=self.config)
        (directory/f'{prefix}_{self.session_id}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return result

    async def close(self, status='interrupted'):
        if self._close_task is None:
            self.closed=True
            if self.status=='in_progress': self.status=status
            self._close_task=asyncio.create_task(self._close_resources())
            self._close_task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        await asyncio.shield(self._close_task)

    async def _close_resources(self):
        tasks=[task for task in [self.worker,self.playback_task,*self._partials,*self.cache.values()] if task is not None]
        for task in tasks: task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        self.save()

    # Created lazily to keep task ownership per session, never as a class set.
    @property
    def _partials(self):
        if not hasattr(self,'_partial_tasks'): self._partial_tasks=set()
        return self._partial_tasks


async def typed_main():
    """M5 diagnostic: route supplied transcripts without pretending audio ran."""
    import argparse
    from dotenv import load_dotenv
    from engine.config import ROOT, load_config
    from engine.router import Router
    parser=argparse.ArgumentParser(description='Inspect slot changes from typed transcripts; this does not run voice confirmation.')
    parser.add_argument('--domain',choices=['pizza','clinic'],default='pizza')
    parser.add_argument('--text',action='append',required=True)
    args=parser.parse_args();load_dotenv(ROOT/'.env')
    config=load_config(ROOT/'configs'/f'{args.domain}.json')
    state=TaskState(config);router=Router(config,ROOT)
    try:
        for text in args.text:
            response=await router.route(normalize(text,config),state.values)
            action=state.consume(response.model_dump())
            print(json.dumps({'state':state.values,'next_action':action.kind,'slot':action.slot,'confirmed':state.confirmed},ensure_ascii=False))
    finally: await router.close()


if __name__=='__main__': asyncio.run(typed_main())
