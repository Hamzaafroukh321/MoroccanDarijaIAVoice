import json
from pathlib import Path
import pytest
from engine.state import TaskState


@pytest.fixture
def config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/pizza.json').read_text(encoding='utf-8'))


def op(kind,slot,value=None): return {'op':kind,'slot':slot,'value':value}


def reply(ops=None,yes=False,no=False):
    return {'ops':ops or [],'confidence':1.0,'unclear':False,'is_affirmation':yes,'is_negation':no}


def fill(state):
    state.apply([op('set','size','large'),op('set','quantity',1),op('set','phone','+212612345678'),op('set','address','Example address')])


def test_all_operations(config):
    state=TaskState(config)
    state.apply([op('set','size','large'),op('set','size','small'),op('add','toppings','cheese'),op('add','toppings',['cheese','olive'])])
    assert state.values=={'size':'small','toppings':['cheese','olive']}
    state.apply([op('remove','toppings','olive'),op('remove','size','large')])
    assert state.values['size']=='small' and state.values['toppings']==['cheese']
    state.apply([op('clear','size'),op('remove','toppings')])
    assert state.values=={}


def test_atomic_invalid_response(config):
    state=TaskState(config)
    with pytest.raises(ValueError): state.apply([op('set','quantity',2),op('set','size','giant')])
    assert state.values=={}
    with pytest.raises(ValueError): state.apply([op('add','quantity',2)])
    with pytest.raises(ValueError): state.apply([op('set','quantity',True)])


def test_confirmation_is_mandatory_and_versioned(config):
    state=TaskState(config)
    assert state.next_action().slot=='size'
    fill(state)
    assert state.ready and state.next_action().kind=='readback'
    state.consume(reply(yes=True))
    assert not state.confirmed
    state.begin_confirmation(state.version)
    state.consume(reply([op('set','quantity',2)],yes=True))
    assert not state.confirmed and state.next_action().kind=='readback'
    state.begin_confirmation(state.version)
    assert state.consume(reply(yes=True)).kind=='accepted'


def test_negation_waits_for_correction(config):
    state=TaskState(config); fill(state); state.begin_confirmation(state.version)
    assert state.consume(reply(no=True)).kind=='listen'
    assert state.consume(reply([op('set','size','small')])).kind=='readback'


def test_barge_in_invalidates_readback(config):
    state=TaskState(config); fill(state); state.begin_confirmation(state.version)
    state.invalidate_confirmation(); state.consume(reply(yes=True))
    assert not state.confirmed


def test_pipeline_confirm_and_barge_in(config,tmp_path):
    import asyncio
    from engine.pipeline import VoiceSession, TurnState
    from engine.responder import RenderedAudio
    from engine.stt import Transcript, wav_bytes
    class ASR:
        calls=[]
        def __init__(self): self.count=0
        async def transcribe(self,pcm):
            self.count+=1
            return Transcript('complete order' if self.count==1 else 'yes',0.9,'fixture')
    class Route:
        async def route(self,text,state):
            if not state: return reply([op('set','size','large'),op('set','quantity',1),op('set','phone','0612345678'),op('set','address','Example address')])
            return reply(yes=True)
    class Bank:
        async def render(self,action,values): return RenderedAudio(wav_bytes(bytes(1024),config['runtime']),512,0,['fixture.wav'])
    async def run():
        events=[]
        async def emit(event):
            events.append(event)
            if event['type']=='audio_end': await voice.playback_finished(event['playback_id'])
        async def emit_audio(data): pass
        voice=VoiceSession(config,tmp_path,ASR(),Route(),Bank(),lambda pcm: 1.0 if pcm[0] else 0.0,emit,emit_audio)
        await voice.start(); await voice.playback_task
        async def speak(value):
            # Different PCM yields distinct transcript cache keys.
            for _ in range(10): await voice.feed(bytes([value,0])*512); await asyncio.sleep(0)
            for _ in range(90): await voice.feed(bytes(1024)); await asyncio.sleep(0)
            await voice.queue.join()
            if voice.playback_task: await voice.playback_task
        await speak(1)
        assert voice.task.ready and voice.mode==TurnState.CONFIRMING
        assert not voice.task.confirmed
        await speak(2)
        assert voice.mode==TurnState.DONE and voice.task.confirmed
        await voice.close()
        saved=json.loads(next(tmp_path.glob('bench/results/session_*.json')).read_text())
        assert saved['status']=='completed' and saved['synthetic_output_fraction']==0
    asyncio.run(run())


def test_pipeline_barge_in_preserves_first_frames(config,tmp_path):
    import asyncio
    from engine.pipeline import VoiceSession, TurnState
    from engine.responder import RenderedAudio
    from engine.stt import wav_bytes
    class Bank:
        async def render(self,*args): return RenderedAudio(wav_bytes(bytes(1024),config['runtime']),512,0,['fixture.wav'])
    async def run():
        events=[]
        async def emit(event): events.append(event)
        async def send(data): pass
        voice=VoiceSession(config,tmp_path,None,None,Bank(),lambda frame:1.0,emit,send)
        await voice.start(); await asyncio.sleep(0)
        for _ in range(2): await voice.feed(bytes(1024))
        assert voice.playback_id is not None
        await voice.feed(bytes(1024))
        assert voice.mode==TurnState.LISTENING and voice.detector.speech_ms==96
        assert any(event['type']=='audio_stop' for event in events)
        await voice.close()
    asyncio.run(run())


def test_audio_bank_readback_and_review_gate(config,tmp_path):
    import asyncio
    from engine.responder import AudioBank, AudioBankError, bank_manifest
    from engine.state import Action
    from engine.stt import wav_bytes
    bank=AudioBank(config,tmp_path)
    with pytest.raises(AudioBankError): bank.preflight()
    plan=bank.plan(Action('readback'),{'size':'large','quantity':2,'phone':'0612345678','address':'Example'})
    assert [item for item in plan if isinstance(item,str) and item.startswith('digit_')]==[f'digit_{digit}.wav' for digit in '0612345678']
    assert any(isinstance(item,dict) and item['spell'] for item in plan)
    def reviewed(value):
        if isinstance(value,dict): return {('fixture' if '[[' in key else key):(False if key=='needs_human_review' else reviewed(item)) for key,item in value.items()}
        if isinstance(value,list): return [reviewed(item) for item in value]
        return 'fixture phrase' if isinstance(value,str) and '[[' in value else value
    approved=reviewed(config);bank=AudioBank(approved,tmp_path)
    bank.directory.mkdir(parents=True)
    for name in bank_manifest(approved): bank.path(name).write_bytes(wav_bytes(bytes(320),approved['runtime']))
    async def run():
        rendered=await bank.render(Action('readback'),{'quantity':2})
        # Prefix, carrier, quantity, question; three 120 ms gaps.
        assert rendered.total_samples==4*160+3*1920
        assert rendered.synthetic_samples==0
    asyncio.run(run())
    with pytest.raises(AudioBankError): bank.path('../outside.wav')


def test_clinic_generality_without_engine_changes():
    import hashlib
    from engine.config import load_config
    from engine.responder import AudioBank
    from engine.state import Action
    root=Path(__file__).resolve().parents[1]
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'engine').glob('*.py')}
    clinic=load_config(root/'configs/clinic.json')
    # Test-only enum choice: owner clinic choices are deliberately still pending.
    clinic['slots'][0]['values']=['doctor_fixture']
    state=TaskState(clinic)
    state.apply([op('set','doctor','doctor_fixture'),op('set','date','15/09/2026'),op('set','time','9h30'),op('set','patient_name','Fixture Patient'),op('set','phone','+212712345678')])
    assert state.values['date']=='2026-09-15' and state.values['time']=='09:30'
    assert state.values['phone']=='0712345678' and state.ready
    plan=AudioBank(clinic,root).plan(Action('readback'),state.values)
    assert 'date_separator.wav' in plan and 'time_separator.wav' in plan
    assert any(isinstance(item,dict) and item['slot']=='patient_name' for item in plan)
    state.begin_confirmation(state.version);state.consume(reply(yes=True));assert state.confirmed
    after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'engine').glob('*.py')}
    assert before==after


def test_handoff_after_exceeding_three_clarifications(config,tmp_path):
    import asyncio
    from engine.pipeline import VoiceSession
    from engine.responder import RenderedAudio
    from engine.stt import Transcript,wav_bytes
    class ASR:
        async def transcribe(self,pcm):return Transcript('',0.0,'fixture')
    class Route:
        async def route(self,*args):raise AssertionError('Unusable audio must bypass the router.')
    class Bank:
        async def render(self,action,values):return RenderedAudio(wav_bytes(bytes(1024),config['runtime']),512,0,[action.kind])
    async def run():
        async def emit(event):
            if event['type']=='audio_end':await voice.playback_finished(event['playback_id'])
        async def send(data):pass
        voice=VoiceSession(config,tmp_path,ASR(),Route(),Bank(),lambda pcm:float(bool(pcm[0])),emit,send)
        await voice.start();await voice.playback_task
        for count in range(1,5):
            for _ in range(10):await voice.feed(bytes([1,0])*512);await asyncio.sleep(0)
            for _ in range(90):await voice.feed(bytes(1024));await asyncio.sleep(0)
            await voice.queue.join()
            if voice.playback_task:await voice.playback_task
            assert voice.clarify_count==count
        assert voice.status=='handoff' and not voice.task.confirmed
        assert [item['action'] for item in voice.outputs]==['greeting','clarify_overlap','clarify_overlap','clarify_overlap','handoff']
        await voice.close()
    asyncio.run(run())


def test_demo_isolation_and_speech_cache(config,tmp_path,monkeypatch):
    import asyncio
    from copy import deepcopy
    import httpx
    from engine.demo import demo_config, DemoVoice, pcm16
    from engine.normalize import normalize
    from engine.responder import AudioBank, AudioBankError
    from engine.state import Action
    from engine.stt import wav_bytes
    monkeypatch.setenv('DEMO_TTS_PROVIDER','groq')
    original=deepcopy(config)
    demo=demo_config(config)
    assert config==original
    assert {s['id'] for s in demo['slots']}=={'quantity','size','toppings','drink'}
    assert normalize('بجوج ديال pizza',demo).startswith('2 ')
    with pytest.raises(AudioBankError): AudioBank(config,tmp_path).preflight()
    monkeypatch.setenv('GROQ_API_KEY','fixture')
    requests=[]
    def handler(request):
        payload=json.loads(request.content)
        assert payload['model']==demo['demo']['tts_model'] and len(payload['input'])<=200
        requests.append(payload)
        return httpx.Response(200,content=wav_bytes(bytes(1024),demo['runtime']))
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            voice=DemoVoice(demo,tmp_path,client=client)
            for _ in range(2):
                audio=await voice.render(Action('greeting'),{})
                assert audio.total_samples==audio.synthetic_samples==512
                assert audio.text and pcm16(audio.wav,demo['runtime'])==bytes(1024)
        assert len(requests)==1
        assert [(call['cache_hits'],call['cache_misses']) for call in voice.calls]==[(0,1),(1,0)]
        assert all(call['ok'] and call['elapsed_ms']>=0 for call in voice.calls)
        with pytest.raises(AudioBankError): pcm16(b'not a wav',demo['runtime'])
        finite=wav_bytes(bytes(1024),demo['runtime'])
        streamed=bytearray(finite); streamed[4:8]=b'\xff'*4; streamed[40:44]=b'\xff'*4
        assert pcm16(bytes(streamed),demo['runtime'])==bytes(1024)
        with pytest.raises(AudioBankError): pcm16(finite[:-2],demo['runtime'])
    asyncio.run(run())


@pytest.mark.parametrize('schema_version',[1,2])
def test_demo_correction_readback_and_confirmation(config,tmp_path,schema_version):
    import asyncio
    from engine.demo import demo_config, reply_text
    from engine.pipeline import VoiceSession
    from engine.responder import RenderedAudio
    from engine.stt import Transcript,wav_bytes
    demo=demo_config(config)
    demo['demo']['order_schema_version']=schema_version
    class ASR:
        count=0
        async def transcribe(self,pcm):
            self.count+=1
            return Transcript(str(self.count),.9,'fixture')
    class Route:
        async def route(self,text,state):
            if schema_version==2:
                def item_op(kind,item_id,slot=None,value=None):
                    return {'op':kind,'item_id':item_id,'slot':slot,'value':value}
                if text=='1':
                    return reply([item_op('create',1),item_op('set',1,'quantity',1),item_op('set',1,'size','large'),item_op('set',1,'toppings',['cheese']),
                        item_op('create',2),item_op('set',2,'quantity',1),item_op('set',2,'size','medium'),item_op('set',2,'toppings',['beef']),item_op('set',None,'drink',['water'])])
                assert state['state']['items'][0]['size']=='large'
                assert state['last_assistant_action']['kind']=='readback' and state['readback_complete']
                if text=='2': return reply([item_op('set',2,'size','small')],yes=True)
                return reply(yes=True)
            if text=='1': return reply([op('set','quantity',1),op('set','size','large'),op('add','toppings','cheese'),op('add','drink','water')])
            if text=='2': return reply([op('set','quantity',2),op('set','size','small')],yes=True)
            return reply(yes=True)
    class Bank:
        async def render(self,action,values):
            return RenderedAudio(wav_bytes(bytes(1024),demo['runtime']),512,512,['synthetic fixture'],reply_text(action,values,demo['demo']))
    async def run():
        events=[]
        async def emit(event):
            events.append(event)
            if event['type']=='audio_end': await voice.playback_finished(event['playback_id'])
        async def send(data): pass
        voice=VoiceSession(demo,tmp_path,ASR(),Route(),Bank(),lambda pcm:float(bool(pcm[0])),emit,send)
        await voice.start(); await voice.playback_task
        async def speak(value):
            for _ in range(10): await voice.feed(bytes([value,0])*512); await asyncio.sleep(0)
            for _ in range(50): await voice.feed(bytes(1024)); await asyncio.sleep(0)
            await voice.queue.join()
            await voice.playback_task
        await speak(1)
        assert voice.task.ready and not voice.task.confirmed
        await speak(2)
        if schema_version==2:
            assert voice.task.values['items']==[{'id':1,'quantity':1,'size':'large','toppings':['cheese']},{'id':2,'quantity':1,'size':'small','toppings':['beef']}]
            assert voice.turns[0]['state']['items'][1]['size']=='medium'
        else:
            assert voice.task.values['quantity']==2 and voice.task.values['size']=='small'
        assert not voice.task.confirmed
        await speak(3)
        assert voice.status=='demo_completed' and voice.task.confirmed
        await voice.close()
        saved=json.loads(next(tmp_path.glob('bench/results/demo_session_*.json')).read_text(encoding='utf-8'))
        assert saved['evaluation_eligible'] is False and saved['synthetic_output_fraction']==1
        assert not list(tmp_path.glob('bench/results/session_*.json'))
        assert len([e for e in events if e['type']=='assistant_text'])==4
    asyncio.run(run())


def test_demo_provider_failure_stops_without_retry_loop(config,tmp_path):
    import asyncio
    from engine.demo import demo_config
    from engine.endpointing import Segment
    from engine.pipeline import VoiceSession
    from engine.stt import STTError
    class ASR:
        async def transcribe(self,pcm): raise STTError('Provider quota exhausted')
    async def run():
        events=[]
        async def emit(event): events.append(event)
        voice=VoiceSession(demo_config(config),tmp_path,ASR(),None,None,None,emit,None)
        voice.worker=asyncio.create_task(voice._worker())
        await voice.queue.put(Segment(bytes(1024),0,320,1728,320))
        await asyncio.wait_for(voice.done.wait(),1)
        assert voice.status=='error' and not voice.task.confirmed
        assert any(e['type']=='error' and 'quota' in e['message'] for e in events)
        await voice.close()
    asyncio.run(run())


def test_elevenlabs_demo_audio_and_cache(config,tmp_path,monkeypatch):
    import asyncio
    import httpx
    from engine.demo import demo_config,DemoVoice,pcm16
    from engine.state import Action
    monkeypatch.setenv('DEMO_TTS_PROVIDER','elevenlabs')
    monkeypatch.setenv('ELEVENLABS_API_KEY','eleven_fixture')
    monkeypatch.setenv('GROQ_API_KEY','groq_not_for_speech')
    demo=demo_config(config);seen=[]
    def handler(request):
        seen.append(request)
        assert request.url.host=='api.elevenlabs.io'
        assert request.url.params['output_format']=='pcm_16000'
        assert request.headers['xi-api-key']=='eleven_fixture'
        assert 'authorization' not in request.headers
        assert json.loads(request.content)['model_id']=='eleven_multilingual_v2'
        return httpx.Response(200,content=bytes(1024))
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            bank=DemoVoice(demo,tmp_path,client=client)
            for _ in range(2):
                rendered=await bank.render(Action('greeting'),{})
                assert pcm16(rendered.wav,demo['runtime'])==bytes(1024)
                assert rendered.total_samples==rendered.synthetic_samples==512
        assert len(seen)==1
        assert not (tmp_path/demo['stt']['rate_limit_file']).exists()
    asyncio.run(run())


def test_elevenlabs_plan_error_does_not_fallback(config,tmp_path,monkeypatch):
    import asyncio
    import httpx
    from engine.demo import demo_config,DemoVoice
    from engine.state import Action
    from engine.responder import AudioBankError
    monkeypatch.setenv('DEMO_TTS_PROVIDER','elevenlabs')
    monkeypatch.setenv('ELEVENLABS_API_KEY','fixture')
    seen=[]
    def handler(request):
        seen.append(request.url.host)
        return httpx.Response(402,json={'detail':{'status':'payment_required','message':'provider internal text'}})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            bank=DemoVoice(demo_config(config),tmp_path,client=client)
            with pytest.raises(AudioBankError,match='paid plan') as error:
                await bank.render(Action('greeting'),{})
            assert 'provider internal text' not in str(error.value)
        assert seen==['api.elevenlabs.io']
    asyncio.run(run())


def test_azure_mouna_ssml_audio_and_cache(config,tmp_path,monkeypatch):
    import asyncio
    import httpx
    import xml.etree.ElementTree as ET
    from engine.demo import demo_config,DemoVoice,pcm16
    from engine.state import Action
    from engine.stt import wav_bytes
    monkeypatch.setenv('DEMO_TTS_PROVIDER','azure')
    monkeypatch.setenv('AZURE_SPEECH_KEY','azure_fixture')
    monkeypatch.setenv('AZURE_SPEECH_REGION','westeurope')
    monkeypatch.setenv('GROQ_API_KEY','groq_not_for_speech')
    monkeypatch.setenv('ELEVENLABS_API_KEY','eleven_not_for_speech')
    demo=demo_config(config);seen=[]
    text='سلام! <voice> & شنو بغيتي؟'
    demo['demo']['responses']['greeting']=text
    def handler(request):
        seen.append(request)
        assert str(request.url)=='https://westeurope.tts.speech.microsoft.com/cognitiveservices/v1'
        assert request.headers['Ocp-Apim-Subscription-Key']=='azure_fixture'
        assert 'authorization' not in request.headers and 'xi-api-key' not in request.headers
        assert request.headers['Content-Type']=='application/ssml+xml'
        assert request.headers['X-Microsoft-OutputFormat']=='riff-16khz-16bit-mono-pcm'
        root=ET.fromstring(request.content)
        voice=root.find('{http://www.w3.org/2001/10/synthesis}voice')
        assert voice.attrib['name']=='ar-MA-MounaNeural'
        assert voice.text==text and len(voice)==0
        return httpx.Response(200,content=wav_bytes(bytes(1024),demo['runtime']))
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            bank=DemoVoice(demo,tmp_path,client=client)
            for _ in range(2):
                rendered=await bank.render(Action('greeting'),{})
                assert pcm16(rendered.wav,demo['runtime'])==bytes(1024)
                assert rendered.total_samples==rendered.synthetic_samples==512
        assert len(seen)==1
        assert not (tmp_path/demo['stt']['rate_limit_file']).exists()
    asyncio.run(run())


@pytest.mark.parametrize('status',[401,429])
def test_azure_failure_no_fallback_or_cache(config,tmp_path,monkeypatch,status):
    import asyncio
    import httpx
    from engine.demo import demo_config,DemoVoice
    from engine.state import Action
    from engine.responder import AudioBankError
    monkeypatch.setenv('DEMO_TTS_PROVIDER','azure')
    monkeypatch.setenv('AZURE_SPEECH_KEY','azure_fixture')
    monkeypatch.setenv('AZURE_SPEECH_REGION','westeurope')
    seen=[]
    def handler(request):
        seen.append(request.url.host)
        return httpx.Response(status,text='private provider detail')
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            bank=DemoVoice(demo_config(config),tmp_path,client=client)
            with pytest.raises(AudioBankError,match='quota' if status==429 else 'matching region') as error:
                await bank.render(Action('greeting'),{})
            assert 'private provider detail' not in str(error.value)
        assert seen==['westeurope.tts.speech.microsoft.com']
        assert not list(tmp_path.rglob('*.wav'))
    asyncio.run(run())


@pytest.mark.parametrize('region',['','https://example.com','west europe','eastus@example.com'])
def test_azure_rejects_invalid_region(monkeypatch,region):
    from engine.demo import azure_speech_url
    from engine.responder import AudioBankError
    monkeypatch.setenv('AZURE_SPEECH_REGION',region)
    with pytest.raises(AudioBankError,match='AZURE_SPEECH_REGION'):
        azure_speech_url()
