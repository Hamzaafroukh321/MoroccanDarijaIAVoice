"""Actual local ASGI/WebSocket transport; fake VAD/providers, no external calls."""
import asyncio
from copy import deepcopy
import json
import socket

import pytest
import uvicorn
import websockets

from engine import server
from engine.config import ROOT, load_config
from engine.responder import RenderedAudio
from engine.stt import wav_bytes


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
@pytest.mark.parametrize('termination', ['stop', 'disconnect'])
def test_loopback_disconnect_or_stop_cancels_asr_and_closes_all_owners(monkeypatch, tmp_path, domain, termination):
    config=load_config(ROOT/'configs'/f'{domain}.json')
    monkeypatch.setattr(server,'ROOT',tmp_path)
    monkeypatch.setattr(server,'domain_config',lambda value: deepcopy(config))
    monkeypatch.setattr(server,'demo_issues',lambda value: [])
    monkeypatch.setattr(server,'SileroVAD',lambda value: lambda pcm: 1.0 if pcm[0] else 0.0)

    async def run():
        entered,cancelled,closed=asyncio.Event(),asyncio.Event(),asyncio.Event()
        closure=[]
        class ASR:
            def __init__(self,*args,**kwargs):
                assert kwargs=={'primary':'moulsot','fallback':False}
            async def transcribe(self,pcm):
                entered.set()
                try: await asyncio.Event().wait()
                finally: cancelled.set()
            async def close(self): closure.append('asr');closed.set()
        class Router:
            def __init__(self,*args): pass
            async def route(self,*args): raise AssertionError('The stopped ASR must not route a turn.')
            async def close(self): closure.append('router')
        class Bank:
            def __init__(self,*args): pass
            async def render(self,action,values):
                assert action.kind=='greeting'
                return RenderedAudio(wav_bytes(bytes(1024),config['runtime']),512,512,['silent fixture'])
            async def close(self): closure.append('bank')
        monkeypatch.setattr(server,'SpeechToText',ASR)
        monkeypatch.setattr(server,'Router',Router)
        monkeypatch.setattr(server,'DemoVoice',Bank)
        listener=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        listener.bind(('127.0.0.1',0));listener.listen(8)
        port=listener.getsockname()[1]
        application=uvicorn.Server(uvicorn.Config(server.app,host='127.0.0.1',port=port,
            lifespan='off',ws='websockets',ws_max_size=1024,ws_per_message_deflate=False,
            log_config=None,log_level='critical'))
        serving=asyncio.create_task(application.serve(sockets=[listener]))
        events=[]
        try:
            async with asyncio.timeout(7):
                while not application.started:
                    if serving.done(): await serving
                    await asyncio.sleep(.01)
                async with websockets.connect(f'ws://127.0.0.1:{port}/ws?domain={domain}&mode=demo',
                    origin=f'http://127.0.0.1:{port}',open_timeout=2,close_timeout=1) as connection:
                    while True:
                        raw=await connection.recv()
                        if isinstance(raw,bytes): continue
                        event=json.loads(raw);events.append(event)
                        if event['type']=='ready': await connection.send(json.dumps({'type':'start'}))
                        if event['type']=='audio_end':
                            # Protocol fixture acknowledgement, not audio-device playback.
                            await connection.send(json.dumps({'type':'playback_finished','playback_id':event['playback_id']}))
                        if event['type']=='state' and event['state']=='LISTENING': break
                    for _ in range(10): await connection.send(bytes([1,0])*512)
                    for _ in range(45): await connection.send(bytes(1024))
                    await entered.wait()
                    if termination=='disconnect': await connection.close()
                    else:
                        await connection.send(json.dumps({'type':'stop'}))
                        while True:
                            raw=await connection.recv()
                            if isinstance(raw,bytes): continue
                            event=json.loads(raw);events.append(event)
                            if event['type']=='done':
                                assert event['status']=='interrupted'
                                break
                await closed.wait()
                assert cancelled.is_set()
                assert sorted(closure)==['asr','bank','router']
                saved,=tmp_path.rglob('demo_session_*.json')
                result=json.loads(saved.read_text(encoding='utf-8'))
                assert result['domain']==domain and result['status']=='interrupted'
                assert result['confirmed'] is False and result['evaluation_eligible'] is False
                assert not any(event['type']=='error' for event in events)
                if termination=='stop':
                    assert any(event['type']=='progress' and event['stage']=='transcribing' and event['active'] for event in events)
        finally:
            application.should_exit=True
            try: await asyncio.wait_for(serving,3)
            finally:
                if not serving.done():
                    serving.cancel();await asyncio.gather(serving,return_exceptions=True)
                listener.close()
    asyncio.run(run())
