"""Multi-turn runner over real local TCP/ASGI, with explicitly fake providers."""

import asyncio
from copy import deepcopy
import json
import socket
from types import SimpleNamespace

import pytest
import uvicorn
import websockets

from bench import voice_transport
from engine import server
from engine.config import ROOT, load_config
from engine.demo import demo_config, reply_text
from engine.responder import RenderedAudio
from engine.stt import Transcript, wav_bytes


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_three_turn_loopback_corrects_then_confirms_one_task(monkeypatch, tmp_path, domain):
    base = load_config(ROOT / 'configs' / f'{domain}.json')
    config = demo_config(base)
    monkeypatch.setattr(server, 'ROOT', tmp_path)
    monkeypatch.setattr(server, 'domain_config', lambda value: deepcopy(base))
    monkeypatch.setattr(server, 'demo_issues', lambda value: [])
    monkeypatch.setattr(server, 'SileroVAD', lambda value: lambda pcm: 1.0 if pcm[0] else 0.0)
    original_connect = websockets.connect

    async def run():
        calls, closed = [], []

        class ASR:
            def __init__(self, *args, **kwargs):
                assert kwargs == {'primary': 'moulsot', 'fallback': False}

            async def transcribe(self, pcm):
                calls.append(('asr', pcm[0]))
                # This label checks protocol routing only: no MoulSot inference.
                return Transcript(f'offline fixture turn {pcm[0]}', .99, 'moulsot')

            async def close(self):
                closed.append('asr')

        if domain == 'clinic':
            first = {'doctor': 'doctor_a', 'date': '2026-09-15', 'time': '10:30'}
            final = {**first, 'time': '11:00'}
            initial_ops = [{'op': 'set', 'slot': key, 'value': value} for key, value in first.items()]
            correction = [{'op': 'set', 'slot': 'time', 'value': '11:00'}]
        else:
            item = {'id': 1, 'quantity': 1, 'size': 'large', 'toppings': ['cheese']}
            first = {'items': [item], 'drink': ['water']}
            final = {'items': [{**item, 'size': 'small'}], 'drink': ['water']}
            initial_ops = [{'op': 'create', 'item_id': 1, 'slot': None, 'value': None}]
            initial_ops += [{'op': 'set', 'item_id': 1, 'slot': key, 'value': value}
                            for key, value in item.items() if key != 'id']
            initial_ops.append({'op': 'set', 'item_id': None, 'slot': 'drink', 'value': ['water']})
            correction = [{'op': 'set', 'item_id': 1, 'slot': 'size', 'value': 'small'}]

        class Router:
            def __init__(self, *args):
                self.turn = 0

            async def route(self, text, context):
                self.turn += 1
                calls.append(('router', self.turn))
                return {'intent': 'task', 'ops': initial_ops if self.turn == 1 else correction if self.turn == 2 else [],
                        'unclear': False, 'is_affirmation': self.turn == 3, 'is_negation': False}

            async def close(self):
                closed.append('router')

        class Bank:
            def __init__(self, selected, *args):
                self.config = selected

            async def render(self, action, values):
                return RenderedAudio(wav_bytes(bytes(1024), config['runtime']), 512, 512,
                                     ['offline silence'], reply_text(action, values, self.config['demo']))

            async def close(self):
                closed.append('bank')

        monkeypatch.setattr(server, 'SpeechToText', ASR)
        monkeypatch.setattr(server, 'Router', Router)
        monkeypatch.setattr(server, 'DemoVoice', Bank)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(('127.0.0.1', 0))
        listener.listen(8)
        port = listener.getsockname()[1]

        def connect(uri, **kwargs):
            kwargs['origin'] = f'http://127.0.0.1:{port}'
            return original_connect(uri.replace('127.0.0.1:8000', f'127.0.0.1:{port}'), **kwargs)

        monkeypatch.setattr(websockets, 'connect', connect)
        application = uvicorn.Server(uvicorn.Config(server.app, host='127.0.0.1', port=port,
            lifespan='off', ws='websockets', ws_max_size=1024, ws_per_message_deflate=False,
            log_config=None, log_level='critical'))
        serving = asyncio.create_task(application.serve(sockets=[listener]))
        report = {'events': [], 'outputs': [], 'timings_ms': {}, 'frames_sent': 0}
        turns = [{'pcm': bytes([index, 0]) * 5120, 'source_kind': 'synthetic_diagnostic',
                  'expected_action': 'accepted' if index == 3 else 'readback',
                  'expected_slots': first if index == 1 else final} for index in (1, 2, 3)]
        try:
            async with asyncio.timeout(40):
                while not application.started:
                    if serving.done():
                        await serving
                    await asyncio.sleep(.01)
                await voice_transport.run_manifest_transport(SimpleNamespace(domain=domain, timeout_seconds=35),
                    turns, config, report, tmp_path)
                assert report['server_status'] == 'demo_completed'
                assert report['final_state'] == final
                assert [turn['reply_action'] for turn in report['turns']] == ['readback', 'readback', 'accepted']
                assert all(turn['status'] == 'completed' for turn in report['turns'])
                assert [turn['readback_complete'] for turn in report['turns']] == [True, True, True]
                assert 'stop_sent' not in report['timings_ms']
                assert [value for kind, value in calls if kind == 'asr'] == [1, 2, 3]
                assert [value for kind, value in calls if kind == 'router'] == [1, 2, 3]
                assert len(report['outputs']) == 4
                for output in report['outputs']:
                    assert output['received_bytes'] == output['announced_bytes'] == 1068
                while len(closed) != 3:
                    await asyncio.sleep(.01)
                saved, = tmp_path.rglob('demo_session_*.json')
                result = json.loads(saved.read_text(encoding='utf-8'))
                assert result['status'] == 'demo_completed' and result['confirmed'] is True
                assert result['evaluation_eligible'] is False
        finally:
            (tmp_path / 'loopback_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            application.should_exit = True
            try:
                await asyncio.wait_for(serving, 3)
            finally:
                if not serving.done():
                    serving.cancel()
                    await asyncio.gather(serving, return_exceptions=True)
                listener.close()
    asyncio.run(run())
