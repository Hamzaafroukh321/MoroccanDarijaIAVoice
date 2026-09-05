"""Local configuration recovery; no HTTP requests, startup checks, or providers."""

import asyncio
import json
import re

import pytest

from engine import server
from engine import demo as demo_module
from engine.config import ROOT


@pytest.fixture
def setup_environment(monkeypatch):
    # These placeholders are never sent anywhere. Isolate from the user's keys.
    monkeypatch.setenv('GROQ_API_KEY', 'offline-test-placeholder')
    monkeypatch.setenv('MOULSOT_ENDPOINT', 'https://offline-test.invalid')
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'misspelled_voice_provider')
    # Reviewed-bank readiness is independent of this synthetic setup regression.
    monkeypatch.setattr(server, 'voice_issues', lambda selected: [])


def page_config(response):
    assert response.status_code == 200
    match = re.search(r'<script id="capture-config" type="application/json">(.*?)</script>',
                      response.body.decode('utf-8'), re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def assert_provider_issue(issues):
    assert any('DEMO_TTS_PROVIDER' in issue for issue in issues), issues


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_invalid_demo_provider_is_reported_as_setup_issue(setup_environment, domain):
    selected = server.domain_config(domain)
    assert_provider_issue(server.demo_issues(selected))


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_capture_page_remains_usable_with_invalid_demo_provider(setup_environment, domain):
    response = asyncio.run(server.index(domain=domain, mode='capture'))
    config = page_config(response)
    assert config['domain_id'] == domain
    assert config['initial_mode'] == 'capture'
    assert config['websocket_path'] == '/ws'
    assert_provider_issue(config['demo_issues'])
    assert b'<option value="capture">Microphone check</option>' in response.body


class SetupSocket:
    def __init__(self):
        self.events = []
        self.close_codes = []

    async def send_json(self, event):
        self.events.append(event)

    async def close(self, code):
        self.close_codes.append(code)


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_invalid_demo_provider_rejected_before_allocating_clients(setup_environment, monkeypatch, domain):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid demo setup must not allocate providers or VAD.')

    for name in ('SpeechToText', 'Router', 'DemoVoice', 'SileroVAD', 'VoiceSession'):
        monkeypatch.setattr(server, name, forbidden)
    socket = SetupSocket()
    asyncio.run(server.voice_connection(socket, server.domain_config(domain), demo=True))
    assert len(socket.events) == 1
    assert socket.events[0]['type'] == 'error'
    assert 'DEMO_TTS_PROVIDER' in socket.events[0]['message']
    assert socket.close_codes == [1008]


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_valid_xtts_keeps_registered_demo_metadata(setup_environment, monkeypatch, domain):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    response = asyncio.run(server.index(domain=domain, mode='demo'))
    config = page_config(response)
    assert config['demo_supported'] is True
    assert config['initial_mode'] == 'demo'
    assert config['demo_issues'] == []
    assert config['demo_tts_provider'] == 'darija_xtts'
    assert config['demo_tts_voice_label']
    assert config['demo_max_session_ms'] > 0
    assert config['demo_ui']['state_title']
    assert config['demo_labels'] and config['demo_values']


@pytest.fixture
def missing_clinic_question(setup_environment, monkeypatch, tmp_path):
    directory = tmp_path / 'configs/demo'
    directory.mkdir(parents=True)
    profile = json.loads((ROOT / 'configs/demo/clinic.json').read_text(encoding='utf-8'))
    del profile['questions']['doctor']
    (directory / 'clinic.json').write_text(json.dumps(profile), encoding='utf-8')
    (directory / 'voice.json').write_bytes((ROOT / 'configs/demo/voice.json').read_bytes())
    monkeypatch.setattr(demo_module, 'ROOT', tmp_path)
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')


def test_mapping_error_names_missing_question_and_preserves_capture(missing_clinic_question):
    config = page_config(asyncio.run(server.index(domain='clinic', mode='capture')))
    assert config['initial_mode'] == 'capture'
    assert config['demo_issues'] == ['demo.questions.doctor is required for the selected field.']


def test_mapping_error_rejects_socket_before_provider_allocation(missing_clinic_question, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid speech mappings must not allocate providers.')

    for name in ('SpeechToText', 'Router', 'DemoVoice', 'SileroVAD', 'VoiceSession'):
        monkeypatch.setattr(server, name, forbidden)
    socket = SetupSocket()
    asyncio.run(server.voice_connection(socket, server.domain_config('clinic'), demo=True))
    assert socket.close_codes == [1008]
    assert socket.events == [{'type': 'error', 'message': 'demo.questions.doctor is required for the selected field.'}]
