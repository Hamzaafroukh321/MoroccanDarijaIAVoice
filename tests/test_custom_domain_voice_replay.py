"""Configured third-domain replay validation stays offline and fails early."""

import io
import json
from pathlib import Path
import sys
import wave
import websockets

import httpx
import pytest

from bench import voice_manifest, voice_transport
from engine import demo
from engine.config import ROOT as SOURCE_ROOT


@pytest.fixture
def configured_replay(monkeypatch, tmp_path):
    domain = 'service-desk'
    configs = tmp_path / 'configs'
    profiles = configs / 'demo'
    profiles.mkdir(parents=True)
    (profiles / 'voice.json').write_bytes((SOURCE_ROOT / 'configs/demo/voice.json').read_bytes())
    base = json.loads((SOURCE_ROOT / 'configs/clinic.json').read_text(encoding='utf-8').replace('doctor', 'counter'))
    base.update(domain_id=domain, display_name='Fictional counter preferences')
    profile = json.loads((SOURCE_ROOT / 'configs/demo/clinic.json').read_text(encoding='utf-8').replace('doctor', 'counter'))
    base_path = configs / f'{domain}.json'
    profile_path = profiles / f'{domain}.json'
    base_path.write_text(json.dumps(base), encoding='utf-8')
    profile_path.write_text(json.dumps(profile), encoding='utf-8')
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b'\x01\x00' * 512)
    audio_path = tmp_path / 'fixture.wav'
    audio_path.write_bytes(output.getvalue())
    manifest_path = tmp_path / 'manifest.json'
    data = {'domain': domain, 'turns': [{'audio': audio_path.name,
            'source_kind': 'synthetic_diagnostic', 'expected_slots': {'counter': 'counter_b'}}]}
    manifest_path.write_text(json.dumps(data), encoding='utf-8')
    monkeypatch.setattr(voice_manifest, 'ROOT', tmp_path, raising=False)
    monkeypatch.setattr(voice_transport, 'ROOT', tmp_path)
    monkeypatch.setattr(demo, 'ROOT', tmp_path)
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'darija_xtts')
    def forbidden(*args, **kwargs):
        pytest.fail('Offline validation attempted transport or provider allocation')
    monkeypatch.setattr(httpx, 'Client', forbidden)
    monkeypatch.setattr(httpx, 'AsyncClient', forbidden)
    monkeypatch.setattr(websockets, 'connect', forbidden)
    monkeypatch.setattr(voice_transport, 'run_transport', forbidden)
    monkeypatch.setattr(voice_transport, 'run_manifest_transport', forbidden)
    return dict(root=tmp_path, domain=domain, base=base, profile=profile,
        base_path=base_path, profile_path=profile_path, audio=audio_path,
        manifest=manifest_path, data=data)


@pytest.mark.parametrize('mode', ['manifest_default', 'manifest_check', 'audio_check'])
def test_config_added_domain_replay_validates_without_network(configured_replay, monkeypatch, capsys, mode):
    fixture = configured_replay
    if mode == 'audio_check':
        flags = ['--audio', str(fixture['audio']), '--domain', fixture['domain'],
                 '--source-kind', 'synthetic_diagnostic', '--check']
    else:
        flags = ['--manifest', str(fixture['manifest'])]
        if mode == 'manifest_check':
            flags += ['--execute', '--check', '--domain', fixture['domain']]
    monkeypatch.setattr(sys, 'argv', ['voice_transport.py', *flags])
    assert voice_transport.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report['network_used'] is False
    assert report['status'] == ('input_validated_only' if mode == 'audio_check' else 'manifest_validated_only')
    if mode != 'audio_check':
        assert report['domain'] == fixture['domain']
        assert report['turns'][0]['expected_slots'] == {'counter': 'counter_b'}
        assert report['turns'][0]['source_kind'] == 'synthetic_diagnostic'
    assert not (fixture['root'] / 'bench/results').exists()


@pytest.mark.parametrize('failure,mode', [
    ('traversal', 'manifest'), ('traversal', 'audio'),
    ('unsafe', 'manifest'), ('missing_config', 'audio'),
    ('missing_profile', 'manifest'), ('mismatched_id', 'manifest'),
    ('malformed_profile', 'audio'), ('unsupported_profile', 'manifest'),
    ('explicit_mismatch', 'manifest')])
def test_invalid_configured_domain_fails_before_connection(configured_replay, monkeypatch, capsys, failure, mode):
    fixture = configured_replay
    domain = fixture['domain']
    if failure == 'traversal':
        domain = '../service-desk'
    elif failure == 'unsafe':
        domain = 'ServiceDesk'
    elif failure == 'missing_config':
        fixture['base_path'].unlink()
    elif failure == 'missing_profile':
        fixture['profile_path'].unlink()
    elif failure == 'mismatched_id':
        fixture['base']['domain_id'] = 'different-id'
        fixture['base_path'].write_text(json.dumps(fixture['base']), encoding='utf-8')
    elif failure == 'malformed_profile':
        fixture['profile_path'].write_text('{', encoding='utf-8')
    elif failure == 'unsupported_profile':
        fixture['profile']['state_kind'] = 'collection_scoped'
        fixture['profile_path'].write_text(json.dumps(fixture['profile']), encoding='utf-8')
    if mode == 'audio':
        flags = ['--audio', str(fixture['audio']), '--domain', domain, '--source-kind', 'synthetic_diagnostic']
    else:
        fixture['data']['domain'] = domain
        fixture['manifest'].write_text(json.dumps(fixture['data']), encoding='utf-8')
        flags = ['--manifest', str(fixture['manifest']), '--execute']
        if failure == 'explicit_mismatch':
            flags += ['--domain', 'clinic']
    monkeypatch.setattr(sys, 'argv', ['voice_transport.py', *flags])
    assert voice_transport.main() == 2
    output = capsys.readouterr().out
    assert 'NOT RUN:' in output
    assert any(word in output.lower() for word in ('domain', 'config', 'profile', 'preview', 'flat', 'json', 'file'))
    assert not (fixture['root'] / 'bench/results').exists()
