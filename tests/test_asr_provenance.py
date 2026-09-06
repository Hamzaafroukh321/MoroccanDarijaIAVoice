"""Closed launch metadata and static diagnostics, entirely offline."""
from copy import deepcopy
import json

import pytest

from engine.asr_provenance import (
    VERIFICATION, call_provenance, decode_snapshot, record_response_provenance,
    response_binding, runtime_ack, snapshot_sha256, validate_snapshot,
)
from test_local_demo_supervisor import rig


@pytest.fixture
def snapshot(rig):
    return rig.module.launch_snapshot({'ready': True, 'verification': VERIFICATION,
        'verified_at': '2026-09-06T12:00:00Z', 'selected_device': 'cpu'})


def test_verified_launch_uses_current_pins_and_new_identity(rig, snapshot):
    assets = {name: sha for name, _, sha, _ in rig.module.ASSETS}
    assert snapshot['decoder_sha256'] == assets['moulsot.v0.3.Q4_K_M.gguf']
    assert snapshot['projector_sha256'] == assets['moulsot.v0.3.mmproj-Q8_0.gguf']
    again = rig.module.launch_snapshot({'ready': True, 'verification': VERIFICATION,
        'verified_at': snapshot['verified_at'], 'selected_device': 'cpu'})
    assert again['run_id'] != snapshot['run_id']
    assert 'actual_device' not in snapshot


@pytest.mark.parametrize('check', [{}, {'ready': True}, {'ready': True, 'verification': False},
    {'ready': False, 'verification': VERIFICATION}, {'ready': 1, 'verification': VERIFICATION}])
def test_no_completed_verification_has_no_launch_snapshot(rig, check):
    assert rig.module.launch_snapshot(check) is None


@pytest.mark.parametrize('raw', [None, ''])
def test_absent_snapshot(raw):
    assert decode_snapshot(raw) == (None, 'missing')


@pytest.mark.parametrize('raw', ['[]', 'null', '{', 'x' * 2049, '\ud800',
    '{"run_id":"a","run_id":"b"}', '[' * 1000, 7, {}])
def test_malformed_environment_is_ignored(raw):
    assert decode_snapshot(raw) == (None, 'invalid')


@pytest.mark.parametrize('key,value', [
    ('run_id', 'A' * 32), ('conversion_revision', 'g' * 40),
    ('runtime_release', 'private/build/path'), ('selected_device', 'actual_cuda'),
    ('verified_at', '2026-02-30T12:00:00Z'), ('verified_at', '٢٠٢٦-09-06T12:00:00Z'),
    ('decoder_sha256', 'a' * 63), ('schema_version', True),
])
def test_closed_snapshot_rejects_unsafe_or_ambiguous_facts(snapshot, key, value):
    snapshot[key] = value
    with pytest.raises(ValueError, match='^Invalid local ASR runtime snapshot\\.$'):
        validate_snapshot(snapshot)


def test_snapshot_and_call_records_are_detached(snapshot):
    saved = deepcopy(snapshot)
    decoded, status = decode_snapshot(json.dumps(snapshot))
    record = call_provenance(decoded, status, provider='moulsot', protocol='json',
        endpoint='http://127.0.0.1:8012/transcribe')
    decoded['selected_device'] = 'cuda'
    snapshot['selected_device'] = 'cuda'
    assert record['launch_snapshot'] == saved
    assert record['actual_device'] is None
    assert record['response_binding'] == 'unavailable'


def test_ack_digest_is_canonical_and_not_a_raw_asset_report(snapshot):
    reverse = dict(reversed(list(snapshot.items())))
    assert snapshot_sha256(snapshot) == snapshot_sha256(reverse)
    ack = runtime_ack(snapshot)
    assert set(ack) == {'schema_version', 'run_id', 'snapshot_sha256'}
    assert response_binding(snapshot, ack) == 'matched'
    assert response_binding(None, ack) == 'unverified'
    ack['snapshot_sha256'] = 'b' * 64
    assert response_binding(snapshot, ack) == 'mismatch'


def test_remote_metadata_cannot_become_launch_or_device_evidence(snapshot):
    record = call_provenance(snapshot, 'valid', provider='moulsot', protocol='json',
        endpoint='http://127.0.0.1:8012/transcribe')
    payload = {'runtime_snapshot': runtime_ack(snapshot), 'model_variant': snapshot['model_variant'],
        'actual_device': 'cuda', 'provenance': {'source_model': snapshot['source_model'],
            'quantized': True, 'upstream_model': 'PRIVATE_PATH_SENTINEL'}}
    record_response_provenance(record, snapshot, payload, eligible=True)
    assert record['bridge_reported'] == {'source_model': snapshot['source_model'],
        'model_variant': snapshot['model_variant'], 'quantized': True}
    assert record['actual_device'] is None
    assert 'PRIVATE_PATH_SENTINEL' not in json.dumps(record)


def test_failed_real_preflight_cannot_stamp_verification(rig, monkeypatch):
    monkeypatch.setattr(rig.module, 'ASSETS', [])
    monkeypatch.setattr(rig.module, 'occupied_ports', lambda: [8011])
    config_dir = rig.module.ROOT / 'configs'
    config_dir.mkdir()
    (config_dir / 'pizza.json').write_text(json.dumps({'runtime': {'host': '127.0.0.1', 'port': 8000}}))
    check = rig.real_preflight(verify=True, device='cpu')
    assert check['ready'] is False
    assert 'verification' not in check and 'verified_at' not in check
