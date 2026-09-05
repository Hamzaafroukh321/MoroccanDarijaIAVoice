"""Saved synthetic input provenance checks; no provider access."""
import hashlib
import json

import pytest

from bench import spoken_choice_probe as probe


@pytest.fixture
def saved(tmp_path, monkeypatch):
    source = tmp_path/'question.json'
    source.write_text('{"fixture": true}', encoding='utf-8')
    monkeypatch.setattr(probe, 'SOURCE', source)
    wav = b'fixture audio bytes; WAV validation occurs in read_audio'
    (tmp_path/'synthetic_answer.wav').write_bytes(wav)
    report = tmp_path/'report.json'
    record = dict(evaluation_eligible=False, synthetic_input_text=probe.ANSWER,
                  source_report_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  input_sha256=hashlib.sha256(wav).hexdigest())
    report.write_text(json.dumps(record), encoding='utf-8')
    return report, record, wav


def test_verified_input_reuses_original_bytes(saved):
    report, _, wav = saved
    assert probe.saved_input(report) == wav


@pytest.mark.parametrize('field,value', [
    ('evaluation_eligible', True), ('synthetic_input_text', 'another answer'),
    ('source_report_sha256', 'wrong'), ('input_sha256', 'wrong'),
])
def test_mismatched_input_is_rejected(saved, field, value):
    report, record, _ = saved
    record[field] = value
    report.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError):
        probe.saved_input(report)
