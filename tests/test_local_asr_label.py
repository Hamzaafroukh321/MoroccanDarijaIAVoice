"""Configured ASR location is visible without exposing endpoint credentials."""
import asyncio
import json
import re

import pytest

from engine import server


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_demo_page_reports_configured_local_asr_without_endpoint(monkeypatch, domain):
    monkeypatch.setenv('MOULSOT_PROTOCOL', 'json')
    monkeypatch.setenv('MOULSOT_ENDPOINT', 'http://user:private@127.0.0.1:8012/transcribe')
    page = asyncio.run(server.index(domain=domain, mode='demo')).body.decode()
    public = json.loads(re.search(r'<script id="capture-config" type="application/json">(.*?)</script>', page, re.S)[1])
    assert public['demo_asr_label'] == 'MoulSot on this computer'
    assert 'private' not in page and '8012/transcribe' not in page


def test_default_hosted_label_and_invalid_override_preflight(monkeypatch):
    monkeypatch.setenv('MOULSOT_ENDPOINT', 'https://example.hf.space')
    monkeypatch.delenv('MOULSOT_PROTOCOL', raising=False)
    config = server.domain_config('clinic')
    assert server.demo_asr_label(config) == 'Hosted MoulSot'
    monkeypatch.setenv('MOULSOT_PROTOCOL', 'whisper')
    assert 'Set MOULSOT_PROTOCOL to json or gradio.' in server.demo_issues(config)
