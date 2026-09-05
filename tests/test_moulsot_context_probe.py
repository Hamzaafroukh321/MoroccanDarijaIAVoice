"""No inference: enforce the frozen ASR experiment's gates and request ceiling."""
import asyncio
import hashlib
import io
import json
import wave

import httpx
import pytest

from bench import moulsot_context_probe as probe


def response(text, finish='stop'):
    return dict(model=probe.MODEL, choices=[dict(finish_reason=finish,
        message=dict(role='assistant', content=probe.PREFIX + text))],
        usage=dict(completion_tokens=12))


def transport(outputs, *, build='b10809-test', format_valid=True):
    requests = []

    def handler(request):
        body = json.loads(request.content) if request.content else None
        requests.append((request.url.path, body))
        if request.url.path == '/props':
            return httpx.Response(200, json=dict(model_path=probe.MODEL, build_info=build, chat_template='fixture'))
        if request.url.path == '/apply-template':
            messages = body['messages']
            prompt = '<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n'
            if messages[0]['role'] == 'system' and format_valid:
                prompt = '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' + prompt
            return httpx.Response(200, json=dict(prompt=prompt))
        assert request.url.path == '/v1/chat/completions'
        return httpx.Response(200, json=outputs.pop(0))

    return httpx.MockTransport(handler), requests


def run(outputs, **kwargs):
    mock, requests = transport(outputs, **kwargs)
    report = {'calls': []}

    async def execute():
        async with httpx.AsyncClient(base_url=probe.BASE_URL, transport=mock) as client:
            await probe.run_probe(client, dict(positive=b'positive', negative=b'negative'), report)

    return execute, report, requests


def test_four_call_ceiling_with_fixed_negative_control_and_unchanged_audio():
    negative = 'لا بدل الساعة خليها مع الحداش'
    execute, report, requests = run([response('الطبيب ألف'), response(probe.CONTEXT),
                                     response(negative), response(negative)])
    asyncio.run(execute())
    calls = [body for path, body in requests if path == '/v1/chat/completions']
    assert len(calls) == 4
    assert report['negative_output_unchanged'] is True
    assert report['negative_vocabulary_inserted'] is False
    for index in (0, 2):
        assert calls[index]['messages'] == calls[index+1]['messages'][1:]
        assert calls[index+1]['messages'][0] == {'role': 'system', 'content': probe.CONTEXT}
        assert {key: value for key, value in calls[index].items() if key != 'messages'} == dict(
            max_tokens=512, temperature=0, cache_prompt=False, stream=False)
    assert all(call['usage']['completion_tokens'] == 12 for call in report['calls'])


@pytest.mark.parametrize('a,b', [('الطبيب ألف', 'الطبيب ألف'),
                                (probe.CONTEXT, probe.CONTEXT), ('nothing', 'الطبيب باء')])
def test_no_negative_calls_without_improvement_covering_both_labels(a, b):
    execute, report, requests = run([response(a), response(b)])
    asyncio.run(execute())
    assert len(report['calls']) == 2
    assert report['status'] == 'stopped_no_exact_label_improvement'


@pytest.mark.parametrize('kwargs', [dict(build='b108090-test'), dict(format_valid=False)])
def test_wrong_runtime_or_dropped_system_context_stops_before_inference(kwargs):
    execute, report, requests = run([], **kwargs)
    with pytest.raises(ValueError):
        asyncio.run(execute())
    assert report['calls'] == []
    assert not any(path == '/v1/chat/completions' for path, body in requests)


@pytest.mark.parametrize('payload', [response('partial', 'length'),
    response('a b c d a b c d a b c d'), response('الطبيب باء الطبيب باء'),
    dict(model=probe.MODEL, choices=[]), response('<|im_start|>')])
def test_malformed_truncated_or_repetitive_output_is_saved_then_stops(payload):
    execute, report, requests = run([payload])
    with pytest.raises(ValueError):
        asyncio.run(execute())
    assert len(report['calls']) == 1
    assert report['calls'][0]['raw_response'] == payload
    assert report['calls'][0]['status'] == 'error'


def test_negative_hint_insertion_is_reported_without_rewriting():
    execute, report, requests = run([response('الطبيب ألف'), response(probe.CONTEXT),
        response('مع الحداش'), response('الطبيب باء مع الحداش')])
    asyncio.run(execute())
    assert report['negative_vocabulary_inserted'] is True
    assert report['negative_output_unchanged'] is False
    assert report['calls'][-1]['text'] == 'الطبيب باء مع الحداش'


def test_changed_frozen_audio_is_rejected_before_decode(tmp_path):
    path = tmp_path / 'input.wav'
    path.write_bytes(b'changed audio')
    with pytest.raises(ValueError, match='SHA256'):
        probe.frozen_audio(tmp_path, dict(path='input.wav', sha256='0'*64))


def test_hash_valid_input_still_requires_complete_mono_pcm16(tmp_path):
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b'\0' * 400)
    raw = stream.getvalue()
    (tmp_path / 'input.wav').write_bytes(raw)
    with pytest.raises(ValueError, match='mono'):
        probe.frozen_audio(tmp_path, dict(path='input.wav', sha256=hashlib.sha256(raw).hexdigest()))


def test_dry_mode_never_opens_http_client(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('Dry mode opened an HTTP client')
    monkeypatch.setattr(probe.httpx, 'AsyncClient', forbidden)
    asyncio.run(probe.main(False))
    report = json.loads(capsys.readouterr().out)
    assert report['calls'] == [] and report['status'] == 'dry'
    assert report['sources']['negative']['sha256'] == '3bd55e3191f458d4aca45adc753e9b434658e4bd45b042ba9e6e169e7bac1ad7'
