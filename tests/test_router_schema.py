import asyncio
import json
from pathlib import Path
import httpx
import pytest
from engine.router import Router, build_prompt, parse_response


@pytest.fixture
def config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/pizza.json').read_text(encoding='utf-8'))


def response(slot='size',value='large'):
    return {'ops':[{'op':'set','slot':slot,'value':value}],'confidence':0.9,'unclear':False,'is_affirmation':False,'is_negation':False}


def test_valid(config):
    assert parse_response(json.dumps(response()),config).ops[0].value=='large'


@pytest.mark.parametrize('raw',['not json','```json\n{}\n```',json.dumps(response('unknown')),json.dumps(response(value='giant')),json.dumps(response('quantity',True))])
def test_reject_entire_response(config,raw):
    with pytest.raises(ValueError): parse_response(raw,config)


def test_extra_and_conflicting_flags(config):
    data=response(); data['spoken_text']='unapproved'
    with pytest.raises(ValueError): parse_response(json.dumps(data),config)
    data=response(); data['is_affirmation']=data['is_negation']=True
    with pytest.raises(ValueError): parse_response(json.dumps(data),config)


def test_prompt_preserves_literal_transcript(config):
    assert '{negate_markers}' in build_prompt(config,{},'{negate_markers}')


def test_retry_once(config,tmp_path):
    calls=[]
    def handler(request):
        calls.append(json.loads(request.content))
        raw='broken' if len(calls)==1 else json.dumps(response())
        return httpx.Response(200,json={'choices':[{'message':{'content':raw}}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router=Router(config,tmp_path,api_key='test',client=client)
            result=await router.route('large',{})
            assert result.ops[0].value=='large' and len(calls)==2
    asyncio.run(run())


def test_invalid_retry_is_unclear(config,tmp_path):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request:httpx.Response(200,json={'choices':[{'message':{'content':'{}'}}]}))) as client:
            router=Router(config,tmp_path,api_key='test',client=client)
            result=await router.route('test',{})
            assert result.unclear and not result.ops and router.invalid_responses==2
    asyncio.run(run())


def test_strict_schema_and_model_payload(config, tmp_path):
    from engine.router import response_format
    fmt = response_format(config)
    schema = fmt['json_schema']['schema']
    assert fmt['type'] == 'json_schema' and fmt['json_schema']['strict'] is True
    for obj in (schema, schema['$defs']['Operation']):
        assert obj['additionalProperties'] is False
        assert set(obj['required']) == set(obj['properties'])
    assert schema['$defs']['Operation']['properties']['slot']['enum'] == [s['id'] for s in config['slots']]
    def handler(request):
        body = json.loads(request.content)
        assert body['model'] == 'openai/gpt-oss-120b'
        assert body['response_format'] == fmt
        assert body['messages'][-1] == {'role':'user','content':'large'}
        return httpx.Response(200, json={'choices': [{'finish_reason':'stop', 'message': {'content': json.dumps(response())}}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            router = Router(config, tmp_path, api_key='test', client=client)
            assert (await router.route('large', {})).ops
    asyncio.run(run())


@pytest.mark.parametrize('status,ids,success', [(200,['working'],True),(200,['other'],False),(401,[],False),(503,[],False)])
def test_startup_model_check(status, ids, success):
    from engine.router import check_models, RouterError
    def handler(request):
        assert str(request.url) == 'https://api.groq.com/openai/v1/models'
        return httpx.Response(status, json={'data':[{'id':name} for name in ids]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            if success:
                assert 'working' in await check_models(['working'], 'secret-test', client=client)
            else:
                with pytest.raises(RouterError) as exc:
                    await check_models(['working'], 'secret-test', client=client)
                assert 'secret-test' not in str(exc.value)
    asyncio.run(run())


def test_model_env_override(monkeypatch):
    from engine.config import ROOT, load_config
    monkeypatch.setenv('GROQ_ROUTER_MODEL','openai/gpt-oss-20b')
    assert load_config(ROOT/'configs/pizza.json')['router']['model'] == 'openai/gpt-oss-20b'


@pytest.mark.parametrize('choice', [
    {'finish_reason':'length', 'message':{'content':json.dumps(response())}},
    {'finish_reason':'stop', 'message':{'content':json.dumps(response()),'refusal':'refused'}},
])
def test_reject_truncated_or_refused(config, tmp_path, choice):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'choices':[choice]}))) as client:
            router = Router(config,tmp_path,api_key='test',client=client)
            assert (await router.route('large', {})).unclear
            assert len(router.calls) == 2 and not any(c['ok'] for c in router.calls)
    asyncio.run(run())


def test_eval_requires_owner_review(config):
    from bench.eval_router import validate_cases
    data = {'cases': [{'id':f'{category}_{i}', 'category':category, 'transcript':'', 'reviewed':False,
                      'state':{}, 'expected':{k:v for k,v in response().items() if k != 'confidence'}}
                     for category in ('straightforward','correction','messy') for i in range(10)]}
    with pytest.raises(ValueError, match='supply a transcript'):
        validate_cases(data, config)
    for case in data['cases']:
        case.update(transcript='test fixture', reviewed=True)
    assert len(validate_cases(data, config)) == 30
    data['cases'][1]['id'] = data['cases'][0]['id']
    with pytest.raises(ValueError, match='unique'):
        validate_cases(data, config)


def test_eval_exact_ops_flags_and_failed_calls():
    from bench.eval_router import matches, summarize
    gold = response()
    actual = {**gold, 'confidence':0.2}
    assert matches(actual, gold)
    assert not matches({**actual, 'is_negation':True}, gold)
    rows = [{'passed':True,'valid':True,'elapsed_ms':100,'category':'straightforward','attempts':[{}]},
            {'passed':False,'valid':False,'elapsed_ms':200,'category':'messy','provider_failure':True,'attempts':[{},{}]}]
    score = summarize(rows)
    assert score['exact_ops_and_flags_percent'] == 50
    assert score['valid_response_percent'] == 50 and score['provider_failure_cases'] == 1
    assert score['retries'] == 1 and score['successful_latency_p95_ms'] == 100


def test_smoke_rejects_empty_or_wrong_format(tmp_path):
    import wave
    from bench.smoke_moulsot import read_audio
    path = tmp_path/'sample.wav'
    for rate, pcm in ((16000,b''), (8000,bytes(512))):
        with wave.open(str(path),'wb') as writer:
            writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(rate); writer.writeframes(pcm)
        with pytest.raises(ValueError): read_audio(path)


def test_moulsot_failure_never_uses_groq(config, tmp_path):
    from engine.stt import SpeechToText, STTError
    seen = []
    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(503)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            stt = SpeechToText(config,tmp_path,primary='moulsot',endpoint='https://example.test',client=client,fallback=False)
            with pytest.raises(STTError): await stt.transcribe(bytes(512))
            assert all('groq' not in url for url in seen)
            assert len(stt.calls) == 1 and stt.calls[0]['provider'] == 'moulsot'
    asyncio.run(run())


@pytest.mark.parametrize('endpoint,expected_auth',[
    ('https://owner-test.hf.space','Bearer hf_fixture'),
    ('https://custom.example',''),
    ('http://owner-test.hf.space',''),
])
def test_moulsot_auth_and_safe_quota_error(config,tmp_path,endpoint,expected_auth):
    from engine.stt import SpeechToText,MoulSotQuotaError
    seen=[]
    def handler(request):
        seen.append(request)
        assert request.headers.get('Authorization','')==expected_auth
        if request.url.path.endswith('/upload'): return httpx.Response(200,json=['/tmp/audio.wav'])
        if request.method=='POST': return httpx.Response(200,json={'event_id':'abc123'})
        detail='You have exceeded your ZeroGPU runs limit. private-provider-text hf_never_show'
        return httpx.Response(200,text='event: error\ndata: '+json.dumps({'error':detail})+'\n\n')
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            stt=SpeechToText(config,tmp_path,primary='moulsot',endpoint=endpoint,client=client,fallback=False,hf_token='hf_fixture')
            with pytest.raises(MoulSotQuotaError,match='runs limit') as error:
                await stt.transcribe(bytes(512))
            assert len(seen)==3
            assert 'private-provider-text' not in str(error.value) and 'hf_never_show' not in str(error.value)
            assert ('authenticated account' in str(error.value))==bool(expected_auth)
    asyncio.run(run())


def test_moulsot_unrecognized_error_is_not_exposed():
    from engine.stt import moulsot_inference_error,MoulSotQuotaError
    for payload in (None,{'error':'private traceback and credentials'},['unexpected']):
        error=moulsot_inference_error(payload,False)
        assert not isinstance(error,MoulSotQuotaError)
        assert str(error)=='MoulSot inference failed. Check the Space container logs.'


def test_lifespan_fails_before_serving_missing_model(monkeypatch):
    from engine import server
    from engine.router import RouterError
    monkeypatch.setenv('GROQ_API_KEY','test-key')
    monkeypatch.setenv('GROQ_ROUTER_MODEL','retired-model')
    async def check(models, key):
        assert models == {'retired-model'} and key == 'test-key'
        raise RouterError('Unavailable model')
    monkeypatch.setattr(server,'check_models',check)
    async def run():
        with pytest.raises(RouterError, match='Unavailable'):
            async with server.lifespan(server.app):
                pytest.fail('Must not serve traffic after failed model check.')
    asyncio.run(run())


def test_lifespan_without_key_allows_capture(monkeypatch):
    from engine import server
    monkeypatch.delenv('GROQ_API_KEY',raising=False)
    async def check(*args):
        pytest.fail('Capture-only startup must not make a provider call.')
    monkeypatch.setattr(server,'check_models',check)
    async def run():
        async with server.lifespan(server.app):
            pass
    asyncio.run(run())
