import copy
import json
from pathlib import Path

import pytest

from engine.endpointing import EndpointDetector


@pytest.fixture
def config():
    return json.loads((Path(__file__).resolve().parents[1] / 'configs/pizza.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('tail,wait', [('',900), ('euh',1500), ('avec',1800), ('0 6 1',1600), ('euh avec',2400), ('0612345678',900)])
def test_holds(config, tail, wait):
    assert EndpointDetector(config).required_wait(tail) == wait


def test_cap(config):
    config['endpointing']['max_wait_ms'] = 2000
    assert EndpointDetector(config).required_wait('euh avec') == 2000


def test_endpoint_and_blip(config):
    detector = EndpointDetector(config)
    frame = bytes(detector.frame_bytes)
    for _ in range(8): assert detector.feed(frame, 1) is None
    for _ in range(28): assert detector.feed(frame, 0) is None
    segment = detector.feed(frame, 0)
    assert segment.speech_ms == 256
    assert segment.endpoint_ms == 1184
    assert len(segment.pcm) == 8 * len(frame)
    for _ in range(2): detector.feed(frame, 1)
    assert all(detector.feed(frame, 0) is None for _ in range(40))


def test_resuming_speech_cancels_timer(config):
    detector = EndpointDetector(config)
    frame = bytes(detector.frame_bytes)
    for _ in range(10): detector.feed(frame, 1)
    for _ in range(20): detector.feed(frame, 0)
    detector.feed(frame, 1)
    assert detector.silence_ms == 0
    for _ in range(28): assert detector.feed(frame, 0) is None
    assert detector.feed(frame, 0).speech_ms == 352


def test_pending_tail_and_forced_limit(config):
    config = copy.deepcopy(config)
    config['endpointing']['max_segment_ms'] = 1600
    detector = EndpointDetector(config)
    frame = bytes(detector.frame_bytes)
    for _ in range(10): detector.feed(frame, 1)
    for _ in range(39): assert detector.feed(frame, 0, tail_pending=True) is None
    assert detector.feed(frame, 0, tail_pending=True).forced


def test_reject_frame_and_probability(config):
    detector = EndpointDetector(config)
    with pytest.raises(ValueError): detector.feed(b'bad', 0)
    with pytest.raises(ValueError): detector.feed(bytes(detector.frame_bytes), float('nan'))


def test_persistent_quota_windows(config,tmp_path):
    from engine.stt import RateLimiter, RateLimitError
    config['stt']['limits']=[{'window_seconds':60,'requests':2},{'window_seconds':3600,'audio_seconds':5}]
    now=[1000.0]
    limiter=RateLimiter(config,tmp_path,clock=lambda:now[0])
    limiter.reserve(2);limiter.reserve(2)
    with pytest.raises(RateLimitError): limiter.reserve(0)
    now[0]+=61
    restored=RateLimiter(config,tmp_path,clock=lambda:now[0])
    with pytest.raises(RateLimitError): restored.reserve(2)
    restored.reserve(1)
    now[0]+=3600
    restored.reserve(5)


def test_asr_fallback_and_confidence_provenance(config,tmp_path):
    import asyncio
    import httpx
    from engine.stt import SpeechToText
    requests=[]
    def handler(request):
        requests.append(str(request.url))
        if request.url.host=='moulsot.test': return httpx.Response(503)
        return httpx.Response(200,json={'text':'test transcript','segments':[{'avg_logprob':-0.5,'tokens':[1,2]}]})
    async def run():
        config['stt']['moulsot_protocol']='json'
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            stt=SpeechToText(config,tmp_path,endpoint='https://moulsot.test',api_key='test',client=client)
            result=await stt.transcribe(bytes(1024))
            assert result.provider=='groq' and result.confidence==pytest.approx(0.60653066)
            assert result.confidence_kind=='geometric_mean_token_probability'
            assert len(requests)==2
    asyncio.run(run())


def test_moulsot_gradio_adapter_without_invented_confidence(config,tmp_path):
    import asyncio
    import httpx
    from engine.stt import SpeechToText
    def handler(request):
        if request.url.path.endswith('/upload'): return httpx.Response(200,json=['/tmp/audio.wav'])
        if request.method=='POST': return httpx.Response(200,json={'event_id':'abc123'})
        return httpx.Response(200,text='event: complete\ndata: ["fixture transcript"]\n\n')
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            stt=SpeechToText(config,tmp_path,endpoint='https://moulsot.test',api_key='',client=client,fallback=False)
            result=await stt.transcribe(bytes(1024))
            assert result.text=='fixture transcript' and result.confidence is None
    asyncio.run(run())


def test_overlap_requires_two_signals(config):
    import numpy as np
    from engine.overlap import assess
    pcm=np.tile(np.concatenate([np.zeros(512,dtype='<i2'),np.full(512,30000,dtype='<i2')]),10).tobytes()
    assert assess(pcm,'',None,640,config)['unusable']
    result=assess(bytes(1024),'enough words for a frame',None,32,config)
    assert not result['unusable'] and not result['confidence_available']


def test_metric_matching_does_not_hide_merged_boundaries(config):
    from bench.metrics import endpoint_matches, compute
    utterances=[{'start_ms':0,'end_ms':1000},{'start_ms':1200,'end_ms':2000}]
    endpoints=[{'start_ms':0,'endpoint_ms':500},{'start_ms':500,'endpoint_ms':2200}]
    matches=endpoint_matches(utterances,endpoints)
    assert matches['false_cutoffs']==1 and matches['missed_boundaries']==1
    assert matches['latencies_ms']==[1200]
    scenarios=[{'id':'test','utterances':utterances,'gold_state':{'size':'large'},'corrections':[]}]
    metrics=compute(scenarios,[{'scenario_id':'test','state':{'size':'large'},'status':'incomplete','endpoints':endpoints}],config)
    assert metrics['TSR']==0 and metrics['FCR']==0.5 and metrics['CSR'] is None
    assert metrics['boundary_coverage']==0.5


def test_sweep_grid_and_selection(config):
    from bench.sweep_endpointing import variation, choose, chart
    grid=[variation(config,silence,holds) for silence in config['benchmark']['silence_grid_ms'] for holds in config['benchmark']['hold_modes']]
    assert len(grid)==28 and grid[0]['endpointing']['digit_hold_ms']==0
    rows=[{'holds':'all','base_silence_ms':900,'MEL':900,'FCR':0.1,'TSR':80,'boundary_coverage':1}, {'holds':'all','base_silence_ms':1500,'MEL':1600,'FCR':0.0,'TSR':100,'boundary_coverage':1}]
    assert choose(rows,'all',1500)['base_silence_ms']==900
    assert '<svg' in chart(rows,config['benchmark'])
