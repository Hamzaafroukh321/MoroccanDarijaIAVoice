import asyncio
import json

import httpx
import pytest

from engine.config import ROOT, load_config
from engine.demo import DemoTaskState, demo_config, reply_text
from engine.router import Router, RouterError, RouterOutputError, parse_response, response_format


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER','darija_xtts')
    result=demo_config(load_config(ROOT/'configs/pizza.json'))
    result['demo']['order_schema_version']=1  # Retain explicit coverage of the legacy guard.
    return result


def result(intent='unclear',ops=None,yes=False):
    return {'ops':ops or [],'confidence':.9,'unclear':intent!='task',
        'is_affirmation':yes,'is_negation':False,'intent':intent}


def test_repair_asks_only_for_missing_drink_and_preserves_order(config):
    state=DemoTaskState(config)
    state.apply([{'op':'set','slot':k,'value':v} for k,v in
        {'quantity':2,'size':'large','toppings':['cheese']}.items()])
    before=dict(state.values)
    for _ in range(config['demo']['max_repairs']):
        action=state.consume(result())
        assert (action.kind,action.slot)==('repair','drink')
        assert reply_text(action,state.values,config['demo'])==config['demo']['questions']['drink']
        assert state.values==before and not state.confirmed
    assert state.consume(result()).kind=='handoff'


def test_greeting_does_not_blame_user_or_exhaust_repair_budget(config):
    state=DemoTaskState(config)
    for _ in range(5):
        action=state.consume(result('greeting'))
        assert action.kind=='repair' and action.slot=='size'
        assert state.repair_count==0


def test_valid_slot_resets_consecutive_repair_count(config):
    state=DemoTaskState(config)
    state.consume(result());assert state.repair_count==1
    state.consume(result('task',[{'op':'set','slot':'size','value':'small'}]))
    assert state.repair_count==0


def test_identical_slot_does_not_reset_failed_repair_budget(config):
    state=DemoTaskState(config)
    payload=result('task',[{'op':'set','slot':'size','value':'small'}])
    state.consume(payload)
    state.consume(result());state.consume(payload)
    assert state.repair_count==1


def test_multiple_items_never_mutate_or_confirm_existing_state(config):
    state=DemoTaskState(config)
    state.apply([{'op':'set','slot':k,'value':v} for k,v in
        {'quantity':2,'size':'large','toppings':['cheese'],'drink':['cola']}.items()])
    state.begin_confirmation(state.version)
    before=dict(state.values)
    assert state.consume(result('multiple_items')).kind=='multiple_items'
    assert state.values==before and state.readback_version is None
    assert state.consume(result('task',yes=True)).kind=='handoff'
    assert not state.confirmed


def test_multiple_items_schema_rejects_partial_application(config):
    bad=result('multiple_items',[{'op':'set','slot':'size','value':'large'}])
    bad['unclear']=False
    with pytest.raises(ValueError): parse_response(json.dumps(bad),config)
    schema=response_format(config)['json_schema']['schema']
    assert 'intent' in schema['required']
    assert 'multiple_items' in schema['properties']['intent']['enum']
    clear_but_unsupported=result('multiple_items');clear_but_unsupported['unclear']=False
    assert parse_response(json.dumps(clear_but_unsupported),config).intent=='multiple_items'


@pytest.mark.parametrize('status',[503,429,200])
def test_demo_router_distinguishes_provider_failure_from_invalid_output(config,tmp_path,status):
    def handle(request):
        return httpx.Response(status,json={'choices':[{'message':{'content':'{}'}}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            router=Router(config,tmp_path,api_key='fixture',client=client)
            expected = RouterOutputError if status == 200 else RouterError
            message = 'interpret that turn' if status == 200 else ('rate limit' if status == 429 else 'router could not return')
            with pytest.raises(expected,match=message) as failure:
                await router.route('pizza',{})
            assert type(failure.value) is expected
            assert len(router.calls)==(1 if status==429 else config['router']['retries']+1)
    asyncio.run(run())
