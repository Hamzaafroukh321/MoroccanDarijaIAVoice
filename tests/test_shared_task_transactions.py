"""The same transaction implementation backs collection and appointment tasks.

Canonical fixtures only: no native-reviewed speech or real clinic availability.
"""
from copy import deepcopy

import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.demo_order import DemoOrderState
from engine.state import TaskState


def reply(ops=None,yes=False):
    return dict(ops=ops or [],unclear=False,is_affirmation=yes,is_negation=False)


@pytest.fixture(params=['pizza','clinic'])
def task_case(request):
    config=load_config(ROOT/'configs'/f'{request.param}.json')
    if request.param=='pizza':
        state=DemoOrderState(demo_config(config))
        def operation(field,value):
            return dict(op='set',item_id=None if field=='drink' else 1,slot=field,value=value)
        initial=[dict(op='create',item_id=1,slot=None,value=None),operation('size','large'),
                 operation('quantity',2),operation('toppings',['cheese']),operation('drink',['water'])]
        change=operation('size','small')
        invalid=operation('size','invalid_size')
    else:
        config['slots'][0]['values']=['doctor_fixture']
        state=TaskState(config)
        def operation(field,value): return dict(op='set',slot=field,value=value)
        initial=[operation('doctor','doctor_fixture'),operation('date','15/09/2026'),
                 operation('time','9h30'),operation('patient_name','Fixture Patient'),operation('phone','+212712345678')]
        change=operation('time','10h30')
        invalid=operation('date','not a date')
    return state,initial,change,invalid


def test_both_task_shapes_have_atomic_corrections_and_fresh_confirmation(task_case,monkeypatch):
    from engine import transactions
    executor=transactions.apply_transaction
    calls=[]
    def observed(*args,**kwargs):
        calls.append(kwargs['config']['domain_id'])
        return executor(*args,**kwargs)
    monkeypatch.setattr(transactions,'apply_transaction',observed)
    state,initial,change,invalid=task_case
    assert state.consume(reply(initial)).kind=='readback'
    assert state.ready and not state.confirmed
    original=deepcopy(state.values);version=state.version
    with pytest.raises(ValueError): state.apply([change,invalid])
    assert state.values==original and state.version==version
    state.begin_confirmation(version)
    assert state.consume(reply([change],yes=True)).kind=='readback'
    assert state.values!=original and not state.confirmed
    state.begin_confirmation(version)
    assert not state.confirmed
    state.begin_confirmation(state.version)
    assert state.consume(reply(yes=True)).kind=='accepted'
    assert state.confirmed
    assert calls and set(calls)=={state.config['domain_id']}
    if state.config['domain_id']=='clinic':
        assert state.values['date']=='2026-09-15'
        assert state.values['time']=='10:30'
        assert state.values['phone']=='0712345678'


def test_redundant_correction_cannot_reuse_readback(task_case):
    state,initial,_,_=task_case
    state.consume(reply(initial))
    state.begin_confirmation(state.version)
    state.consume(reply([initial[-1]],yes=True))
    assert not state.confirmed and state.readback_version is None
    assert state.consume(reply(yes=True)).kind=='readback'
    state.begin_confirmation(state.version)
    assert state.consume(reply(yes=True)).kind=='accepted'


def test_negation_and_interruption_revoke_acceptance(task_case):
    state,initial,_,_=task_case
    state.consume(reply(initial))
    state.begin_confirmation(state.version)
    state.consume(reply(yes=True))
    assert state.confirmed
    state.consume(dict(reply(),is_negation=True))
    assert not state.confirmed and state.readback_version is None
    state.consume(reply(yes=True))
    assert not state.confirmed
    state.begin_confirmation(state.version)
    state.invalidate_confirmation()  # Interrupted playback cannot be accepted.
    state.consume(reply(yes=True))
    assert not state.confirmed


def test_pizza_adapter_rejects_a_collection_its_dialogue_cannot_address():
    config=demo_config(load_config(ROOT/'configs/pizza.json'))
    config['demo']['transaction_schema']['collection']='products'
    with pytest.raises(ValueError,match='pizza dialogue adapter'):
        DemoOrderState(config)
