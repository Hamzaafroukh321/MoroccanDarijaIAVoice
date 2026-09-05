"""Conservative recovery for observed mixed clarification/field extraction."""

import json
import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.router import parse_response
from engine.scoped_task import ScopedTaskState


def mixed(**changes):
    result=dict(ops=[dict(op='set',slot='time',value='10:30')],confidence=.8,
                unclear=False,is_affirmation=False,is_negation=False,intent='task',
                clarification=dict(kind='unsupported_value',slot='doctor',item_ids=[]),
                proposed_ops=[],resolves_clarification=None,discard_clarification=None)
    result.update(changes)
    return result


def test_mixed_extraction_becomes_only_an_uncommitted_proposal():
    config=demo_config(load_config(ROOT/'configs/clinic.json'))
    response=parse_response(json.dumps(mixed()),config)
    assert response.ops==[] and response.intent=='out_of_scope'
    assert response._mixed_proposal_normalized and response._clarification_intent_normalized
    state=ScopedTaskState(config)
    state.consume(response.model_dump())
    assert state.values=={} and state.version==0 and not state.confirmed
    assert state.pending_proposal['state']=={'time':'10:30'}
    with pytest.raises(ValueError): state.consume(response.model_dump())
    assert state.values=={} and state.pending_proposal['state']=={'time':'10:30'}


@pytest.mark.parametrize('changes',[
    {'is_affirmation':True}, {'is_negation':True}, {'unclear':True},
    {'resolves_clarification':1}, {'discard_clarification':1},
    {'ops':[dict(op='set',slot='doctor',value='doctor_b')]},
    {'ops':[dict(op='set',slot='time',value='25:99')]},
    {'ops':[dict(op='set',slot='missing',value='value')]},
    {'ops':[dict(op='set',slot='time',value='10:30')]*41},
    {'clarification':dict(kind='ambiguous_value',slot='doctor',item_ids=[])},
    {'proposed_ops':[dict(op='set',slot='date',value='2026-09-15')]},
])
def test_mixed_recovery_does_not_authorize_conflicting_or_invalid_operations(changes):
    config=demo_config(load_config(ROOT/'configs/clinic.json'))
    with pytest.raises(ValueError): parse_response(json.dumps(mixed(**changes)),config)
