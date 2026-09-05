"""Expose held candidate values separately from committed state, without ops."""
import asyncio
from copy import deepcopy

import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config
from engine.pipeline import TurnState, VoiceSession


@pytest.mark.parametrize('domain', ['pizza', 'clinic'])
def test_pending_view_is_separate_copied_and_cleared_on_discard(tmp_path, domain):
    async def run():
        events=[]
        async def emit(event): events.append(event)
        session=VoiceSession(demo_config(load_config(ROOT/'configs'/f'{domain}.json')),
            tmp_path, None, None, None, None, emit, emit)
        payload=dict(ops=[], unclear=False, is_affirmation=False, is_negation=False,
            intent='out_of_scope', resolves_clarification=None, discard_clarification=None)
        if domain=='clinic':
            payload.update(clarification={'kind':'unsupported_value','slot':'doctor','item_ids':[]},
                proposed_ops=[{'op':'set','slot':'time','value':'10:30'}])
        else:
            payload.update(clarification={'kind':'unsupported_drink','slot':'drink','item_ids':[]},
                proposed_ops=[{'op':'create','item_id':1,'slot':None,'value':None},
                              {'op':'set','item_id':1,'slot':'size','value':'large'}])
        committed=deepcopy(session.task.values)
        session.task.consume(payload)
        await session.set_mode(TurnState.SPEAKING)
        view=events[-1]
        assert view['slots']==committed
        assert view['pending_proposal']['state']==session.task.pending_proposal['state']
        assert set(view['pending_proposal'])=={'state'}
        view['pending_proposal']['state'].clear()
        assert session.task.pending_proposal['state']
        clarification_id=session.task.pending_clarification['id']
        session.task.consume(dict(ops=[],unclear=False,is_affirmation=False,is_negation=False,
            intent='task',discard_clarification=clarification_id))
        await session.set_mode(TurnState.LISTENING)
        assert events[-1]['pending_proposal'] is None
        assert events[-1]['pending_clarification'] is None
        assert events[-1]['slots']==committed
        await session.close()
    asyncio.run(run())
