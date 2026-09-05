"""Offline transport/control checks, not an ASR or Darija quality evaluation."""
import asyncio
from copy import deepcopy
import json

import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config, reply_text
from engine.endpointing import Segment
from engine.pipeline import VoiceSession
from engine.responder import RenderedAudio
from engine.stt import Transcript, wav_bytes


@pytest.mark.parametrize('long_request',[False,True])
@pytest.mark.parametrize('partial_resolution',[False,True])
def test_retained_request_resolves_then_requires_new_readback(tmp_path,long_request,partial_resolution):
    config=demo_config(load_config(ROOT/'configs/pizza.json'))
    original=('one large cheese pizza and a large bottle of cola' if not long_request else 'pizza '*250)
    contexts=[]
    class ASR:
        count=0
        async def transcribe(self,pcm):
            result=([original,'cola','large','yes'] if partial_resolution else [original,'cola','yes'])[self.count]
            self.count+=1
            return Transcript(result,.99,'fixture')
    def operation(kind,item_id,slot=None,value=None):
        return dict(op=kind,item_id=item_id,slot=slot,value=value)
    class Route:
        count=0
        async def route(self,text,state):
            contexts.append(deepcopy(state))
            self.count+=1
            response=dict(intent='task',ops=[],unclear=False,is_affirmation=False,is_negation=False,clarification=None,resolves_clarification=None)
            if self.count==1:
                response.update(intent='out_of_scope',clarification=dict(kind='drink_size',slot='drink',item_ids=[]))
            elif self.count==2:
                assert state['pending_request']['truncated']==long_request
                assert len(state['pending_request']['text'])<=1000
                response.update(resolves_clarification=state['pending_clarification']['id'],ops=[operation('set',None,'drink',['cola'])])
                if not partial_resolution:
                    response['ops'][:0]=[operation('create',1),operation('set',1,'size','large'),operation('set',1,'quantity',1),operation('set',1,'toppings',['cheese'])]
            elif self.count==3 and partial_resolution:
                assert state['pending_clarification'] is None
                assert state['pending_request']['clarification_resolved'] is True
                assert state['state']['drink']==['cola']
                response['ops']=[operation('create',1),operation('set',1,'size','large'),operation('set',1,'quantity',1),operation('set',1,'toppings',['cheese'])]
            else:
                assert state['pending_request'] is None and state['pending_clarification'] is None
                assert state['readback_complete']
                response['is_affirmation']=True
            return response
    class Bank:
        async def render(self,action,values):
            return RenderedAudio(wav_bytes(bytes(1024),config['runtime']),512,512,['fixture'],reply_text(action,values,config['demo']))
    async def run():
        async def emit(event):
            if event['type']=='audio_end': await voice.playback_finished(event['playback_id'])
        async def audio(data): pass
        voice=VoiceSession(config,tmp_path,ASR(),Route(),Bank(),None,emit,audio)
        await voice.start();await voice.playback_task
        for turn in range(4 if partial_resolution else 3):
            await voice.queue.put(Segment(bytes([turn+1,0])*512,0,320,1728,320))
            await voice.queue.join();await voice.playback_task
            if turn==0:
                assert voice.task.values=={'items':[]} and voice.pending_request
            if turn==1:
                if partial_resolution:
                    assert not voice.task.ready and voice.pending_request['clarification_resolved']
                else:
                    assert voice.task.ready and not voice.task.confirmed and voice.pending_request is None
        assert voice.task.confirmed and voice.status=='demo_completed'
        await voice.close()
        saved=json.loads(next(tmp_path.glob('bench/results/demo_session_*.json')).read_text(encoding='utf-8'))
        assert saved['turns'][1]['router_context']['pending_request']['text']==contexts[1]['pending_request']['text']
        assert saved['turns'][0]['state']=={'items':[]}
    asyncio.run(run())
