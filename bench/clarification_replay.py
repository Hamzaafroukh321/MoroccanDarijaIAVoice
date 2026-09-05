"""Two-call scoped repair diagnostic. No ASR/TTS or performance claims."""
import asyncio
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import httpx
from dotenv import load_dotenv

from engine.config import ROOT, load_config
from engine.demo import demo_config, reply_text
from engine.demo_order import DemoOrderState
from engine.normalize import normalize
from engine.router import Router


async def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reasoning',choices=['low','medium'],default='low')
    parser.add_argument('--followup-only',action='store_true',help='Reuse the legacy clarification WITHOUT a staged proposal; one live call. Omit to test staged recovery.')
    args=parser.parse_args()
    load_dotenv(ROOT/'.env')
    config=demo_config(load_config(ROOT/'configs/pizza.json'))
    config['router']['retries']=0
    config['router']['reasoning_effort']=args.reasoning
    source=json.loads((ROOT/'bench/results/demo_session_d0a4e4c564834d4e999086eddf355e46.json').read_text(encoding='utf-8'))
    original=normalize(source['turns'][2]['transcript'],config)
    state=DemoOrderState(config)
    report={'purpose':'Saved Darija request plus synthetic short follow-up; development diagnostic, not native-reviewed gold',
            'evaluation_eligible':False,'turns':[],'passed':False,'reasoning_effort':args.reasoning}
    async def inspect(response):
        await response.aread()
        if response.is_success:
            content=response.json().get('choices',[{}])[0].get('message',{}).get('content')
            try: report.setdefault('raw_structured_responses',[]).append(json.loads(content))
            except (TypeError,ValueError): pass
    client=httpx.AsyncClient(timeout=30,event_hooks={'response':[inspect]})
    router=Router(config,ROOT,client=client)
    pending=None
    last=None
    last_text=None
    try:
        for index,text in enumerate((original,'كوكا')):
            context=state.router_context()
            context.update(pending_request=pending,last_assistant_action=last,last_assistant_text=last_text,readback_complete=False)
            if args.followup_only and index==0:
                payload={'ops':[],'confidence':1.0,'unclear':False,'is_affirmation':False,'is_negation':False,'intent':'out_of_scope',
                         'clarification':{'kind':'drink_size','slot':'drink','item_ids':[]},'resolves_clarification':None}
            else:
                response=await router.route(text,context)
                payload=response.model_dump()
            action=state.consume(payload)
            spoken=reply_text(action,state.values,config['demo'])
            report['turns'].append({'input':text,'response':payload,'action':action.kind,'spoken':spoken,'state':deepcopy(state.values),'live':not(args.followup_only and index==0)})
            last={'kind':action.kind,'slot':action.slot,'item_id':action.item_id}
            last_text=spoken
            if len(report['turns'])==1:
                if action.kind!='drink_size' or state.values!={'items':[]}:
                    break
                pending={'text':original[:1000],'truncated':len(original)>1000}
        expected={'items':[dict(id=1,quantity=1,size='large',toppings=['cheese']),dict(id=2,quantity=1,size='medium',toppings=['beef'])],'drink':['cola']}
        report['passed']=len(report['turns'])==2 and state.values==expected and action.kind=='readback' and not state.confirmed
    except Exception as exc:
        report['error_type']=type(exc).__name__
        report['error']=str(exc)
    finally:
        report['calls']=router.calls
        await router.close()
        await client.aclose()
        path=ROOT/'bench/results'/('clarification_live_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'.json')
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__': asyncio.run(main())
