"""Opt-in, bounded live development diagnostic; not a reviewed router evaluation."""
import asyncio
import argparse
from copy import deepcopy
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
    parser.add_argument('--explicit',action='store_true',help='Replay the later, more explicit saved turn.')
    parser.add_argument('--synthetic',action='store_true',help='English control fixture; not Darija accuracy.')
    args=parser.parse_args()
    load_dotenv(ROOT/'.env')
    config=demo_config(load_config(ROOT/'configs/pizza.json'))
    config['router']['retries']=0
    source=json.loads((ROOT/'bench/results/demo_repair_live_regression_20260905.json').read_text(encoding='utf-8'))
    transcript=source['rows'][0]['transcript']
    if args.explicit:
        session=json.loads((ROOT/'bench/results/demo_session_d0a4e4c564834d4e999086eddf355e46.json').read_text(encoding='utf-8'))
        transcript=session['turns'][2]['transcript']
    if args.synthetic:
        transcript='One large cheese pizza and one medium beef pizza, with cola.'
    state=DemoOrderState(config)
    report={'purpose':'One saved-transcript replay; interpreted expected values, not native-reviewed gold',
            'evaluation_eligible':False,'transcript':transcript,'model':config['router']['model'],
            'max_tokens':config['router']['max_tokens'],'reasoning_effort':config['router'].get('reasoning_effort')}
    if args.synthetic: report['purpose']='Synthetic English control for multi-item routing, not Darija accuracy or microphone evidence'
    async def inspect_response(response):
        await response.aread()
        if response.is_error:
            try:
                error=response.json().get('error',{})
                report['provider_error']={key:error.get(key) for key in ('code','type','message')}
                report['failed_generation_tail']=str(error.get('failed_generation',''))[-400:]
            except ValueError: pass
    client=httpx.AsyncClient(timeout=30,event_hooks={'response':[inspect_response]})
    router=Router(config,ROOT,client=client)
    try:
        response=await router.route(normalize(transcript,config),state.router_context())
        report['response']=response.model_dump()
        action=state.consume(response.model_dump())
        report.update(state=deepcopy(state.values),action=action.kind,
                      readback=reply_text(action,state.values,config['demo']))
        expected={'items':[dict(id=1,quantity=1,size='large',toppings=['cheese']),
                           dict(id=2,quantity=1,size='medium' if args.explicit or args.synthetic else 'small',toppings=['beef'])],'drink':['cola']}
        report['passed']=state.values==expected and action.kind=='readback' and not state.confirmed
    except Exception as exc:
        report.update(passed=False,error_type=type(exc).__name__,error=str(exc))
    finally:
        report['calls']=router.calls
        await router.close()
        await client.aclose()
        path=ROOT/('bench/results/multi_item_live_explicit_20260905.json' if args.explicit else 'bench/results/multi_item_live_replay_20260905.json')
        if args.synthetic: path=ROOT/'bench/results/multi_item_live_control_20260905.json'
        if path.exists():
            previous=json.loads(path.read_text(encoding='utf-8'))
            report['earlier_attempts']=previous.pop('earlier_attempts',[])+[previous]
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__': asyncio.run(main())
