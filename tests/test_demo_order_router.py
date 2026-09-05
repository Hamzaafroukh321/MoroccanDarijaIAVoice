import json

import pytest

from engine.config import ROOT, load_config
from engine.demo import demo_config, reply_text
from engine.demo_order import DemoOrderState
from engine.router import build_messages, build_prompt, parse_response, response_format
from engine.state import Action


@pytest.fixture
def config():
    return demo_config(load_config(ROOT/'configs/pizza.json'))


def payload(ops, intent='task'):
    return json.dumps(dict(ops=ops, intent=intent, confidence=.9, unclear=False,
                          is_affirmation=False, is_negation=False,clarification=None,resolves_clarification=None,proposed_ops=[],discard_clarification=None))


def test_strict_schema_requires_explicit_item_address(config):
    schema=response_format(config)['json_schema']['schema']
    operation=schema['$defs']['DemoItemOperation']
    assert set(operation['required'])=={'op','item_id','slot','value'}
    assert operation['additionalProperties'] is False
    assert 'ambiguous' in schema['properties']['intent']['enum']
    with pytest.raises(ValueError):
        parse_response(payload([dict(op='set',slot='size',value='large')]),config)


@pytest.mark.parametrize('operation',[
    dict(op='create',item_id=1,slot='size',value='large'),
    dict(op='delete',item_id=None,slot=None,value=None),
    dict(op='set',item_id=None,slot='size',value='large'),
    dict(op='set',item_id=1,slot='drink',value=['cola']),
    dict(op='set',item_id=1,slot='size',value='enormous'),
    dict(op='set',item_id=None,slot='drink',value=['none','cola']),
    dict(op='clear',item_id=1,slot='toppings',value=[]),
])
def test_invalid_address_or_value_never_reaches_state(config,operation):
    with pytest.raises(ValueError): parse_response(payload([operation]),config)


def test_schema_and_state_keep_separate_items_and_ordinals(config):
    ops=[]
    for item_id,size,toppings in [(1,'large',['cheese']),(2,'medium',['beef'])]:
        ops.append(dict(op='create',item_id=item_id,slot=None,value=None))
        ops.extend(dict(op='set',item_id=item_id,slot=slot,value=value)
                   for slot,value in dict(quantity=1,size=size,toppings=toppings).items())
    ops.append(dict(op='set',item_id=None,slot='drink',value=['none']))
    state=DemoOrderState(config)
    response=parse_response(payload(ops),config)
    assert state.consume(response.model_dump()).kind=='readback'
    text=reply_text(Action('readback'),state.values,config['demo'])
    assert text.index('بيتزا 1') < text.index('كبيرة') < text.index('بيتزا 2') < text.index('متوسطة')
    assert text.count(config['demo']['labels']['drink'])==1
    state.apply([dict(op='delete',item_id=1,slot=None,value=None)])
    assert reply_text(Action('ask','size',2),state.values,config['demo']).startswith('بيتزا 1.')
    with pytest.raises(ValueError): parse_response(payload(ops,'ambiguous'),config)
    prompt=build_prompt(config,state.router_context(),'fixture')
    assert 'identical pizzas only' not in prompt
    assert 'requested_item_id' in prompt and 'next_item_id' in prompt


def test_plain_pizza_readback_is_explicit(config):
    values={'items':[dict(id=3,quantity=1,size='small',toppings=[])],'drink':['none']}
    text=reply_text(Action('readback'),values,config['demo'])
    assert config['demo']['plain_toppings'] in text
    assert 'بيتزا 1' in text and 'بيتزا 3' not in text


def test_ambiguous_new_order_does_not_ask_to_change_nonexistent_item(config):
    settings=config['demo']
    assert reply_text(Action('ambiguous'),{'items':[]},settings)==settings['responses']['ambiguous_new_order']
    assert reply_text(Action('ambiguous'),{'items':[{'id':1}]},settings)==settings['responses']['ambiguous']


def test_demo_router_budget_does_not_change_research(config):
    research=load_config(ROOT/'configs/pizza.json')
    assert research['router']['max_tokens']==1024
    assert config['router']['max_tokens']==2048
    assert 'reasoning_effort' not in research['router']


@pytest.mark.parametrize('intent,kind,slot',[
    ('ambiguous','item_reference','size'),('ambiguous','order_details',None),
    ('out_of_scope','drink_size','drink'),('out_of_scope','unsupported_drink','drink'),
    ('out_of_scope','unsupported_menu','toppings'),('unclear','unintelligible',None)])
def test_clarification_contract(config,intent,kind,slot):
    data=json.loads(payload([],intent))
    data['clarification']={'kind':kind,'slot':slot,'item_ids':[]}
    assert parse_response(json.dumps(data),config).clarification.kind==kind
    data['resolves_clarification']=1
    with pytest.raises(ValueError): parse_response(json.dumps(data),config)


def test_drink_scope_and_required_fields_cannot_be_omitted(config):
    schema=response_format(config)['json_schema']['schema']
    assert {'clarification','resolves_clarification'} <= set(schema['required'])
    data=json.loads(payload([],'out_of_scope'))
    data['clarification']={'kind':'drink_size','slot':'toppings','item_ids':[1]}
    with pytest.raises(ValueError): parse_response(json.dumps(data),config)


def test_supported_menu_options_are_spoken_for_target_field(config):
    text=reply_text(Action('unsupported_menu','toppings',1),{'items':[{'id':1}]},config['demo'])
    assert config['demo']['values']['cheese'] in text
    assert config['demo']['values']['cola'] not in text


def test_repair_history_uses_bounded_dialogue_roles_without_mutating_context(config):
    context={'state':{'items':[]},'pending_request':{'text':'x'*1100,'truncated':True},'last_assistant_text':'a'*500}
    messages=build_messages(config,context,'cola')
    assert [message['role'] for message in messages]==['system','user','assistant','user']
    assert len(messages[1]['content'])==1000 and len(messages[2]['content'])==400
    assert messages[3]['content']=='cola'
    assert len(context['pending_request']['text'])==1100
    assert 'x'*1000 not in messages[0]['content']
    assert len(build_messages(config,{'state':{'items':[]}},'hello'))==2


def test_specific_clarification_only_downgrades_contradictory_task_label(config):
    data=json.loads(payload([]))
    data['clarification']={'kind':'drink_size','slot':'drink','item_ids':[]}
    data['proposed_ops']=[dict(op='create',item_id=1,slot=None,value=None),dict(op='set',item_id=1,slot='size',value='small')]
    parsed=parse_response(json.dumps(data),config)
    assert parsed.intent=='out_of_scope' and not parsed.ops
    assert parsed._clarification_intent_normalized
    state=DemoOrderState(config)
    assert state.consume(parsed.model_dump()).kind=='drink_size'
    assert state.values=={'items':[]} and state.version==0
    assert state.pending_proposal['state']['items'][0]['size']=='small'
    for key,value in [('ops',[dict(op='set',item_id=None,slot='drink',value=['cola'])]),
                      ('is_affirmation',True),('is_negation',True),('resolves_clarification',1),('discard_clarification',1)]:
        bad={**data,key:value}
        with pytest.raises(ValueError): parse_response(json.dumps(bad),config)
    data['proposed_ops'][1]['value']='enormous'
    with pytest.raises(ValueError): parse_response(json.dumps(data),config)


def test_draft_preview_drives_router_ids_without_changing_saved_state(config):
    committed={'items':[]}
    preview={'items':[dict(id=1,size='large')]}
    context={'state':committed,'next_item_id':1,'pending_clarification':{'id':1,'kind':'drink_size','slot':'drink','item_ids':[]},
             'pending_proposal':{'state':preview,'next_item_id':2,'ops':[],'base_version':0}}
    message=build_messages(config,context,'cola')[0]['content']
    encoded=message.split('CONTEXT: ',1)[1].split('\nFIRST ',1)[0]
    effective=json.loads(encoded)
    assert effective['state']==preview and effective['committed_state']==committed
    assert effective['next_item_id']==2 and effective['requested_slot']=='drink'
    assert context['state']=={'items':[]} and context['next_item_id']==1
