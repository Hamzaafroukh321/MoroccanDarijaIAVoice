"""Temporary English synthetic collection profiles; never a shipped business task."""
from copy import deepcopy
import json

from engine.config import ROOT, load_config
from engine.demo import demo_config


def collection_documents(*, renamed=False):
    names = (dict(domain='kit-hire', collection='loans', asset='resource', quantity='units', date='due_date')
             if renamed else dict(domain='equipment', collection='reservations', asset='asset', quantity='quantity', date='return_date'))
    base = json.loads((ROOT / 'configs/clinic.json').read_text(encoding='utf-8'))
    base.update(domain_id=names['domain'], display_name='Equipment reservation diagnostic', needs_human_review=True)
    base['slots'] = []
    for index, (key, kind, label) in enumerate([
        (names['asset'], 'enum', 'Equipment'), (names['quantity'], 'integer', 'Quantity'),
        (names['date'], 'date', 'Return date')], 1):
        slot = dict(id=key, type=kind, required=True, critical=True, ask_order=index,
            prompt_audio=f'ask_{key}.wav', prompt_text_ary=f'English diagnostic: choose {label}.',
            readback_audio=f'rb_{key}.wav', readback_text_ary=f'English diagnostic: {label}.')
        if kind == 'enum':
            slot.update(values=['camera', 'tripod'], aliases={'camera': ['camera'], 'tripod': ['tripod']})
        if kind == 'integer':
            slot.update(min=1, max=10)
        base['slots'].append(slot)
    profile = json.loads((ROOT / 'configs/demo/clinic.json').read_text(encoding='utf-8'))
    fields = [names[key] for key in ('asset', 'quantity', 'date')]
    profile.update(needs_human_review=True, language_review='English synthetic diagnostic; unreviewed, not a language accuracy evaluation.',
        label='Equipment reservation diagnostic', state_kind='configured_collection_scoped',
        profile_kind='fictional_diagnostic', evaluation_eligible=False,
        task_scope='fictional equipment reservation preferences; no inventory check or actual reservation',
        fields=fields, required_slots=fields, slot_overrides={},
        transaction_schema=dict(collection=names['collection'], root_slots=[names['date']],
            item_slots=[names['asset'], names['quantity']], exclusive_values={}),
        collection_min_items=1, collection_max_items=10, collection_label='Reservation',
        questions={names['asset']: 'Choose camera or tripod.', names['quantity']: 'How many?', names['date']: 'Which return date?'},
        labels={names['asset']: 'Equipment', names['quantity']: 'Quantity', names['date']: 'Return date'},
        values={'camera': 'Camera', 'tripod': 'Tripod'}, readback_order=fields,
        readback_formats={names['date']: 'day_month_year'},
        readback_prefix='Your diagnostic preferences:', readback_question='Are these correct?')
    profile['responses'] = {key: f'English diagnostic response: {key}.' for key in profile['responses']}
    profile['responses'].update(item_reference='Which reservation?', ambiguous='Which reservation?',
        ambiguous_value='Which option?', unsupported_value='That option is unavailable in this diagnostic.')
    profile['ui'] = dict(title='Equipment preference diagnostic', note='Fictional English synthetic preview.',
        state_title='Preferences', state_empty='No preferences yet.',
        completion_text='Preferences confirmed. No reservation was placed.')
    return base, profile, names


def collection_config(*, renamed=False):
    """Build merged state-test input while profile discovery is integrated separately."""
    base, profile, names = collection_documents(renamed=renamed)
    config = demo_config(load_config(ROOT / 'configs/clinic.json'))
    config.update(domain_id=base['domain_id'], display_name=base['display_name'], slots=deepcopy(base['slots']))
    config['demo'].update(deepcopy(profile))
    config['demo'].pop('order_schema_version', None)
    config['demo'].pop('order_max_items', None)
    return config, names


def write_collection_fixture(directory, *, renamed=False):
    """Write complete base/profile documents below the caller's temporary root."""
    base, profile, names = collection_documents(renamed=renamed)
    destination = directory / 'configs/demo'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'voice.json').write_bytes((ROOT / 'configs/demo/voice.json').read_bytes())
    config_path = directory / 'configs' / (names['domain'] + '.json')
    profile_path = destination / (names['domain'] + '.json')
    config_path.write_text(json.dumps(base), encoding='utf-8')
    profile_path.write_text(json.dumps(profile), encoding='utf-8')
    return dict(root=directory, config_path=config_path, profile_path=profile_path,
                base=base, profile=profile, names=names)
