"""Offline speech-mapping invariants for merged synthetic task previews."""

from engine.config import ConfigError


class DemoConfigError(ConfigError):
    """A local preview configuration issue, before clients or audio are allocated."""


def _text(value, path):
    if not isinstance(value, str) or not value.strip():
        raise DemoConfigError(f'{path} must contain nonempty text.')


def _mapping(settings, key):
    value = settings.get(key)
    if not isinstance(value, dict):
        raise DemoConfigError(f'demo.{key} must be an object.')
    for name, text in value.items():
        _text(text, f'demo.{key}.{name}')
    return value


def validate_demo_config(config):
    """Require every selectable field to be askable and included in readback.

    This validates rendering contracts, not pronunciation, provider availability,
    or research-bank readiness. The base domain schema remains authoritative.
    """
    settings = config['demo']
    mode = settings.get('tts_readback_mode', 'whole')
    if mode not in ('whole', 'grouped'):
        raise DemoConfigError('demo.tts_readback_mode must be whole or grouped.')
    if 'tts_readback_group_chars' in settings:
        maximum = settings['tts_readback_group_chars']
        provider_maximum = settings.get('tts_max_chars')
        if (type(maximum) is not int or type(provider_maximum) is not int or
                not 32 <= maximum <= provider_maximum):
            raise DemoConfigError('demo.tts_readback_group_chars must be an integer from 32 to tts_max_chars.')
    ui = _mapping(settings, 'ui')
    for key in ('title', 'note', 'state_title', 'state_empty', 'completion_text'):
        if key not in ui:
            raise DemoConfigError(f'demo.ui.{key} is required for the task preview.')
    slots = {slot['id']: slot for slot in config['slots']}
    kind = settings.get('state_kind')
    if kind not in {'flat_scoped', 'collection_scoped'}:
        raise DemoConfigError('demo.state_kind must be flat_scoped or collection_scoped.')
    schema = settings.get('transaction_schema')
    if not isinstance(schema, dict):
        raise DemoConfigError('demo.transaction_schema must be an object.')
    root_fields, item_fields = schema.get('root_slots'), schema.get('item_slots')
    if (not isinstance(root_fields, list) or not isinstance(item_fields, list) or
            any(not isinstance(key, str) for key in root_fields + item_fields) or
            len(root_fields + item_fields) != len(slots) or
            set(root_fields + item_fields) != set(slots)):
        raise DemoConfigError('demo.transaction_schema fields must match every selected field exactly once; check demo.questions and fields.')
    if kind == 'flat_scoped' and (schema.get('collection') is not None or item_fields):
        raise DemoConfigError('A flat_scoped preview requires root fields and a null collection.')
    questions = _mapping(settings, 'questions')
    labels = _mapping(settings, 'labels')
    values = _mapping(settings, 'values')
    responses = _mapping(settings, 'responses')
    for key in slots:
        for name, mapping in [('questions', questions), ('labels', labels)]:
            if key not in mapping:
                raise DemoConfigError(f'demo.{name}.{key} is required for the selected field.')
    required = settings.get('required_slots')
    if (not isinstance(required, list) or
            any(not isinstance(key, str) or key not in slots for key in required) or
            len(required) != len(set(required))):
        raise DemoConfigError('demo.required_slots must list unique selected field IDs.')
    for key in ['readback_prefix', 'readback_question']:
        _text(settings.get(key), f'demo.{key}')
    response_keys = {'greeting', 'accepted', 'handoff', 'listen', 'not_understood',
                     'clarify_overlap', 'unsupported_option', 'unintelligible'}
    if kind == 'flat_scoped':
        response_keys.update({'unsupported_value', 'ambiguous_value'})
        order = settings.get('readback_order', list(labels))
        if (not isinstance(order, list) or
                any(not isinstance(key, str) for key in order) or
                len(order) != len(slots) or set(order) != set(slots)):
            raise DemoConfigError('demo.readback_order must include every selected field exactly once.')
        formats = settings.get('readback_formats', {})
        if not isinstance(formats, dict):
            raise DemoConfigError('demo.readback_formats must be an object.')
        for key, formatter in formats.items():
            expected = {'day_month_year': 'date', '24_hour': 'time'}.get(formatter) if isinstance(formatter, str) else None
            if key not in slots or expected is None or slots[key]['type'] != expected:
                raise DemoConfigError(f'demo.readback_formats.{key} must use a supported format matching its field type.')
    else:
        response_keys.update({'order_details', 'drink_size', 'unsupported_drink',
                             'unsupported_menu', 'ambiguous', 'ambiguous_new_order'})
        for key in ['item_label', 'plain_toppings']:
            _text(settings.get(key), f'demo.{key}')
        # The existing pizza adapter renders drink choices by direct lookup.
        for choice in slots.get('drink', {}).get('values', []):
            if choice not in values:
                raise DemoConfigError(f'demo.values.{choice} is required for drink readback.')
    for key in sorted(response_keys):
        if key not in responses:
            raise DemoConfigError(f'demo.responses.{key} is required.')
