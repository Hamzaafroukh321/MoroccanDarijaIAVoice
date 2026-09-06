"""Exact answers to one currently requested linked collection field.

This is a deliberately narrow configuration lookup, not a language parser or
ASR repair. Only whole canonical enum values/aliases and ASCII integer tokens
qualify. It never interprets a correction, fills a companion, or confirms a
task. The caller must still validate the response identity and atomically apply
the answer through the task adapter. No transcript or supplied state is changed.
"""

import re

from engine.collection_schema import validate_collection_schema
from engine.lexicon import AFFIRM_MARKERS, NEGATE_MARKERS
from engine.normalize import slot_value
from engine.state import validate_operation


def _positive_integer(value):
    return type(value) is int and value > 0


def _addresses(value, row_id, fields):
    if not isinstance(value, list):
        raise ValueError('Expected linked addresses.')
    result = []
    for address in value:
        if (not isinstance(address, dict) or set(address) != {'item_id', 'slot'} or
                not _positive_integer(address['item_id']) or address['item_id'] != row_id or
                not isinstance(address['slot'], str) or address['slot'] not in fields):
            raise ValueError('Invalid linked address.')
        pair = (row_id, address['slot'])
        if pair in result:
            raise ValueError('Duplicate linked address.')
        result.append(pair)
    return result


def _current_field(config, state):
    schema = validate_collection_schema(config)
    if not isinstance(state, dict) or state.get('collection') != schema['collection']:
        raise ValueError('Expected collection context.')
    question, proposal = state.get('pending_clarification'), state.get('pending_proposal')
    if not isinstance(question, dict) or not isinstance(proposal, dict):
        raise ValueError('Expected a pending linked question and proposal.')
    if question.get('kind') != 'ambiguous_value' or not _positive_integer(question.get('id')):
        raise ValueError('Expected an identified ambiguous value question.')
    ids = question.get('item_ids')
    if not isinstance(ids, list) or len(ids) != 1 or not _positive_integer(ids[0]):
        raise ValueError('Expected one row identity.')
    row_id, field = ids[0], question.get('slot')
    fields = schema['item_slots']
    if (not isinstance(field, str) or field not in fields or state.get('requested_slot') != field or
            not _positive_integer(state.get('requested_item_id')) or state['requested_item_id'] != row_id):
        raise ValueError('The requested scope must match the pending question.')
    version, base = state.get('version'), proposal.get('base_version')
    if type(version) is not int or type(base) is not int or version < 0 or base != version:
        raise ValueError('The pending proposal must be current.')
    full = _addresses(proposal.get('linked_addresses'), row_id, fields)
    answered = _addresses(proposal.get('answered_addresses'), row_id, fields)
    remaining = _addresses(proposal.get('remaining_addresses'), row_id, fields)
    if (not 2 <= len(full) <= min(40, len(fields)) or not remaining or
            remaining[0] != (row_id, field) or set(answered) & set(remaining) or
            set(answered) | set(remaining) != set(full) or
            answered != [address for address in full if address in answered] or
            remaining != [address for address in full if address in remaining]):
        raise ValueError('The pending linked group must have consistent explicit coverage.')
    companions = _addresses(question.get('linked_addresses'), row_id, fields)
    if companions != [address for address in full if address != (row_id, field)]:
        raise ValueError('The question must retain the same linked group.')
    preview = proposal.get('state')
    rows = preview.get(schema['collection']) if isinstance(preview, dict) else None
    if not isinstance(rows, list) or len(rows) > config['demo']['collection_max_items']:
        raise ValueError('Expected a bounded proposal preview.')
    row_ids = []
    for row in rows:
        if not isinstance(row, dict) or not _positive_integer(row.get('id')) or row['id'] in row_ids:
            raise ValueError('Expected unique preview row identities.')
        row_ids.append(row['id'])
    if row_id not in row_ids:
        raise ValueError('The requested row must exist in the proposal preview.')
    return row_id, field


def _enum_answer(slot, token):
    values, aliases = slot.get('values'), slot.get('aliases', {})
    if (not isinstance(values, list) or not values or
            any(not isinstance(value, str) or not value.strip() for value in values) or
            len(set(values)) != len(values) or not isinstance(aliases, dict)):
        raise ValueError('Expected configured enum values and aliases.')
    for key, entries in aliases.items():
        if (key not in values or not isinstance(entries, list) or
                any(not isinstance(entry, str) or not entry.strip() for entry in entries)):
            raise ValueError('Invalid configured alias mapping.')
    matches = {value for value in values
        if token in {entry.strip().casefold() for entry in [value, *aliases.get(value, [])]}}
    return next(iter(matches)) if len(matches) == 1 else None


def exact_linked_answer(config, state, transcript):
    """Return one normalized four-key set operation, or abstain without mutation."""
    if not isinstance(transcript, str):
        return None
    token = transcript.strip().casefold()
    reserved = {marker.strip().casefold() for marker in AFFIRM_MARKERS + NEGATE_MARKERS}
    if not token or token in reserved | {'yes', 'no'} or any(mark in token for mark in '?؟？'):
        return None
    try:
        row_id, field = _current_field(config, state)
        slots = {slot['id']: slot for slot in config['slots']}
        definition = slots[field]
        if definition.get('type') == 'enum':
            value = _enum_answer(definition, token)
            if value is None:
                return None
        elif definition.get('type') == 'integer' and re.fullmatch(r'[0-9]+', token):
            value = int(token)
        else:
            return None
        operation = {'op': 'set', 'item_id': row_id, 'slot': field, 'value': value}
        validate_operation(operation, slots, config)
        operation['value'] = slot_value(value, definition, config)
        return operation
    except (ValueError, TypeError, KeyError, OverflowError):
        return None
