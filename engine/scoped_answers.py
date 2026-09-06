"""Exact answers to one currently requested linked flat or collection field.

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


def _addresses(value, roots, fields, row_ids):
    if not isinstance(value, list):
        raise ValueError('Expected linked addresses.')
    result = []
    for address in value:
        if (not isinstance(address, dict) or set(address) != {'item_id', 'slot'} or
                not isinstance(address['slot'], str) or address['slot'] not in roots + fields):
            raise ValueError('Invalid linked address.')
        target, field = address['item_id'], address['slot']
        if ((field in roots and target is not None) or
                (field in fields and (not _positive_integer(target) or target not in row_ids))):
            raise ValueError('Linked addresses must match configured root or existing row scope.')
        pair = (target, field)
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
    roots, fields = schema['root_slots'], schema['item_slots']
    field = question.get('slot')
    if (not isinstance(ids, list) or not isinstance(field, str) or field not in roots + fields or
            (field in roots and ids != []) or
            (field in fields and (len(ids) != 1 or not _positive_integer(ids[0])))):
        raise ValueError('Expected one configured root or row-field scope.')
    row_id = ids[0] if ids else None
    requested_id = state.get('requested_item_id')
    if (state.get('requested_slot') != field or requested_id != row_id or
            (row_id is not None and not _positive_integer(requested_id))):
        raise ValueError('The requested scope must match the pending question.')
    version, base = state.get('version'), proposal.get('base_version')
    if type(version) is not int or type(base) is not int or version < 0 or base != version:
        raise ValueError('The pending proposal must be current.')
    preview = proposal.get('state')
    rows = preview.get(schema['collection']) if isinstance(preview, dict) else None
    if not isinstance(rows, list) or len(rows) > config['demo']['collection_max_items']:
        raise ValueError('Expected a bounded proposal preview.')
    row_ids = []
    for row in rows:
        if not isinstance(row, dict) or not _positive_integer(row.get('id')) or row['id'] in row_ids:
            raise ValueError('Expected unique preview row identities.')
        row_ids.append(row['id'])
    if row_id is not None and row_id not in row_ids:
        raise ValueError('The requested row must exist in the proposal preview.')
    full = _addresses(proposal.get('linked_addresses'), roots, fields, row_ids)
    answered = _addresses(proposal.get('answered_addresses'), roots, fields, row_ids)
    remaining = _addresses(proposal.get('remaining_addresses'), roots, fields, row_ids)
    maximum = len(roots) + config['demo']['collection_max_items'] * len(fields)
    if (not 2 <= len(full) <= min(40, maximum) or not remaining or
            remaining[0] != (row_id, field) or set(answered) & set(remaining) or
            set(answered) | set(remaining) != set(full) or
            answered != [address for address in full if address in answered] or
            remaining != [address for address in full if address in remaining]):
        raise ValueError('The pending linked group must have consistent explicit coverage.')
    companions = _addresses(question.get('linked_addresses'), roots, fields, row_ids)
    if companions != [address for address in full if address != (row_id, field)]:
        raise ValueError('The question must retain the same linked group.')
    return row_id, field


def _flat_current_field(config, state):
    settings = config['demo']
    schema = settings.get('transaction_schema')
    definitions = config.get('slots')
    if (not isinstance(schema, dict) or
            set(schema) != {'collection', 'root_slots', 'item_slots', 'exclusive_values'} or
            schema['collection'] is not None or schema['item_slots'] != [] or
            not isinstance(schema['exclusive_values'], dict) or
            not isinstance(definitions, list) or not definitions):
        raise ValueError('Expected a configured flat transaction schema.')
    selected = []
    for definition in definitions:
        if not isinstance(definition, dict):
            raise ValueError('Expected configured field definitions.')
        name = definition.get('id')
        if not isinstance(name, str) or not name or name != name.strip() or name in selected:
            raise ValueError('Expected unique configured field names.')
        selected.append(name)
    fields = schema['root_slots']
    if (not isinstance(fields, list) or any(not isinstance(field, str) for field in fields) or
            len(fields) != len(selected) or set(fields) != set(selected)):
        raise ValueError('All configured fields must be distinct roots.')
    if not isinstance(state, dict) or state.get('collection') is not None or state.get('requested_item_id') is not None:
        raise ValueError('Expected root-field context.')
    question, proposal = state.get('pending_clarification'), state.get('pending_proposal')
    if (not isinstance(question, dict) or not isinstance(proposal, dict) or
            question.get('kind') != 'ambiguous_value' or not _positive_integer(question.get('id')) or
            question.get('item_ids') != []):
        raise ValueError('Expected an identified root-field ambiguous question.')
    field = question.get('slot')
    if not isinstance(field, str) or field not in fields or state.get('requested_slot') != field:
        raise ValueError('The requested root must match the pending question.')
    version, base = state.get('version'), proposal.get('base_version')
    if type(version) is not int or type(base) is not int or version < 0 or base != version:
        raise ValueError('The pending proposal must be current.')
    full, answered, remaining, companions = (proposal.get('coupled_slots'), proposal.get('answered_slots'),
        proposal.get('remaining_slots'), question.get('coupled_slots'))
    for group in (full, answered, remaining, companions):
        if (not isinstance(group, list) or
                any(not isinstance(name, str) or name not in fields for name in group) or
                len(set(group)) != len(group)):
            raise ValueError('Expected distinct configured coupled fields.')
    if (not 2 <= len(full) <= min(40, len(fields)) or not remaining or remaining[0] != field or
            set(answered) & set(remaining) or set(answered) | set(remaining) != set(full) or
            answered != [name for name in full if name in answered] or
            remaining != [name for name in full if name in remaining] or
            companions != [name for name in full if name != field] or
            not isinstance(proposal.get('state'), dict)):
        raise ValueError('The pending coupled group must have consistent explicit coverage.')
    return field


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
    """Return one set operation (three flat/four collection keys), or abstain."""
    if not isinstance(transcript, str):
        return None
    token = transcript.strip().casefold()
    reserved = {marker.strip().casefold() for marker in AFFIRM_MARKERS + NEGATE_MARKERS}
    if not token or token in reserved | {'yes', 'no'} or any(mark in token for mark in '?؟？'):
        return None
    try:
        if not isinstance(config, dict) or not isinstance(config.get('demo'), dict):
            return None
        kind = config['demo'].get('state_kind')
        if kind == 'configured_collection_scoped':
            row_id, field = _current_field(config, state)
        elif kind == 'flat_scoped':
            field = _flat_current_field(config, state)
        else:
            return None
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
        operation = {'op': 'set', 'slot': field, 'value': value}
        if kind == 'configured_collection_scoped':
            operation['item_id'] = row_id
        validate_operation(operation, slots, config)
        operation['value'] = slot_value(value, definition, config)
        return operation
    except (ValueError, TypeError, KeyError, OverflowError):
        return None
