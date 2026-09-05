"""Domain-independent atomic field and stable-row transactions."""

from copy import deepcopy

from engine.normalize import slot_value
from engine.state import validate_operation


def _field_operation(target, operation, slots, config):
    kind, key, value = (operation[name] for name in ('op', 'slot', 'value'))
    validate_operation({'op': kind, 'slot': key, 'value': value}, slots, config)
    slot = slots[key]
    if kind == 'clear' or (kind == 'remove' and value is None):
        target.pop(key, None)
    elif kind == 'set':
        target[key] = slot_value(value, slot, config)
    elif slot['type'] == 'enum_list':
        incoming = slot_value([value] if isinstance(value, str) else value, slot, config)
        if kind == 'add':
            target[key] = list(dict.fromkeys(target.get(key, []) + incoming))
        elif key in target:
            # An absent list is unknown, unlike an explicit empty list.
            target[key] = [item for item in target[key] if item not in incoming]
    elif kind == 'remove' and target.get(key) == slot_value(value, slot, config):
        target.pop(key, None)


def apply_transaction(values, next_ids, operations, *, root_slots, collections, config, exclusive_values=None):
    """Return new values and ID allocators after validating the whole batch.

    Collection IDs never get reused, even if a row is created and deleted in
    the same transaction. The caller's values, allocators, operations and slot
    definitions are never mutated, on success or on failure.
    """
    candidate = deepcopy(values)
    allocated = deepcopy(next_ids)
    for name, definition in collections.items():
        slots = definition['slots']
        if 'id' in slots or any(slot.get('id') == 'id' for slot in slots.values()):
            raise ValueError('The id field is reserved for stable row identity.')
        if name in root_slots:
            raise ValueError('Root fields and collections must have distinct names.')
        maximum = definition['max_items']
        if type(maximum) is not int or maximum < 1:
            raise ValueError('Collection max_items must be a positive integer.')
        next_id = allocated.get(name)
        if type(next_id) is not int or next_id < 1:
            raise ValueError('Each collection requires a positive next ID.')
        rows = candidate.get(name, [])
        if not isinstance(rows, list) or len(rows) > maximum:
            raise ValueError('Invalid collection rows or collection limit exceeded.')
        identifiers = []
        for row in rows:
            if not isinstance(row, dict) or type(row.get('id')) is not int or not 0 < row['id'] < next_id:
                raise ValueError('Existing rows require positive IDs below the next unused ID.')
            identifiers.append(row['id'])
        if len(set(identifiers)) != len(identifiers):
            raise ValueError('Collection row IDs must be unique.')

    for operation in operations:
        if not isinstance(operation, dict) or set(operation) != {'op', 'collection', 'item_id', 'slot', 'value'}:
            raise ValueError('An operation requires op, collection, item_id, slot and value only.')
        kind, collection, item_id, key, value = (operation[name] for name in ('op', 'collection', 'item_id', 'slot', 'value'))
        if item_id is not None and (type(item_id) is not int or item_id < 1):
            raise ValueError('Row IDs must be positive integers.')
        if collection is None:
            if item_id is not None:
                raise ValueError('Root field operations require a null item ID.')
            if not isinstance(key, str) or key not in root_slots:
                raise ValueError('Unknown root field.')
            _field_operation(candidate, operation, root_slots, config)
            continue
        if not isinstance(collection, str) or collection not in collections:
            raise ValueError('Unknown collection.')
        definition = collections[collection]
        rows = candidate.get(collection, [])
        if kind == 'create':
            if item_id != allocated[collection] or key is not None or value is not None:
                raise ValueError('Create requires the next unused ID and null slot/value.')
            if len(rows) >= definition['max_items']:
                raise ValueError('The collection has reached its row limit.')
            candidate.setdefault(collection, []).append({'id': item_id})
            allocated[collection] += 1
            continue
        target = next((row for row in rows if row['id'] == item_id), None)
        if target is None:
            raise ValueError('Unknown row ID.')
        if kind == 'delete':
            if key is not None or value is not None:
                raise ValueError('Delete requires null slot/value.')
            candidate[collection] = [row for row in rows if row['id'] != item_id]
            continue
        if not isinstance(key, str) or key not in definition['slots']:
            raise ValueError('Unknown collection field.')
        _field_operation(target, operation, definition['slots'], config)

    for field, exclusive in (exclusive_values or {}).items():
        choices = candidate.get(field)
        if isinstance(choices, list) and len(choices) > 1 and any(choice in exclusive for choice in choices):
            raise ValueError('An exclusive root choice cannot be combined with another choice.')
    return candidate, allocated
