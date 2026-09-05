"""Strict structural contract for one configurable row collection and roots."""

from copy import deepcopy


def validate_collection_schema(config):
    """Return an isolated transaction schema; validate before adapter allocation."""
    if not isinstance(config, dict) or not isinstance(config.get('demo'), dict):
        raise ValueError('A configured collection requires demo settings.')
    settings = config['demo']
    if settings.get('state_kind') != 'configured_collection_scoped':
        raise ValueError('Expected configured_collection_scoped task kind.')
    if settings.get('order_schema_version') is not None:
        raise ValueError('Configured collections cannot select a legacy order schema.')
    if type(settings.get('collection_min_items')) is not int or settings['collection_min_items'] != 1:
        raise ValueError('Configured collections currently require exactly one minimum row.')
    maximum = settings.get('collection_max_items')
    if type(maximum) is not int or not 1 <= maximum <= 10:
        raise ValueError('Configured collection maximum must be an integer from one to ten.')
    label = settings.get('collection_label')
    if not isinstance(label, str) or not label.strip():
        raise ValueError('Configured collections require a nonempty row label.')
    definitions = config.get('slots')
    if not isinstance(definitions, list) or not definitions:
        raise ValueError('Configured collections require selected slot definitions.')
    slots = {}
    for definition in definitions:
        if not isinstance(definition, dict):
            raise ValueError('Each configured slot must be an object.')
        name = definition.get('id')
        if (not isinstance(name, str) or not name or name != name.strip() or
                name == 'id' or name in slots):
            raise ValueError('Selected slot IDs must be distinct nonempty names other than id.')
        if (type(definition.get('required')) is not bool or
                type(definition.get('ask_order')) is not int or definition['ask_order'] < 0):
            raise ValueError('Configured slots require boolean required and nonnegative integer ask_order.')
        slots[name] = definition
    schema = settings.get('transaction_schema')
    if not isinstance(schema, dict) or set(schema) != {'collection', 'root_slots', 'item_slots', 'exclusive_values'}:
        raise ValueError('A collection schema requires collection, root_slots, item_slots and exclusive_values only.')
    name = schema['collection']
    if not isinstance(name, str) or not name or name != name.strip() or name == 'id' or name in slots:
        raise ValueError('The collection name must be distinct from configured fields and id.')
    roots, items = schema['root_slots'], schema['item_slots']
    if (not isinstance(roots, list) or not isinstance(items, list) or not items or
            any(not isinstance(key, str) for key in roots + items) or
            len(set(roots + items)) != len(roots + items) or set(roots + items) != set(slots)):
        raise ValueError('Root and row fields must partition selected fields exactly once.')
    if not any(slots[key]['required'] for key in items):
        raise ValueError('A configured collection needs at least one required row field.')
    exclusive = schema['exclusive_values']
    if not isinstance(exclusive, dict):
        raise ValueError('Exclusive values must be a mapping of configured root list fields.')
    for key, choices in exclusive.items():
        if (key not in roots or slots[key].get('type') != 'enum_list' or not isinstance(choices, list) or
                any(not isinstance(choice, str) or choice not in slots[key].get('values', []) for choice in choices) or
                len(set(choices)) != len(choices)):
            raise ValueError('Exclusive choices must be distinct configured enum-list root values.')
    return deepcopy(schema)
