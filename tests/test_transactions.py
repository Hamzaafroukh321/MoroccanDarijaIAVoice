"""Generic transactions tested independently of any dialogue or provider."""

from copy import deepcopy
import json

import pytest

from engine.config import ROOT
from engine.transactions import apply_transaction


def op(kind, collection=None, item_id=None, slot=None, value=None):
    return dict(op=kind, collection=collection, item_id=item_id, slot=slot, value=value)


@pytest.fixture
def schema():
    config = json.loads((ROOT / 'configs/clinic.json').read_text(encoding='utf-8'))
    row_slots = {
        'amount': {'id': 'amount', 'type': 'integer', 'min': 1, 'max': 20},
        'format': {'id': 'format', 'type': 'enum', 'values': ['small', 'large']},
        'extras': {'id': 'extras', 'type': 'enum_list', 'values': ['cheese', 'olive', 'beef']},
        'description': {'id': 'description', 'type': 'text'},
    }
    return dict(config=config, root_slots={
        'selection': {'id': 'selection', 'type': 'enum_list', 'values': ['none', 'cola', 'water']},
        'name': {'id': 'name', 'type': 'text'},
        'date': {'id': 'date', 'type': 'date'},
        'time': {'id': 'time', 'type': 'time'},
        'phone': {'id': 'phone', 'type': 'phone'},
        'party_count': {'id': 'party_count', 'type': 'integer', 'min': 1, 'max': 10},
        'category': {'id': 'category', 'type': 'enum', 'values': ['new', 'followup']},
    }, collections={'products': {'slots': row_slots, 'max_items': 3}},
        exclusive_values={'selection': ['none']})


def test_arbitrary_collection_names_keep_distinct_rows_and_inputs_immutable(schema):
    values, ids = {'products': []}, {'products': 1}
    operations = [op('create', 'products', 1), op('set', 'products', 1, 'format', 'large'),
                  op('set', 'products', 1, 'extras', ['cheese']), op('create', 'products', 2),
                  op('set', 'products', 2, 'format', 'small'), op('set', 'products', 2, 'extras', ['beef'])]
    before = deepcopy((values, ids, operations, schema))
    result, allocated = apply_transaction(values, ids, operations, **schema)
    assert result['products'] == [{'id': 1, 'format': 'large', 'extras': ['cheese']},
                                  {'id': 2, 'format': 'small', 'extras': ['beef']}]
    assert allocated == {'products': 3}
    assert (values, ids, operations, schema) == before
    result['products'][0]['extras'].append('olive')
    assert operations[2]['value'] == ['cheese'] and values == {'products': []}


def test_flat_appointment_fields_use_existing_normalization(schema):
    schema['collections'] = {}
    values, ids = apply_transaction({}, {}, [
        op('set', slot='name', value='  Example   Patient '),
        op('set', slot='date', value='15/09/2026'), op('set', slot='time', value='9h30'),
        op('set', slot='phone', value='+212612345678'), op('set', slot='party_count', value=2),
        op('set', slot='category', value='followup')], **schema)
    assert values == {'name': 'Example Patient', 'date': '2026-09-15', 'time': '09:30',
                      'phone': '0612345678', 'party_count': 2, 'category': 'followup'}
    assert ids == {}


def test_two_independent_collections_allocate_stable_ids(schema):
    schema['collections']['packages'] = deepcopy(schema['collections']['products'])
    values = {'products': [], 'packages': []}
    ids = {'products': 1, 'packages': 1}
    result, ids = apply_transaction(values, ids, [op('create', 'products', 1), op('create', 'packages', 1),
        op('create', 'products', 2), op('delete', 'products', 1)], **schema)
    assert result == {'products': [{'id': 2}], 'packages': [{'id': 1}]}
    assert ids == {'products': 3, 'packages': 2}
    result, ids = apply_transaction(result, ids, [op('create', 'products', 3)], **schema)
    assert [row['id'] for row in result['products']] == [2, 3]


def test_create_delete_same_batch_consumes_id_without_reusing_it(schema):
    result, ids = apply_transaction({}, {'products': 1}, [op('create', 'products', 1), op('delete', 'products', 1)], **schema)
    assert result == {'products': []} and ids == {'products': 2}
    with pytest.raises(ValueError, match='next unused'):
        apply_transaction(result, ids, [op('create', 'products', 1)], **schema)


def test_unknown_row_late_failure_rolls_back_allocators_and_root_changes(schema):
    values, ids = {'products': [], 'name': 'Original'}, {'products': 1}
    operations = [op('create', 'products', 1), op('set', slot='name', value='Changed'),
                  op('set', 'products', 8, 'format', 'small')]
    before = deepcopy((values, ids, operations))
    with pytest.raises(ValueError, match='Unknown row'):
        apply_transaction(values, ids, operations, **schema)
    assert (values, ids, operations) == before


def test_remove_unknown_list_preserves_absence_and_remove_last_known_yields_empty(schema):
    values, ids = {'products': [{'id': 1}]}, {'products': 2}
    unchanged, _ = apply_transaction(values, ids, [op('remove', 'products', 1, 'extras', 'cheese'),
        op('remove', slot='selection', value='cola')], **schema)
    assert unchanged == values and 'selection' not in unchanged
    populated, _ = apply_transaction(values, ids, [op('set', 'products', 1, 'extras', ['cheese', 'cheese'])], **schema)
    assert populated['products'][0]['extras'] == ['cheese']
    emptied, _ = apply_transaction(populated, ids, [op('remove', 'products', 1, 'extras', 'cheese')], **schema)
    assert emptied['products'][0]['extras'] == []
    cleared, _ = apply_transaction(emptied, ids, [op('clear', 'products', 1, 'extras')], **schema)
    assert 'extras' not in cleared['products'][0]


def test_enum_list_add_dedupe_and_scalar_remove_semantics(schema):
    values, ids = {'products': [{'id': 1, 'format': 'large', 'extras': ['cheese']}]}, {'products': 2}
    result, _ = apply_transaction(values, ids, [op('add', 'products', 1, 'extras', ['cheese', 'olive']),
        op('remove', 'products', 1, 'format', 'small')], **schema)
    assert result['products'][0] == {'id': 1, 'format': 'large', 'extras': ['cheese', 'olive']}
    result, _ = apply_transaction(result, ids, [op('remove', 'products', 1, 'format', 'large'),
        op('remove', 'products', 1, 'extras')], **schema)
    assert result['products'][0] == {'id': 1}


def test_exclusive_root_choice_checked_atomically_on_final_state(schema):
    values, ids = {'products': [], 'selection': ['none']}, {'products': 1}
    with pytest.raises(ValueError, match='exclusive'):
        apply_transaction(values, ids, [op('create', 'products', 1), op('add', slot='selection', value='cola')], **schema)
    assert values == {'products': [], 'selection': ['none']} and ids == {'products': 1}
    result, _ = apply_transaction(values, ids, [op('add', slot='selection', value='cola'),
        op('remove', slot='selection', value='none')], **schema)
    assert result['selection'] == ['cola']


@pytest.mark.parametrize('bad', [op('create', 'unknown', 1), op('create', 'products', True),
    op('create', 'products', 0), op('create', 'products', 2), op('delete', 'products', 1),
    op('set', item_id=1, slot='name', value='Example'), op('set', slot='unknown', value='Example'),
    op('set', slot='party_count', value=True), op('set', slot='category', value='unsupported'),
    op('set', slot='date', value='not a date'), op('create', 'products', 1, 'amount', 1),
    {'op': 'create', 'collection': 'products', 'item_id': 1, 'slot': None},
    {**op('create', 'products', 1), 'extra': 'forbidden'}])
def test_invalid_operations_fail_without_mutation(schema, bad):
    values, ids = {'products': []}, {'products': 1}
    with pytest.raises(ValueError):
        apply_transaction(values, ids, [bad], **schema)
    assert values == {'products': []} and ids == {'products': 1}


def test_collection_limit_and_reserved_id_field(schema):
    schema['collections']['products']['max_items'] = 1
    with pytest.raises(ValueError, match='limit'):
        apply_transaction({}, {'products': 1}, [op('create', 'products', 1), op('create', 'products', 2)], **schema)
    schema['collections']['products']['slots']['id'] = {'id': 'id', 'type': 'integer'}
    with pytest.raises(ValueError, match='reserved'):
        apply_transaction({}, {'products': 1}, [], **schema)


def test_row_field_id_is_never_editable_and_rows_cannot_cross_collections(schema):
    schema['collections']['packages'] = deepcopy(schema['collections']['products'])
    values = {'products': [{'id': 1}], 'packages': []}
    ids = {'products': 2, 'packages': 1}
    for bad in (op('set', 'products', 1, 'id', 2), op('set', 'packages', 1, 'format', 'large')):
        with pytest.raises(ValueError):
            apply_transaction(values, ids, [bad], **schema)
    assert values == {'products': [{'id': 1}], 'packages': []}
