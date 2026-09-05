"""Atomic, explicitly addressed pizza items for the synthetic demo only."""

from copy import deepcopy

from engine.state import Action
from engine.confirmation import VersionedConfirmation
from engine.dialogue import ScopedDialogue


class DemoOrderState(ScopedDialogue, VersionedConfirmation):
    def __init__(self, config):
        self.config = config
        schema = config['demo']['transaction_schema']
        if (schema['collection'] != 'items' or set(schema['root_slots']) != {'drink'} or
                set(schema['item_slots']) != {'quantity', 'size', 'toppings'}):
            raise ValueError('The pizza dialogue adapter requires items with quantity, size, toppings and a root drink field.')
        self.slots = {slot['id']: slot for slot in config['slots']}
        self.values = {'items': []}
        self.version = 0
        self.readback_version = None
        self.confirmed_version = None
        self.repair_count = 0
        self.awaiting_correction = False
        self.ambiguity_pending = False
        self._ambiguous_item_ids = frozenset()
        self.next_item_id = 1
        self.pending_clarification = None
        self.pending_proposal = None
        self._next_clarification_id = 1

    def _missing(self):
        if not self.values['items']:
            return ('size', self.next_item_id)
        fields = sorted(('size', 'quantity', 'toppings'), key=lambda key: self.slots[key]['ask_order'])
        for item in self.values['items']:
            for key in fields:
                # Explicit [] is a plain pizza; an absent toppings field is unknown.
                if key not in item or item[key] is None:
                    return (key, item['id'])
        if self.values.get('drink') in (None, []):
            return ('drink', None)
        return None

    @property
    def ready(self):
        return self._missing() is None

    @property
    def confirmation_blocked(self):
        return self.pending_clarification is not None or self.ambiguity_pending

    def apply(self, operations, *, _resolve_proposal=False):
        """Validate the entire batch, including IDs, before committing any change."""
        if self.pending_proposal is not None and operations and not _resolve_proposal:
            raise ValueError('Resolve or discard the staged proposal before changing the order.')
        from engine.transactions import apply_transaction
        schema = self.config['demo']['transaction_schema']
        collection = schema['collection']
        mapped = []
        for operation in operations:
            if set(operation) != {'op', 'item_id', 'slot', 'value'}:
                raise ValueError('An order operation requires op, item_id, slot and value only.')
            mapped.append({**operation, 'collection': None if operation['slot'] in schema['root_slots'] else collection})
        candidate, next_ids = apply_transaction(
            self.values, {collection: self.next_item_id}, mapped,
            root_slots={key: self.slots[key] for key in schema['root_slots']},
            collections={collection: {'slots': {key: self.slots[key] for key in schema['item_slots']},
                                      'max_items': self.config['demo']['order_max_items']}},
            config=self.config, exclusive_values=schema['exclusive_values'])
        next_id = next_ids[collection]
        changed_values = candidate != self.values
        resolves_ambiguity = not self.ambiguity_pending
        if self.ambiguity_pending and changed_values:
            before_items = {item['id']: item for item in self.values['items']}
            after_items = {item['id']: item for item in candidate['items']}
            resolves_ambiguity = (bool(after_items) if not self._ambiguous_item_ids else
                any(before_items.get(item_id) != after_items.get(item_id)
                    for item_id in self._ambiguous_item_ids))
        if self.pending_clarification is not None:
            # A changed value is not proof that a scoped question was answered.
            resolves_ambiguity = False
        if changed_values or next_id != self.next_item_id:
            self.values = candidate
            self.next_item_id = next_id
            self.version += 1
            self.invalidate_confirmation()
            if changed_values and resolves_ambiguity:
                self.awaiting_correction = False
                self.ambiguity_pending = False
                self._ambiguous_item_ids = frozenset()
                self.repair_count = 0

    def _repair(self, *, count=True):
        self.invalidate_confirmation()
        if count:
            self.repair_count += 1
        if self.repair_count > self.config['demo']['max_repairs']:
            return Action('handoff')
        if self.pending_clarification is not None:
            return self._pending_action()
        if self.ambiguity_pending:
            return Action('ambiguous')
        missing = self._missing()
        return Action('repair', *missing) if missing else Action('repair')

    def _pending_action(self):
        pending = self.pending_clarification
        kind, slot, ids = (pending[key] for key in ('kind', 'slot', 'item_ids'))
        item_id = ids[0] if len(ids) == 1 else None
        if kind == 'unintelligible' and slot is None:
            missing = self._missing()
            if missing:
                slot, item_id = missing
        return Action(kind, slot, item_id)

    def _clarification(self, clarification):
        """Validate first so a bad clarification cannot invalidate good state."""
        if not isinstance(clarification, dict) or set(clarification) != {'kind', 'slot', 'item_ids'}:
            raise ValueError('A clarification requires kind, slot and item_ids only.')
        kind, slot, item_ids = (clarification[key] for key in ('kind', 'slot', 'item_ids'))
        if kind not in {'item_reference', 'order_details', 'drink_size', 'unsupported_drink', 'unsupported_menu', 'unintelligible'}:
            raise ValueError('Unknown clarification kind.')
        if slot not in {None, 'quantity', 'size', 'toppings', 'drink'}:
            raise ValueError('Unknown clarification slot.')
        if not isinstance(item_ids, list) or any(type(item_id) is not int or item_id < 1 for item_id in item_ids):
            raise ValueError('Clarification item IDs must be positive integers.')
        existing = {item['id'] for item in self.values['items']}
        if any(item_id not in existing for item_id in item_ids):
            raise ValueError('Clarification references an unknown pizza item ID.')
        ids = list(dict.fromkeys(item_ids))
        if kind == 'item_reference' and not ids:
            ids = [item['id'] for item in self.values['items']]
        return {'id': self._next_clarification_id, 'kind': kind, 'slot': slot, 'item_ids': ids}

    def _compatible_resolution(self, operations):
        return self.resolution_matches(self.pending_clarification, operations)

    @staticmethod
    def resolution_matches(pending, operations):
        kind, slot, item_ids = (pending[key] for key in ('kind', 'slot', 'item_ids'))
        for operation in operations:
            op, target, field = (operation[key] for key in ('op', 'item_id', 'slot'))
            # Every clarification kind respects explicit scope. A generic kind
            # must not turn a question about pizza 1 into an order-wide escape.
            if item_ids and target not in item_ids:
                continue
            if slot is not None and field != slot:
                structural_resolution = (
                    (kind == 'item_reference' and op == 'delete') or
                    (kind == 'order_details' and slot == 'quantity' and op in {'create', 'delete'})
                )
                if not structural_resolution:
                    continue
            if kind == 'item_reference':
                if target in item_ids and (op == 'delete' or
                        (op in {'set', 'add', 'remove', 'clear'} and (slot is None or field == slot))):
                    return True
            elif kind == 'order_details':
                if target is not None and (op in {'create', 'delete'} or field in {'quantity', 'size', 'toppings'}):
                    return True
            elif kind in {'drink_size', 'unsupported_drink'}:
                if target is None and field == 'drink' and op in {'set', 'add'} and operation['value'] not in (None, []):
                    return True
            elif kind == 'unsupported_menu':
                if slot is None or field == slot:
                    return True
            elif kind == 'unintelligible':
                return True
        return False

    def _stage_proposal(self, operations, clarification, intent, response):
        if clarification.get('kind') not in {'drink_size', 'unsupported_drink'}:
            raise ValueError('Pizza proposals are allowed only with an initial unsupported drink clarification.')
        for operation in operations:
            if not isinstance(operation, dict) or operation.get('item_id') is None or operation.get('slot') == 'drink':
                raise ValueError('Staged proposals may contain only pizza item operations.')
        trial = deepcopy(self)
        trial.apply(operations)
        return {'ops': deepcopy(operations), 'state': deepcopy(trial.values),
                'next_item_id': trial.next_item_id, 'base_version': self.version}

    def _begin_ambiguity(self):
        if not self.ambiguity_pending:
            self._ambiguous_item_ids = frozenset(item['id'] for item in self.values['items'])
        self.ambiguity_pending = True

    def _clear_ambiguity(self):
        self.ambiguity_pending = False
        self._ambiguous_item_ids = frozenset()

    def next_action(self):
        if self.pending_clarification is not None:
            return self._pending_action()
        if self.ambiguity_pending:
            return Action('ambiguous')
        if self.confirmed:
            return Action('accepted')
        missing = self._missing()
        if missing:
            return Action('ask', *missing)
        if self.awaiting_correction:
            return Action('repair')
        return Action('readback')

    def router_context(self):
        missing = self._missing()
        return {
            'state': deepcopy(self.values),
            'next_item_id': self.next_item_id,
            'requested_slot': missing[0] if missing else None,
            'requested_item_id': missing[1] if missing else None,
            'awaiting_correction': self.awaiting_correction,
            'awaiting_item_clarification': self.ambiguity_pending,
            'pending_clarification': deepcopy(self.pending_clarification),
            'pending_proposal': deepcopy(self.pending_proposal),
        }
