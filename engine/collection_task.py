"""Configured row dialogue composed from shared transactions and scope rules."""

from copy import deepcopy

from engine.collection_schema import validate_collection_schema
from engine.confirmation import VersionedConfirmation
from engine.dialogue import ScopedDialogue
from engine.state import Action
from engine.transactions import apply_transaction


class ConfiguredCollectionState(ScopedDialogue, VersionedConfirmation):
    proposal_intents = frozenset({'out_of_scope', 'ambiguous'})

    def __init__(self, config):
        self.schema = validate_collection_schema(config)
        self.config = deepcopy(config)
        self.collection = self.schema['collection']
        self.slots = {slot['id']: slot for slot in self.config['slots']}
        self.root_slots = {key: self.slots[key] for key in self.schema['root_slots']}
        self.item_slots = {key: self.slots[key] for key in self.schema['item_slots']}
        self.values = {self.collection: []}
        self.next_item_id = 1
        self.version = 0
        self.readback_version = self.confirmed_version = None
        self.pending_clarification = self.pending_proposal = None
        self._next_clarification_id = 1
        self.repair_count = 0
        self.awaiting_correction = self.ambiguity_pending = False

    def _missing(self):
        rows = self.values[self.collection]
        required_items = sorted((slot for slot in self.item_slots.values() if slot['required']),
                                key=lambda slot: slot['ask_order'])
        if not rows:
            return required_items[0]['id'], self.next_item_id
        for row in rows:
            for slot in required_items:
                if row.get(slot['id']) in (None, '', []):
                    return slot['id'], row['id']
        for slot in sorted(self.root_slots.values(), key=lambda slot: slot['ask_order']):
            if slot['required'] and self.values.get(slot['id']) in (None, '', []):
                return slot['id'], None
        return None

    @property
    def ready(self):
        return self._missing() is None

    @property
    def confirmation_blocked(self):
        return self.pending_clarification is not None or self.ambiguity_pending

    def apply(self, operations, *, _resolve_proposal=False):
        if not isinstance(operations, list) or len(operations) > 40:
            raise ValueError('A configured collection transaction supports at most forty operations.')
        if operations and self.pending_proposal is not None and not _resolve_proposal:
            raise ValueError('Resolve or discard the staged proposal before changing the task.')
        mapped = []
        for operation in operations:
            if not isinstance(operation, dict) or set(operation) != {'op', 'item_id', 'slot', 'value'}:
                raise ValueError('A row operation requires op, item_id, slot and value only.')
            if operation['slot'] is not None and not isinstance(operation['slot'], str):
                raise ValueError('A row operation field must be a configured name or null.')
            mapped.append({**operation, 'collection':
                None if operation['slot'] in self.root_slots else self.collection})
        candidate, allocators = apply_transaction(self.values, {self.collection: self.next_item_id}, mapped,
            root_slots=self.root_slots,
            collections={self.collection: {'slots': self.item_slots,
                'max_items': self.config['demo']['collection_max_items']}},
            config=self.config, exclusive_values=self.schema['exclusive_values'])
        rows_changed = candidate[self.collection] != self.values[self.collection]
        values_changed = candidate != self.values
        allocator = allocators[self.collection]
        if values_changed or allocator != self.next_item_id:
            self.values, self.next_item_id = candidate, allocator
            self.version += 1
            self.invalidate_confirmation()
            if self.pending_clarification is None and (not self.ambiguity_pending or rows_changed):
                self.awaiting_correction = False
                self._clear_ambiguity()
                if values_changed:
                    self.repair_count = 0
        if self.pending_clarification is not None:
            self.awaiting_correction = True

    def _scope(self, clarification, values):
        if not isinstance(clarification, dict) or set(clarification) != {'kind', 'slot', 'item_ids'}:
            raise ValueError('A configured collection clarification requires kind, slot and item_ids only.')
        kind, field, ids = (clarification[key] for key in ('kind', 'slot', 'item_ids'))
        if kind not in {'item_reference', 'ambiguous_value', 'unsupported_value', 'unintelligible'}:
            raise ValueError('Unknown configured collection clarification kind.')
        if field is not None and (not isinstance(field, str) or field not in self.slots):
            raise ValueError('Unknown configured collection clarification field.')
        if (not isinstance(ids, list) or len(ids) > self.config['demo']['collection_max_items'] or
                any(type(item_id) is not int or item_id < 1 for item_id in ids) or len(set(ids)) != len(ids)):
            raise ValueError('Question row IDs must be distinct positive integers within the collection limit.')
        existing = {row['id'] for row in values[self.collection]}
        if any(item_id not in existing for item_id in ids):
            raise ValueError('A question references an unknown row ID.')
        if field in self.root_slots and ids:
            raise ValueError('Root questions cannot address row IDs.')
        if kind == 'item_reference':
            if not ids or field in self.root_slots:
                raise ValueError('An item-reference question requires explicit candidate row IDs.')
        elif kind in {'ambiguous_value', 'unsupported_value'}:
            if field is None or (field in self.item_slots and len(ids) != 1):
                raise ValueError('A value question requires a field and exactly one row when row-scoped.')
        elif field in self.item_slots and len(ids) != 1:
            raise ValueError('An unintelligible row-field question requires exactly one row.')
        return {'id': self._next_clarification_id, 'kind': kind, 'slot': field, 'item_ids': deepcopy(ids)}

    def _clarification(self, clarification):
        values = self.pending_proposal['state'] if self.pending_proposal is not None else self.values
        return self._scope(clarification, values)

    def _clarification_for_proposal(self, clarification, proposal):
        return self._scope(clarification, proposal['state']) if proposal is not None else self._clarification(clarification)

    @staticmethod
    def resolution_matches(pending, operations):
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            target, field, kind = operation.get('item_id'), operation.get('slot'), operation.get('op')
            ids = pending['item_ids']
            if ids and target not in ids:
                continue
            if not ids and target is not None and pending['slot'] is not None:
                continue
            if pending['kind'] == 'item_reference' and kind == 'delete' and target in ids:
                return True
            if pending['slot'] is not None and field != pending['slot']:
                continue
            if kind in {'set', 'add'} and operation.get('value') not in (None, '', []):
                return True
        return False

    def _compatible_resolution(self, operations):
        return self.resolution_matches(self.pending_clarification, operations)

    def _plan_resolution(self, operations):
        combined = (self.pending_proposal['ops'] if self.pending_proposal is not None else []) + operations
        trial = deepcopy(self)
        trial.apply(combined, _resolve_proposal=True)
        pending = self.pending_clarification
        if pending['kind'] in {'ambiguous_value', 'unsupported_value'}:
            target = trial.values
            if pending['item_ids']:
                target = next((row for row in trial.values[self.collection]
                               if row['id'] == pending['item_ids'][0]), {})
            if target.get(pending['slot']) in (None, '', []):
                raise ValueError('A value resolution must leave an explicit nonempty answer in its scope.')
        return None

    def _stage_proposal(self, operations, clarification, intent, response):
        if (response.get('unclear') or not isinstance(clarification, dict) or
                {'ambiguous': 'ambiguous_value', 'out_of_scope': 'unsupported_value'}.get(intent) != clarification.get('kind')):
            raise ValueError('Proposals require a matching initial unresolved value question.')
        trial = deepcopy(self)
        trial.apply(operations)
        pending = self._scope(clarification, trial.values)
        for operation in operations:
            target = operation['item_id']
            scope_matches = target in pending['item_ids'] if pending['item_ids'] else target is None
            if scope_matches and operation['slot'] == pending['slot']:
                raise ValueError('A proposal cannot decide the unresolved row or root field.')
        return {'ops': deepcopy(operations), 'state': deepcopy(trial.values),
                'next_item_id': trial.next_item_id, 'base_version': self.version}

    def _begin_ambiguity(self):
        self.ambiguity_pending = self.awaiting_correction = True

    def _clear_ambiguity(self):
        self.ambiguity_pending = False

    def _pending_action(self):
        pending = self.pending_clarification
        field, ids = pending['slot'], pending['item_ids']
        target = ids[0] if len(ids) == 1 else None
        if pending['kind'] == 'unintelligible' and field is None and not ids:
            missing = self._missing()
            if missing:
                field, target = missing
        return Action(pending['kind'], field, target)

    def _repair(self, *, count=True):
        self.invalidate_confirmation()
        if count:
            self.repair_count += 1
        if self.repair_count > self.config['demo']['max_repairs']:
            return Action('handoff')
        if self.pending_clarification is not None:
            return self._pending_action()
        missing = self._missing()
        return Action('repair', *missing) if missing else Action('repair')

    def next_action(self):
        if self.pending_clarification is not None:
            return self._pending_action()
        if self.confirmed:
            return Action('accepted')
        missing = self._missing()
        if self.ambiguity_pending or self.awaiting_correction:
            return Action('repair', *missing) if missing else Action('repair')
        return Action('ask', *missing) if missing else Action('readback')

    def router_context(self):
        missing = self._missing()
        if self.pending_clarification is not None:
            question = self.pending_clarification
            requested = (question['slot'], question['item_ids'][0] if len(question['item_ids']) == 1 else None)
        else:
            requested = missing or (None, None)
        return {'state': deepcopy(self.values), 'collection': self.collection,
            'next_item_id': self.next_item_id, 'requested_slot': requested[0], 'requested_item_id': requested[1],
            'awaiting_correction': self.awaiting_correction,
            'awaiting_item_clarification': self.ambiguity_pending,
            'pending_clarification': deepcopy(self.pending_clarification),
            'pending_proposal': deepcopy(self.pending_proposal)}
