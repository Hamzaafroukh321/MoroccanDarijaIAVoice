"""Configurable flat-task dialogue for explicitly synthetic demonstrations."""

from copy import deepcopy

from engine.dialogue import ScopedDialogue
from engine.state import Action, TaskState


class ScopedTaskState(ScopedDialogue, TaskState):
    """Root-field tasks share pending-turn rules with collection-based tasks."""

    proposal_intents = frozenset({'out_of_scope', 'ambiguous'})

    def __init__(self, config):
        super().__init__(config)
        self.repair_count = 0
        self.ambiguity_pending = False
        self.pending_clarification = None
        self.pending_proposal = None
        self._next_clarification_id = 1

    @property
    def confirmation_blocked(self):
        return self.pending_clarification is not None or self.ambiguity_pending

    def _missing(self):
        missing = [slot for key, slot in self.slots.items()
                   if slot['required'] and self.values.get(key) in (None, '', [])]
        return min(missing, key=lambda slot: slot['ask_order'])['id'] if missing else None

    def apply(self, operations, *, _resolve_proposal=False):
        if self.pending_proposal is not None and operations and not _resolve_proposal:
            raise ValueError('Resolve or discard the staged proposal before changing the task.')
        previous_version = self.version
        super().apply(operations)
        if self.pending_clarification is not None:
            self.awaiting_correction = True
        elif self.version != previous_version:
            self.ambiguity_pending = False
            self.repair_count = 0

    def _begin_ambiguity(self):
        self.awaiting_correction = True
        self.ambiguity_pending = True

    def _clear_ambiguity(self):
        self.ambiguity_pending = False

    def _clarification(self, clarification):
        if not isinstance(clarification, dict) or set(clarification) != {'kind', 'slot', 'item_ids'}:
            raise ValueError('A clarification requires kind, slot and item_ids only.')
        kind, slot, ids = (clarification[key] for key in ('kind', 'slot', 'item_ids'))
        if kind not in {'unsupported_value', 'ambiguous_value', 'unintelligible'}:
            raise ValueError('Unknown clarification kind.')
        if ids != []:
            raise ValueError('Root-field clarifications cannot address row IDs.')
        if slot is not None and (not isinstance(slot, str) or slot not in self.slots):
            raise ValueError('Unknown clarification slot.')
        if kind != 'unintelligible' and slot is None:
            raise ValueError('A value clarification requires an explicit field.')
        return {'id': self._next_clarification_id, 'kind': kind, 'slot': slot, 'item_ids': []}

    def _compatible_resolution(self, operations):
        return self.resolution_matches(self.pending_clarification, operations)

    @staticmethod
    def resolution_matches(pending, operations):
        slot = pending['slot']
        return any(operation.get('op') in {'set', 'add'} and
                   (slot is None or operation.get('slot') == slot) and
                   operation.get('value') not in (None, '', []) for operation in operations)

    def _stage_proposal(self, operations, clarification, intent, response):
        pending = self._clarification(clarification)
        if (response.get('unclear') or
                {'out_of_scope': 'unsupported_value', 'ambiguous': 'ambiguous_value'}.get(intent) != pending['kind']):
            raise ValueError('Only clear fields alongside a matching unresolved value clarification may be staged.')
        if any(operation.get('slot') == pending['slot'] for operation in operations):
            raise ValueError('A proposal cannot decide the unresolved field.')
        trial = deepcopy(self)
        trial.apply(operations)
        return {'ops': deepcopy(operations), 'state': deepcopy(trial.values), 'base_version': self.version}

    def _pending_action(self):
        pending = self.pending_clarification
        return Action(pending['kind'], pending['slot'] or self._missing())

    def _repair(self, *, count=True):
        self.invalidate_confirmation()
        if count:
            self.repair_count += 1
        if self.repair_count > self.config['demo']['max_repairs']:
            return Action('handoff')
        if self.pending_clarification is not None:
            return self._pending_action()
        return Action('repair', self._missing())

    def next_action(self):
        if self.pending_clarification is not None:
            return self._pending_action()
        if self.ambiguity_pending:
            return Action('repair', self._missing())
        if self.confirmed:
            return Action('accepted')
        missing = self._missing()
        if missing:
            return Action('ask', missing)
        return Action('repair' if self.awaiting_correction else 'readback')

    def router_context(self):
        return {'state': deepcopy(self.values), 'requested_slot': self._missing(),
                'awaiting_correction': self.awaiting_correction,
                'pending_clarification': deepcopy(self.pending_clarification),
                'pending_proposal': deepcopy(self.pending_proposal)}
