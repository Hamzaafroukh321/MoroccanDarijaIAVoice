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
        required = {'kind', 'slot', 'item_ids'}
        if (not isinstance(clarification, dict) or not required <= set(clarification) or
                set(clarification) - required - {'coupled_slots'}):
            raise ValueError('A clarification requires kind, slot and item_ids, with optional coupled_slots.')
        kind, slot, ids = (clarification[key] for key in ('kind', 'slot', 'item_ids'))
        if kind not in {'unsupported_value', 'ambiguous_value', 'unintelligible'}:
            raise ValueError('Unknown clarification kind.')
        if ids != []:
            raise ValueError('Root-field clarifications cannot address row IDs.')
        if slot is not None and (not isinstance(slot, str) or slot not in self.slots):
            raise ValueError('Unknown clarification slot.')
        if kind != 'unintelligible' and slot is None:
            raise ValueError('A value clarification requires an explicit field.')
        coupled = clarification.get('coupled_slots')
        if coupled is not None:
            if (not isinstance(coupled, list) or len(coupled) > 39 or len(coupled) >= len(self.slots) or
                    any(not isinstance(key, str) or key not in self.slots for key in coupled) or
                    len(set(coupled)) != len(coupled) or slot in coupled):
                raise ValueError('Coupled fields must be distinct configured fields excluding the primary field.')
            if coupled and kind != 'ambiguous_value':
                raise ValueError('Only ambiguous value clarifications can couple fields.')
        pending = {'id': self._next_clarification_id, 'kind': kind, 'slot': slot, 'item_ids': []}
        if coupled:
            pending['coupled_slots'] = deepcopy(coupled)
        return pending

    def _requires_proposal(self, clarification):
        return bool(self.pending_clarification is None and isinstance(clarification, dict) and
                    clarification.get('coupled_slots'))

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
        group = [pending['slot']] + pending.get('coupled_slots', [])
        if any(operation.get('slot') in group for operation in operations):
            raise ValueError('A proposal cannot decide an unresolved field.')
        trial = deepcopy(self)
        trial.apply(operations)
        proposal = {'ops': deepcopy(operations), 'state': deepcopy(trial.values), 'base_version': self.version}
        if pending.get('coupled_slots'):
            proposal.update(coupled_slots=group, answered_slots=[], remaining_slots=deepcopy(group))
        return proposal

    def _plan_resolution(self, operations):
        proposal = self.pending_proposal
        if proposal is None or not proposal.get('coupled_slots'):
            return None
        combined = proposal['ops'] + operations
        if len(combined) > 40:
            raise ValueError('A coupled transaction supports at most 40 staged operations.')
        group = proposal['coupled_slots']
        if any(operation.get('slot') in group and
               (operation.get('op') not in {'set', 'add'} or operation.get('value') in (None, '', []))
               for operation in operations):
            raise ValueError('Coupled fields require explicit nonempty positive answers.')
        # Use the committed base, not the draft preview, so staged add operations
        # are applied exactly once and no malformed late operation can leak out.
        trial = deepcopy(self)
        trial.apply(combined, _resolve_proposal=True)
        answered = set(proposal['answered_slots'])
        answered.update(operation['slot'] for operation in operations
                        if operation['slot'] in group and operation['op'] in {'set', 'add'} and
                        trial.values.get(operation['slot']) not in (None, '', []))
        remaining = [key for key in group if key not in answered]
        if not remaining:
            return None
        next_slot = remaining[0]
        pending = self._clarification({'kind': 'ambiguous_value', 'slot': next_slot,
            'item_ids': [], 'coupled_slots': [key for key in group if key != next_slot]})
        staged = {'ops': deepcopy(combined), 'state': deepcopy(trial.values),
                  'base_version': proposal['base_version'], 'coupled_slots': deepcopy(group),
                  'answered_slots': [key for key in group if key in answered],
                  'remaining_slots': remaining}
        return {'pending': pending, 'proposal': staged}

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
        requested = self.pending_clarification['slot'] if self.pending_clarification else self._missing()
        return {'state': deepcopy(self.values), 'requested_slot': requested,
                'awaiting_correction': self.awaiting_correction,
                'pending_clarification': deepcopy(self.pending_clarification),
                'pending_proposal': deepcopy(self.pending_proposal)}
