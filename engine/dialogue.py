"""Shared scoped clarification, proposal and confirmation lifecycle.

Adapters own field meaning, scope compatibility and atomic transactions. This
mixin owns the protocol for retaining, resolving or discarding pending work.
"""

from copy import deepcopy


def validate_resolution_identity(pending, operations, resolution_id, matches):
    """Question-scoped answers must explicitly name the current question."""
    if resolution_id is not None:
        if type(resolution_id) is not int or pending is None or resolution_id != pending['id']:
            raise ValueError('Unknown or stale clarification resolution ID.')
    elif pending is not None and operations and matches(pending, operations):
        raise ValueError('Operations answering a pending clarification require its resolution ID.')


def validate_request_discard(response, retained_request=None, *, pending_clarification=None,
                             pending_proposal=None, check_context=True):
    """Forget only the explicitly named, resolved request's recovery history."""
    request_id = response.get('discard_request')
    if request_id is None:
        return
    if (not isinstance(request_id, str) or not request_id or response.get('intent') != 'task' or
            response.get('ops') != [] or response.get('proposed_ops', []) != [] or
            any(response.get(key) is not False for key in
                ('is_affirmation', 'is_negation', 'unclear')) or
            any(response.get(key) is not None for key in
                ('clarification', 'resolves_clarification', 'discard_clarification'))):
        raise ValueError('Discarding retained history requires a standalone task and its request ID.')
    if check_context and (pending_clarification is not None or pending_proposal is not None or
            not isinstance(retained_request, dict) or retained_request.get('id') != request_id or
            retained_request.get('clarification_resolved') is not True):
        raise ValueError('Unknown, stale or unresolved retained request ID.')


class ScopedDialogue:
    """Adapter protocol, deliberately independent of domain fields.

    Required hooks:
      _clarification(payload): validate without mutation; return pending metadata
          containing the current _next_clarification_id as its id.
      resolution_matches(pending, operations): pure check against question scope.
      _compatible_resolution(operations): check against the current pending scope.
      _stage_proposal(operations, clarification, intent, response): validate
          without mutation; return ops/state/base_version plus adapter metadata.
      _requires_proposal(clarification): optionally stage an empty initial
          transaction when the adapter needs explicit coverage of several fields.
      _plan_resolution(operations): optionally clone-validate and return a
          proposal/pending continuation without mutation; None commits normally.
      _begin_ambiguity(), _clear_ambiguity(): manage ambiguity_pending and any
          adapter-specific ambiguity context; awaiting_correction is owned here.
      _repair(count=True), next_action(): select adapter response actions.
      apply(operations, _resolve_proposal=False): commit atomically, preserve
          pending blockers, and advance version for committed changes.
      invalidate_confirmation(), accept_confirmation(operations): provided by
          VersionedConfirmation or an equivalent confirmation guard.

    Adapters initialize pending_clarification, pending_proposal,
    _next_clarification_id, ambiguity_pending, awaiting_correction, repair_count
    and version. Proposal validation and incompatible/stale resolution must
    fail before state mutation. A successful explicit resolution always
    invalidates prior readback, even when its operations restate current values.
    """

    proposal_intents = frozenset({'out_of_scope'})

    def _requires_proposal(self, clarification):
        return False

    def _clarification_for_proposal(self, clarification, proposal):
        """Adapters with new draft rows may validate scope against the preview."""
        return self._clarification(clarification)

    def _plan_resolution(self, operations):
        """Optional validated continuation; None keeps the normal commit path."""
        return None

    def consume(self, response, *, retained_request=None):
        if response.get('discard_request') is not None:
            validate_request_discard(response, retained_request,
                pending_clarification=self.pending_clarification,
                pending_proposal=self.pending_proposal)
            self.awaiting_correction = False
            self._clear_ambiguity()
            self.repair_count = 0
            self.version += 1
            self.invalidate_confirmation()
            return self.next_action()
        intent = response.get('intent', 'task')
        if intent not in {'task', 'greeting', 'help', 'out_of_scope', 'ambiguous', 'unclear'}:
            raise ValueError('Unknown dialogue intent.')
        operations = response['ops']
        yes, no, unclear = (response[key] for key in ('is_affirmation', 'is_negation', 'unclear'))
        if any(type(flag) is not bool for flag in (yes, no, unclear)) or (yes and no):
            raise ValueError('Invalid confirmation flags.')
        if (intent != 'task' or unclear) and (operations or yes or no):
            raise ValueError('An unresolved turn cannot modify or confirm the task.')
        clarification = response.get('clarification')
        resolution_id = response.get('resolves_clarification')
        proposed_ops = response.get('proposed_ops', [])
        discard_id = response.get('discard_clarification')
        if not isinstance(proposed_ops, list) or len(proposed_ops) > 40:
            raise ValueError('A proposal must contain at most 40 operations.')
        if discard_id is not None:
            if type(discard_id) is not int or self.pending_clarification is None or discard_id != self.pending_clarification['id']:
                raise ValueError('Unknown or stale clarification discard ID.')
            if intent != 'task' or operations or proposed_ops or yes or no or unclear or clarification is not None or resolution_id is not None:
                raise ValueError('Discard must explicitly cancel only the pending clarification.')
            self.pending_clarification = None
            self.pending_proposal = None
            self.awaiting_correction = False
            self._clear_ambiguity()
            self.repair_count = 0
            self.version += 1
            self.invalidate_confirmation()
            return self.next_action()
        proposal = None
        validate_resolution_identity(self.pending_clarification, operations, resolution_id,
                                     self.resolution_matches)
        if proposed_ops or self._requires_proposal(clarification):
            if (self.pending_clarification is not None or intent not in self.proposal_intents or
                    operations or yes or no or resolution_id is not None or
                    not isinstance(clarification, dict)):
                raise ValueError('Proposals require an initial supported clarification and clearly extracted fields.')
            proposal = self._stage_proposal(proposed_ops, clarification, intent, response)
        if resolution_id is not None:
            if clarification is not None or intent != 'task' or unclear:
                raise ValueError('A clarification resolution must be an explicit task response.')
            if not operations or not self._compatible_resolution(operations):
                raise ValueError('The operations do not resolve the pending clarification scope.')
            if self.pending_proposal is not None and self.pending_proposal['base_version'] != self.version:
                raise ValueError('The staged proposal is stale; discard it before continuing.')
        if clarification is not None:
            if operations or yes or no:
                raise ValueError('A new clarification cannot change or confirm the task.')
            pending = self._clarification_for_proposal(clarification, proposal)
            if self.pending_clarification is None:
                # Keep the first unresolved question aligned with its retained
                # transcript. A later failure cannot replace it with an easier
                # scope and silently discard the original request.
                self.pending_clarification = pending
                self.pending_proposal = proposal
                self._next_clarification_id += 1
            self.awaiting_correction = True
            return self._repair()
        if intent == 'ambiguous':
            self._begin_ambiguity()
            self.awaiting_correction = True
        if intent != 'task' or unclear:
            return self._repair(count=intent != 'greeting')
        if resolution_id is not None:
            continuation = self._plan_resolution(operations)
            if continuation is not None:
                # The adapter validates the entire candidate before returning.
                # A partial answer changes only the draft, never committed facts.
                self.pending_proposal = continuation['proposal']
                self.pending_clarification = continuation['pending']
                self._next_clarification_id += 1
                self.awaiting_correction = True
                self.repair_count = 0
                self.invalidate_confirmation()
                return self.next_action()
        previous_version = self.version
        previous_values = deepcopy(self.values)
        was_awaiting_correction = self.awaiting_correction
        combined = (self.pending_proposal['ops'] + operations
                    if resolution_id is not None and self.pending_proposal is not None else operations)
        self.apply(combined, _resolve_proposal=resolution_id is not None)
        if resolution_id is not None:
            # Semantic resolution can restate an already stored choice. Advance
            # the epoch even then so an old playback ACK cannot confirm it.
            if self.version == previous_version:
                self.version += 1
            self.pending_clarification = None
            self.pending_proposal = None
            self.awaiting_correction = False
            self._clear_ambiguity()
            self.repair_count = 0
            self.invalidate_confirmation()
            return self.next_action()
        if self.pending_clarification is not None:
            return self._repair()
        if self.ambiguity_pending:
            # Acknowledgement cannot resolve an unresolved reference. A stale
            # readback callback or redundant operation cannot resolve it either.
            return self._repair()
        if operations and not no:
            # An unchanged positive restatement can repair a rejected complete
            # summary. It cannot answer another missing field, or recover by
            # clearing/removing something already absent. Scoped blockers were
            # checked above; empty add carries no restated information.
            restatement = (was_awaiting_correction and self.ready and
                all(operation['op'] in {'set', 'add'} and
                    not (operation['op'] == 'add' and operation['value'] == [])
                    for operation in operations))
            # Allocating then deleting a row advances its ID/version but does
            # not recover any task information.
            if self.values != previous_values or restatement:
                self.awaiting_correction = False
                self.repair_count = 0
            else:
                # Flat adapters may clear this flag merely on receiving ops.
                self.awaiting_correction = was_awaiting_correction
        if no:
            self.invalidate_confirmation()
            if not operations or self.version == previous_version:
                self.awaiting_correction = True
                return self._repair()
        elif yes:
            self.accept_confirmation(operations)
        elif operations:
            # Even a redundant operation alongside 'yes' needs a fresh readback.
            self.invalidate_confirmation()
        if self.awaiting_correction:
            # A rejected summary still needs an explicit correction. Repeated
            # acknowledgements must use the same bounded repair budget.
            return self._repair()
        return self.next_action()
