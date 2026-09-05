"""Conservative numeric time consistency checks for configured flat demos.

This is not a natural-language time parser. Only an isolated current-turn
clock expression in a narrow positive context supplies evidence. Uncovered
wording stays with the router's ordinary clarification policy.
"""
import re
from engine.normalize import temporal


class TemporalGroundingError(ValueError):
    """A routed time contradicts narrowly established current-turn evidence."""


NUMBER = re.compile(r'(?<!\w)(?:\d{1,2}:\d{2}|\d+)(?!\w)')
EXCLUDED = re.compile(r'(?<!\w)(?:ماشي|ليس|ما|ما\w*ش|قبل|بعد|بين|من|ولا|أو|او|'
                      r'or|not|never|avoid|exclude|cancel|don[\'’]t|won[\'’]t|can[\'’]t|cannot|'
                      r'before|after|between|from|instead|except)(?!\w)', re.I)
CUE = re.compile(r'(?<!\w)(?:مع|على|الساعة|ساعة|at|heure|heures|à|time)\s*[:：]?\s*$', re.I)
TRAILING = re.compile(r'[\s.!?؟،,؛]*$')
LEADING_NO = re.compile(r'^\s*(?:لا|no)(?!\w)', re.I)
CORRECTION = re.compile(r'^\s*(?:لا[\s،,]*(?:بدل|بدّل|غير|غيّر)|no[\s,]*change)(?!\w)', re.I)


def validate_temporal_grounding(response, transcript, config, context):
    """Reject conflicting routed times; never infer or rewrite an operation."""
    if config.get('demo', {}).get('state_kind') != 'flat_scoped':
        return
    fields = [slot['id'] for slot in config['slots'] if slot['type'] == 'time']
    if len(fields) != 1 or response.intent not in {'task', 'out_of_scope', 'ambiguous'}:
        return
    field = fields[0]
    operations = [op for op in [*response.ops, *response.proposed_ops]
                  if op.slot == field and op.op == 'set']
    if not operations or EXCLUDED.search(transcript):
        return
    if LEADING_NO.search(transcript) and not CORRECTION.search(transcript):
        return
    numbers = list(NUMBER.finditer(transcript))
    if len(numbers) != 1:
        return
    match = numbers[0]
    if not TRAILING.fullmatch(transcript[match.end():]):
        return
    before = transcript[:match.start()]
    pending = context.get('pending_clarification') or {}
    requested = context.get('requested_slot') == field or pending.get('slot') == field
    bare_answer = requested and not before.strip()
    if not bare_answer and not CUE.search(before):
        return
    value = match[0]
    parts = value.split(':')
    hour = int(parts[0])
    if not 0 <= hour <= 23 or (len(parts) == 2 and not 0 <= int(parts[1]) <= 59):
        return
    for operation in operations:
        # Operation format is already validated by the normal router schema.
        canonical = temporal(operation.value, 'time', config['normalization'])
        routed_hour, routed_minute = map(int, canonical.split(':'))
        # Unmarked twelve-hour speech cannot establish AM/PM. Explicit minutes
        # constrain minutes, but do not settle that period ambiguity either.
        consistent = routed_hour % 12 == hour % 12 if 1 <= hour <= 12 else routed_hour == hour
        if len(parts) == 2:
            consistent = consistent and routed_minute == int(parts[1])
        if not consistent:
            raise TemporalGroundingError('Routed time conflicts with the explicit current-utterance clock value. '
                             'Use a consistent value or ask a time clarification; do not copy a date or prior readback number.')
