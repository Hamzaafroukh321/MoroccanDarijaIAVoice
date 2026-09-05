"""Reject silent selection from narrowly explicit, configured enum alternatives.

This is not a language parser. Evidence is limited to a complete clause containing
two exact configured choices, optionally followed by explicitly named other
fields. Unknown clauses, later selections and corrections stay with the router.
"""
import re
import unicodedata


class EnumAlternativeError(ValueError):
    """The candidate commits an unresolved explicit choice."""


CLAUSES = re.compile(r'[.!?؟;؛\n]+')
CONNECTOR = r'(?:or|ou|ولا|أو|او)'


def _text(value):
    return ' '.join(unicodedata.normalize('NFC', value).casefold().split())


def _aliases(slot):
    owners = {}
    for value in slot.get('values', []):
        for alias in [value, *slot.get('aliases', {}).get(value, [])]:
            if not isinstance(alias, str) or not alias.strip() or '[[' in alias:
                continue
            owners.setdefault(_text(alias), set()).add(value)
    return owners


def unresolved_enum_alternatives(transcript, config):
    """Return only fields with an unambiguous surface pattern, never a value."""
    if config.get('demo', {}).get('state_kind') != 'flat_scoped':
        return set()
    clauses = [_text(part) for part in CLAUSES.split(transcript) if part.strip()]
    if not clauses:
        return set()
    evidence = {}
    ownership = {}
    for slot in config['slots']:
        if slot['type'] != 'enum':
            continue
        aliases = _aliases(slot)
        if not aliases:
            continue
        names = '|'.join(re.escape(name) for name in sorted(aliases, key=len, reverse=True))
        pattern = re.compile(r'(?P<left>' + names + r')\s+' + CONNECTOR +
                             r'\s+(?P<right>' + names + r')')
        ownership[slot['id']] = re.compile(r'(?<!\w)(?:' + names + r')(?!\w)')
        for index, clause in enumerate(clauses):
            match = pattern.fullmatch(clause)
            if not match:
                continue
            left, right = aliases[match['left']], aliases[match['right']]
            if len(left) == len(right) == 1 and left != right:
                evidence.setdefault(slot['id'], set()).add(index)
    if not evidence:
        return set()
    matched = set().union(*evidence.values())
    other_fields = [slot['id'].casefold() for slot in config['slots'] if slot['id'] not in evidence]
    named_field = (re.compile(r'(?:' + '|'.join(re.escape(field) for field in other_fields) +
                             r')(?::|\s)') if other_fields else None)
    for index, clause in enumerate(clauses):
        if index in matched:
            continue
        # Do not mistake a later selection, negation, quotation or restatement
        # for still-unresolved alternatives. Only explicit other-field clauses
        # are inside this deliberately narrow guard's evidence boundary.
        if (named_field is None or not named_field.match(clause) or
                any(ownership[field].search(clause) for field in evidence)):
            return set()
    return set(evidence)


def validate_enum_alternatives(response, transcript, config):
    """Reject a conflicting candidate; never choose or stage details locally."""
    fields = unresolved_enum_alternatives(transcript, config)
    if fields and (response.ops or response.is_affirmation or
                   any(operation.slot in fields for operation in response.proposed_ops)):
        raise EnumAlternativeError('Explicit enum alternatives require a choice before committing the turn.')
