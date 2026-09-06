"""Bounded static diagnostics for collection response validation.

Never expose rejected inputs, arbitrary property names, validator messages or
validator context in diagnostics. Provenance describes a contract, not its value.
"""

from pydantic_core import PydanticCustomError


MAX_VALIDATION_RULES = 8
OTHER_RULE = 'collection_shape_other'
VALIDATION_RULES = frozenset({
    'collection_operation_field', 'collection_operation_address', 'collection_operation_value',
    'collection_question_field', 'collection_question_rows_limit', 'collection_question_address',
    'collection_link_limit', 'collection_link_field', 'collection_link_address', 'collection_link_duplicate',
    'collection_operation_shape', 'collection_question_shape', 'collection_contradictory_flags',
    'collection_unresolved_mutation', 'collection_intent_question', 'collection_resolution_shape',
    'collection_proposal_shape', 'collection_discard_shape', 'collection_request_discard',
    'collection_shape_json', 'collection_shape_required', 'collection_shape_extra',
    'collection_shape_type', 'collection_shape_constraint', OTHER_RULE,
})
_BUILTIN_RULES = {
    'json_invalid': 'collection_shape_json',
    'missing': 'collection_shape_required',
    'extra_forbidden': 'collection_shape_extra',
    **{name: 'collection_shape_type' for name in (
        'json_type', 'model_type', 'model_attributes_type', 'dict_type', 'list_type',
        'string_type', 'int_type', 'int_parsing', 'int_from_float', 'float_type',
        'float_parsing', 'bool_type', 'bool_parsing', 'none_required')},
    **{name: 'collection_shape_constraint' for name in (
        'literal_error', 'enum', 'greater_than', 'greater_than_equal', 'less_than',
        'less_than_equal', 'finite_number', 'too_long', 'too_short', 'string_too_long',
        'string_too_short', 'string_pattern_mismatch')},
}
_MESSAGE = 'The configured collection response did not satisfy its validation contract.'


def _safe_rule(rule):
    return rule if isinstance(rule, str) and rule in VALIDATION_RULES else OTHER_RULE


class CollectionValidationError(ValueError):
    """Only immutable, bounded and allowlisted provenance reaches callers."""

    def __init__(self, rules):
        if isinstance(rules, str):
            rules = [rules]
        if not isinstance(rules, (tuple, list, set, frozenset)):
            rules = [OTHER_RULE]
        normalized = sorted({_safe_rule(rule) for rule in rules} or {OTHER_RULE})
        self.validation_rules = tuple(normalized[:MAX_VALIDATION_RULES])
        self.rules_truncated = len(normalized) > MAX_VALIDATION_RULES
        super().__init__(_MESSAGE)


def collection_rule(rule):
    """A custom Pydantic rejection with a static code/message and no context."""
    return PydanticCustomError(_safe_rule(rule), _MESSAGE)


def validation_error_rules(error):
    """Map Pydantic types only; never inspect or return loc/msg/input/ctx."""
    rules = []
    for detail in error.errors(include_url=False, include_context=False, include_input=False):
        kind = detail.get('type')
        if isinstance(kind, str):
            rule = kind if kind in VALIDATION_RULES else _BUILTIN_RULES.get(kind, OTHER_RULE)
        else:
            rule = OTHER_RULE
        rules.append(rule)
    return rules or [OTHER_RULE]
