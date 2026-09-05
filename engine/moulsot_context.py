"""Bounded, opt-in task vocabulary for the experimental local MoulSot path."""

import hashlib
import unicodedata

LOCAL_CONTEXT_ENDPOINT = 'http://127.0.0.1:8012/transcribe'


def validate_context_target(text, *, demo, primary, fallback, protocol, endpoint):
    """Reject unsupported experiments before any speech resources are allocated."""
    if text and (not demo or primary != 'moulsot' or fallback is not False or
                 protocol != 'json' or not isinstance(endpoint, str) or
                 endpoint != LOCAL_CONTEXT_ENDPOINT):
        raise ValueError('Experimental MoulSot vocabulary requires a demo using only the dedicated local JSON bridge, with fallback disabled.')


def validate_context_text(text):
    """Validate the wire string without rewriting it; empty means no context."""
    if (not isinstance(text, str) or len(text) > 160 or text != text.strip() or
            any(character in '<>' or unicodedata.category(character).startswith('C')
                for character in text)):
        raise ValueError('MoulSot context must be trimmed text of at most 160 characters without controls or model delimiters.')
    return text


def context_sha256(text):
    return hashlib.sha256(validate_context_text(text).encode('utf-8')).hexdigest()


def configured_context(settings):
    """Read an exact enabled/terms object; absence or empty terms is a no-op."""
    if not isinstance(settings, dict):
        raise ValueError('MoulSot context requires an STT settings object.')
    if 'moulsot_context' not in settings:
        return ''
    specification = settings['moulsot_context']
    if (not isinstance(specification, dict) or set(specification) != {'enabled', 'terms'} or
            type(specification['enabled']) is not bool):
        raise ValueError('MoulSot context requires only enabled and terms, with a boolean enabled value.')
    terms = specification['terms']
    if (not isinstance(terms, list) or len(terms) > 8 or
            any(not isinstance(term, str) or not term for term in terms)):
        raise ValueError('MoulSot context supports at most eight nonempty vocabulary terms.')
    for term in terms:
        validate_context_text(term)
    if len(set(terms)) != len(terms):
        raise ValueError('MoulSot vocabulary terms must be distinct.')
    text = validate_context_text('، '.join(terms))
    return text if specification['enabled'] else ''
