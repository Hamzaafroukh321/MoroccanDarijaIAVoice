"""Bounded, same-process recovery of committed demo details only.

Tokens never name files or carry client operations. Old pending work and playback
authority are deliberately absent from each newly validated task.
"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
import secrets
import time


RECOVERY_ERROR = 'Saved details are unavailable or incompatible. Start a new session.'


class RecoveryError(ValueError):
    pass


def semantic_fingerprint(config):
    settings = config.get('demo', {})
    semantic = {
        'domain': config['domain_id'],
        'slots': [{key: value for key, value in slot.items() if key in {
            'id', 'type', 'required', 'values', 'aliases', 'min', 'max', 'ask_order'}}
            for slot in config['slots']],
        'normalization': config['normalization'],
        'demo': {key: settings.get(key) for key in {
            'state_kind', 'order_schema_version', 'transaction_schema', 'order_max_items',
            'collection_min_items', 'collection_max_items', 'collection_label',
            'required_slots', 'task_scope', 'readback_order', 'readback_formats',
            'labels', 'values', 'questions'}},
    }
    return hashlib.sha256(json.dumps(semantic, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def restore_values(config, values):
    """Rebuild a new adapter using validated, internally derived operations."""
    from engine.demo_validation import validate_demo_config
    from engine.task_factory import make_task

    if not config.get('demo') or not isinstance(values, dict):
        raise RecoveryError(RECOVERY_ERROR)
    validate_demo_config(config)
    if len(json.dumps(values, ensure_ascii=False, allow_nan=False).encode('utf-8')) > 65536:
        raise RecoveryError(RECOVERY_ERROR)
    settings = config['demo']
    task = make_task(config)
    if settings['state_kind'] == 'flat_scoped':
        task.apply([{'op': 'set', 'slot': key, 'value': deepcopy(value)}
                    for key, value in values.items()])
    elif settings['state_kind'] in {'collection_scoped', 'configured_collection_scoped'}:
        schema = settings['transaction_schema']
        collection = schema['collection']
        if set(values) - {collection, *schema['root_slots']}:
            raise RecoveryError(RECOVERY_ERROR)
        rows = values.get(collection, [])
        maximum = settings['collection_max_items'] if settings['state_kind'] == 'configured_collection_scoped' else settings['order_max_items']
        if not isinstance(rows, list) or len(rows) > maximum:
            raise RecoveryError(RECOVERY_ERROR)
        identifiers = set()
        operations = []
        for new_id, row in enumerate(rows, 1):
            if (not isinstance(row, dict) or type(row.get('id')) is not int or
                    row['id'] < 1 or row['id'] in identifiers or
                    set(row) - {'id', *schema['item_slots']}):
                raise RecoveryError(RECOVERY_ERROR)
            identifiers.add(row['id'])
            operations.append(dict(op='create', item_id=new_id, slot=None, value=None))
            operations.extend(dict(op='set', item_id=new_id, slot=key, value=deepcopy(value))
                              for key, value in row.items() if key != 'id')
        operations.extend(dict(op='set', item_id=None, slot=key, value=deepcopy(value))
                          for key, value in values.items() if key != collection)
        # The new task remains private until all validated reconstruction batches
        # finish; large saved states need not fit in one router-turn operation cap.
        for offset in range(0, len(operations), 40):
            task.apply(operations[offset:offset + 40])
    else:
        raise RecoveryError(RECOVERY_ERROR)
    task.invalidate_confirmation()
    return task


@dataclass
class RecoveredTask:
    task: object
    source_session_id: str
    discarded_pending: bool


class RecoveryStore:
    def __init__(self, *, ttl_seconds=1800, max_entries=32, clock=time.monotonic):
        if not 0 < ttl_seconds <= 1800 or type(max_entries) is not int or not 1 <= max_entries <= 32:
            raise ValueError('Recovery bounds exceed the supported limits.')
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.clock = clock
        self._entries = OrderedDict()

    def _purge(self):
        now = self.clock()
        for token, entry in list(self._entries.items()):
            if entry['expires'] <= now:
                self._entries.pop(token, None)

    def offer(self, config, task, source_session_id, pending_request=None):
        self._purge()
        try:
            restored = restore_values(config, task.values)
            values = restored.values
            collection = config['demo']['transaction_schema'].get('collection')
            meaningful = bool(values) if collection is None else (
                any(key != collection for key in values) or
                any(set(row) - {'id'} for row in values.get(collection, [])))
            if not meaningful:
                return None
            fingerprint = semantic_fingerprint(config)
        except (ValueError, TypeError, KeyError, OverflowError):
            return None
        discarded = bool(pending_request or getattr(task, 'pending_clarification', None) or
                         getattr(task, 'pending_proposal', None) or
                         getattr(task, 'awaiting_correction', False) or
                         getattr(task, 'ambiguity_pending', False))
        token = secrets.token_urlsafe(32)
        self._entries[token] = dict(domain=config['domain_id'], fingerprint=fingerprint,
            values=deepcopy(values), source_session_id=source_session_id,
            discarded_pending=discarded, expires=self.clock() + self.ttl_seconds)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return dict(token=token, domain_id=config['domain_id'],
                    expires_in_seconds=self.ttl_seconds, discarded_pending=discarded,
                    slots=deepcopy(values))

    def prepare(self, token, config):
        """Validate without spending the token while resources are allocated."""
        self._purge()
        if not isinstance(token, str) or not re.fullmatch(r'[A-Za-z0-9_-]{43}', token):
            raise RecoveryError(RECOVERY_ERROR)
        entry = self._entries.get(token)
        try:
            if (entry is None or entry['domain'] != config['domain_id'] or
                    entry['fingerprint'] != semantic_fingerprint(config)):
                raise RecoveryError(RECOVERY_ERROR)
            task = restore_values(config, entry['values'])
        except (ValueError, TypeError, KeyError, OverflowError):
            raise RecoveryError(RECOVERY_ERROR) from None
        return RecoveredTask(task, entry['source_session_id'], entry['discarded_pending'])

    def restore(self, token, config):
        recovered = self.prepare(token, config)
        self._entries.pop(token)
        return recovered
