"""Safe, launch-time ASR evidence; no file access, network or model hashing.

The supervisor is the local source of launch verification. Matching a bridge
acknowledgment binds that report to a snapshot; it is not runtime attestation.
"""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import re


ENV_VAR = 'MOULSOT_RUNTIME_SNAPSHOT'
MAX_SNAPSHOT_BYTES = 2048
LOCAL_ENDPOINT = 'http://127.0.0.1:8012/transcribe'
SOURCE_MODEL = 'atlasia/moulsot.v0.3'
MODEL_VARIANT = 'moulsot.v0.3-Q4_K_M-mmproj-Q8_0'
VERIFICATION = 'launch_sha256_and_extracted_runtime'
_HASH_FIELDS = ('decoder_sha256', 'projector_sha256', 'runtime_archive_sha256', 'cuda_archive_sha256')
_FIELDS = frozenset({'schema_version', 'run_id', 'verified_at', 'source_model', 'model_variant',
    'conversion_revision', 'runtime_release', 'selected_device', 'verification', *_HASH_FIELDS})
_ERROR = 'Invalid local ASR runtime snapshot.'


def exact_local_endpoint(endpoint, protocol):
    return protocol == 'json' and isinstance(endpoint, str) and endpoint == LOCAL_ENDPOINT


def validate_snapshot(value):
    """Validate a closed local launch record and return a detached copy."""
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError(_ERROR)
    if (type(value['schema_version']) is not int or value['schema_version'] != 1 or
            value['source_model'] != SOURCE_MODEL or value['model_variant'] != MODEL_VARIANT or
            value['verification'] != VERIFICATION or value['selected_device'] not in ('cpu', 'cuda')):
        raise ValueError(_ERROR)
    patterns = {'run_id': r'[0-9a-f]{32}', 'conversion_revision': r'[0-9a-f]{40}',
        'runtime_release': r'b[0-9]{1,10}', 'verified_at': r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z',
        **{key: r'[0-9a-f]{64}' for key in _HASH_FIELDS}}
    for key, pattern in patterns.items():
        if not isinstance(value[key], str) or re.fullmatch(pattern, value[key], flags=re.ASCII) is None:
            raise ValueError(_ERROR)
    try:
        datetime.strptime(value['verified_at'], '%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        raise ValueError(_ERROR) from None
    return deepcopy(value)


def decode_snapshot(raw):
    """Optional environment metadata cannot break transcription configuration."""
    if raw is None or raw == '':
        return None, 'missing'
    if not isinstance(raw, str) or len(raw) > MAX_SNAPSHOT_BYTES:
        return None, 'invalid'
    try:
        if len(raw.encode('utf-8')) > MAX_SNAPSHOT_BYTES:
            return None, 'invalid'
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(_ERROR)
                result[key] = value
            return result
        return validate_snapshot(json.loads(raw, object_pairs_hook=unique_object)), 'valid'
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return None, 'invalid'


def snapshot_sha256(snapshot):
    encoded = json.dumps(validate_snapshot(snapshot), sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(encoded.encode('ascii')).hexdigest()


def runtime_ack(snapshot):
    valid = validate_snapshot(snapshot)
    return {'schema_version': 1, 'run_id': valid['run_id'], 'snapshot_sha256': snapshot_sha256(valid)}


def response_binding(snapshot, acknowledgment):
    if acknowledgment is None:
        return 'missing'
    if (not isinstance(acknowledgment, dict) or
            set(acknowledgment) != {'schema_version', 'run_id', 'snapshot_sha256'} or
            type(acknowledgment['schema_version']) is not int or acknowledgment['schema_version'] != 1 or
            not isinstance(acknowledgment['run_id'], str) or
            re.fullmatch(r'[0-9a-f]{32}', acknowledgment['run_id']) is None or
            not isinstance(acknowledgment['snapshot_sha256'], str) or
            re.fullmatch(r'[0-9a-f]{64}', acknowledgment['snapshot_sha256']) is None):
        return 'invalid'
    if snapshot is None:
        return 'unverified'
    return 'matched' if acknowledgment == runtime_ack(snapshot) else 'mismatch'


def call_provenance(snapshot, snapshot_status, *, provider, protocol, endpoint):
    """Construct fresh safe metadata for one actual provider attempt."""
    applicable = provider == 'moulsot' and exact_local_endpoint(endpoint, protocol)
    result = {'configured': {'provider': provider,
        'transport': protocol if provider == 'moulsot' else 'groq'},
        'launch_snapshot_status': snapshot_status if applicable else 'not_applicable',
        'response_binding': 'unavailable' if applicable else 'not_applicable', 'actual_device': None}
    if applicable and snapshot is not None:
        result['launch_snapshot'] = validate_snapshot(snapshot)
    return result


def record_response_provenance(record, snapshot, payload, *, eligible):
    """Ignore arbitrary provider metadata; report only allowlisted local facts."""
    if not eligible:
        record.pop('launch_snapshot', None)
        record['launch_snapshot_status'] = 'not_applicable'
        record['response_binding'] = 'not_applicable'
        return
    if not isinstance(payload, dict):
        record['response_binding'] = 'invalid'
        return
    record['response_binding'] = response_binding(snapshot, payload.get('runtime_snapshot'))
    reported = payload.get('provenance')
    if (payload.get('model_variant') == MODEL_VARIANT and isinstance(reported, dict) and
            reported.get('source_model') == SOURCE_MODEL and reported.get('quantized') is True):
        record['bridge_reported'] = {'source_model': SOURCE_MODEL, 'model_variant': MODEL_VARIANT, 'quantized': True}
