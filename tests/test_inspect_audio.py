"""Offline signal/endpoint diagnostic checks; fake VAD is not ASR evidence."""

from copy import deepcopy
import io
import json
import math
import struct
import sys
import wave

import httpx
import pytest

from bench import inspect_audio
from bench.inspect_audio import segment_audio, signal_metrics
import engine.demo as demo_module
import engine.router as router_module
import engine.stt as stt_module
from engine.config import ROOT, load_config
from engine.demo import demo_config


def pcm_samples(values):
    return struct.pack('<' + 'h' * len(values), *values)


def test_zero_signal_uses_null_dbfs_and_exact_sample_accounting():
    result = signal_metrics(bytes(1024))
    assert result['samples'] == 512 and result['duration_ms'] == 32
    assert result['rms_dbfs'] is None and result['peak_dbfs'] is None
    assert result['exact_zero_fraction'] == 1 and result['rail_fraction'] == 0
    assert result['frame_rms_dbfs'] == {'p10': None, 'p50': None, 'p90': None}


@pytest.mark.parametrize('value', [16384, -16384])
def test_constant_half_scale_has_known_rms_peak_and_frame_distribution(value):
    result = signal_metrics(pcm_samples([value] * 1024))
    expected = 20 * math.log10(.5)
    assert result['duration_ms'] == 64
    assert result['rms_dbfs'] == pytest.approx(expected, abs=.001)
    assert result['peak_dbfs'] == pytest.approx(expected, abs=.001)
    assert result['exact_zero_fraction'] == result['rail_fraction'] == 0
    for value in result['frame_rms_dbfs'].values():
        assert value == pytest.approx(expected, abs=.001)


def test_positive_and_negative_pcm_rails_are_counted_without_overflow():
    result = signal_metrics(pcm_samples([32767, -32768, 0, 16384] * 128))
    expected_rms = math.sqrt((32767 ** 2 + 32768 ** 2 + 16384 ** 2) / 4) / 32768
    assert result['peak_dbfs'] == pytest.approx(0, abs=.001)
    assert result['rms_dbfs'] == pytest.approx(20 * math.log10(expected_rms), abs=.001)
    assert result['rail_fraction'] == .5 and result['exact_zero_fraction'] == .25


@pytest.mark.parametrize('pcm', [b'', b'\x00', b'\x01\x00\x02'])
def test_invalid_pcm_is_rejected(pcm):
    with pytest.raises(ValueError):
        signal_metrics(pcm)


class SignalVAD:
    def __init__(self):
        self.resets = 0
        self.calls = []

    def reset(self):
        self.resets += 1
        self.calls = []

    def __call__(self, pcm):
        self.calls.append(pcm)
        return .99 if any(pcm) else 0


def endpoint_config(monkeypatch):
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'groq')
    config = demo_config(load_config(ROOT / 'configs/pizza.json'))
    config['endpointing'].update(base_silence_ms=64, max_wait_ms=128,
                                min_speech_duration_ms=32)
    return config


def test_internal_pause_is_retained_but_final_tail_is_excluded_and_source_clock_preserved(monkeypatch):
    config = endpoint_config(monkeypatch)
    original_config = deepcopy(config)
    voiced, silence = pcm_samples([16384] * 512), bytes(1024)
    # A one-frame pause belongs inside the first segment. Two quiet frames
    # separate the next utterance; the file ends directly on its final speech.
    pcm = voiced * 2 + silence + voiced * 2 + silence * 2 + voiced * 2
    vad = SignalVAD()
    result = segment_audio(pcm, config, vad)
    assert config == original_config and vad.resets == 1
    assert result['source_end_ms'] == 288
    assert result['vad_speech_frames'] == 6
    assert result['appended_silence_ms'] >= 128
    assert result['segments'] == [
        {'start_ms': 0, 'speech_end_ms': 160, 'endpoint_ms': 224,
         'speech_ms': 128, 'forced': False, 'uploaded_samples': 2560},
        {'start_ms': 224, 'speech_end_ms': 288, 'endpoint_ms': 352,
         'speech_ms': 64, 'forced': False, 'uploaded_samples': 1024},
    ]
    assert all(set(segment) == {'start_ms', 'speech_end_ms', 'endpoint_ms',
                               'speech_ms', 'forced', 'uploaded_samples'} for segment in result['segments'])
    assert b''.join(vad.calls[:9]) == pcm


def test_repeated_inspection_resets_vad_and_has_no_prior_file_state(monkeypatch):
    config = endpoint_config(monkeypatch)
    vad = SignalVAD()
    speech = pcm_samples([1000] * 512)
    first = segment_audio(speech, config, vad)
    second = segment_audio(speech, config, vad)
    assert first == second and vad.resets == 2
    silent = segment_audio(bytes(1024), config, vad)
    assert vad.resets == 3 and silent['segments'] == []
    assert silent['vad_speech_frames'] == 0 and silent['source_end_ms'] == 32


def write_wav(path):
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm_samples([1000] * 512))
    path.write_bytes(output.getvalue())
    return output.getvalue()


def forbid_providers(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Offline inspection allocated a provider or HTTP client')
    monkeypatch.setattr(httpx, 'AsyncClient', forbidden)
    monkeypatch.setattr(httpx, 'Client', forbidden)
    monkeypatch.setattr(router_module, 'Router', forbidden)
    monkeypatch.setattr(stt_module, 'SpeechToText', forbidden)
    monkeypatch.setattr(demo_module, 'DemoVoice', forbidden)


@pytest.mark.parametrize('use_vad', [False, True])
def test_cli_measures_without_provider_construction_or_calibrated_quality_claim(monkeypatch, tmp_path, capsys, use_vad):
    source = tmp_path / 'source.wav'
    original = write_wav(source)
    forbid_providers(monkeypatch)
    vad = SignalVAD()
    monkeypatch.setattr(inspect_audio, 'SileroVAD', lambda config: vad)
    monkeypatch.setenv('DEMO_TTS_PROVIDER', 'groq')
    monkeypatch.setattr(sys, 'argv', ['inspect_audio.py', str(source), *(['--vad'] if use_vad else [])])
    assert inspect_audio.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report['provider_requests'] == 0 and report['speech_recognition_used'] is False
    assert report['local_vad_used'] is use_vad and report['evaluation_eligible'] is False
    assert 'No intelligibility' in report['limits'] and 'calibrated microphone' in report['limits']
    assert 'optimal endpoint' in report['limits']
    assert len(report['files']) == 1 and report['files'][0]['signal']['samples'] == 512
    assert vad.resets == int(use_vad) and source.read_bytes() == original


@pytest.mark.parametrize('failure', ['malformed_later_wav', 'output_overwrites_source'])
def test_cli_prevalidates_all_inputs_and_output_before_loading_vad(monkeypatch, tmp_path, capsys, failure):
    source = tmp_path / 'source.wav'
    original = write_wav(source)
    forbid_providers(monkeypatch)
    def forbidden_vad(*args, **kwargs):
        pytest.fail('Loaded VAD before validating all source files and output target')
    monkeypatch.setattr(inspect_audio, 'SileroVAD', forbidden_vad)
    arguments = [str(source)]
    if failure == 'malformed_later_wav':
        bad = tmp_path / 'bad.wav'
        bad.write_bytes(b'not a WAV')
        arguments.append(str(bad))
    else:
        arguments.extend(['--output', str(source)])
    monkeypatch.setattr(sys, 'argv', ['inspect_audio.py', *arguments, '--vad'])
    assert inspect_audio.main() == 2
    output = capsys.readouterr().out
    assert 'NOT INSPECTED:' in output
    assert ('WAVE' if failure == 'malformed_later_wav' else 'overwrite') in output
    assert source.read_bytes() == original
