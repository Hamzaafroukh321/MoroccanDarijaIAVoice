"""Frame-based endpointing. Transcript tails are supplied by the pipeline."""

from dataclasses import dataclass
import re

import numpy as np

from engine.lexicon import CONTINUATION_MARKERS, HESITATION_MARKERS


@dataclass(frozen=True)
class Segment:
    pcm: bytes
    start_ms: float
    speech_end_ms: float
    endpoint_ms: float
    speech_ms: float
    forced: bool = False


class SileroVAD:
    """A separate recurrent VAD model per session; no state crosses speakers/sessions."""
    def __init__(self, config):
        import torch
        from silero_vad import load_silero_vad
        torch.set_num_threads(config['engine']['torch_threads'])
        self.torch = torch
        self.model = load_silero_vad()
        self.sample_rate = config['runtime']['sample_rate_hz']

    def __call__(self, pcm):
        samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768
        with self.torch.inference_mode():
            return float(self.model(self.torch.from_numpy(samples), self.sample_rate).item())

    def reset(self):
        self.model.reset_states()


class EndpointDetector:
    def __init__(self, config):
        self.config = config['endpointing']
        self.frame_ms = config['runtime']['frame_ms']
        self.frame_bytes = config['runtime']['sample_rate_hz'] * self.frame_ms // 1000 * config['runtime']['sample_width_bytes']
        self.clock_ms = 0
        self.reset()

    def reset(self):
        self.buffer = bytearray()
        self.speech_ms = 0
        self.silence_ms = 0
        self.start_ms = self.clock_ms
        self.speech_end_ms = self.clock_ms
        self.was_speech = False
        self.tail = ''

    @property
    def active(self):
        return bool(self.buffer)

    def required_wait(self, transcript=None):
        transcript = self.tail if transcript is None else transcript
        tokens = re.findall(r'[\w\u0600-\u06ff]+', transcript.casefold())
        tail = tokens[-self.config['tail_token_count']:]
        wait = self.config['base_silence_ms']
        for markers, key in ((HESITATION_MARKERS, 'hesitation_hold_ms'), (CONTINUATION_MARKERS, 'continuation_hold_ms')):
            if any(marker.casefold() in tail for marker in markers if not marker.startswith('[[')):
                wait += self.config[key]
        digits = re.search(r'(?:\d[\s-]*)+$', transcript.strip())
        if digits:
            count = sum(char.isdecimal() for char in digits.group())
            if count not in self.config['digit_complete_lengths']:
                wait += self.config['digit_hold_ms']
        return min(wait, self.config['max_wait_ms'])

    def set_tail(self, text):
        self.tail = text

    def feed(self, pcm, probability, *, tail_pending=False):
        if len(pcm) != self.frame_bytes:
            raise ValueError('Endpointing requires exactly one PCM frame.')
        if not 0 <= probability <= 1:
            raise ValueError('VAD probability must be between zero and one.')
        speech = probability > self.config['vad_threshold']
        frame_start = self.clock_ms
        self.clock_ms += self.frame_ms
        if speech:
            if not self.active:
                self.start_ms = frame_start
            self.buffer.extend(pcm)
            self.speech_ms += self.frame_ms
            self.speech_end_ms = self.clock_ms
            self.silence_ms = 0
        elif self.active:
            self.buffer.extend(pcm)
            self.silence_ms += self.frame_ms
        self.was_speech = speech
        if not self.active:
            return None
        if self.clock_ms - self.start_ms >= self.config['max_segment_ms']:
            return self.flush(forced=True)
        if self.silence_ms >= self.required_wait() and not tail_pending:
            return self.flush()
        return None

    def flush(self, *, forced=False):
        segment = None
        if self.active and self.speech_ms >= self.config['min_speech_duration_ms']:
            # Exclude trailing silence from ASR; retain internal pauses.
            trim = int(self.silence_ms / self.frame_ms) * self.frame_bytes
            pcm = bytes(self.buffer[:-trim] if trim else self.buffer)
            segment = Segment(pcm, self.start_ms, self.speech_end_ms, self.clock_ms, self.speech_ms, forced)
        self.reset()
        return segment
