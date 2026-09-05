// The AudioContext supplies resampled mono audio at the configured rate.
// Render-quantum sizes may vary: accumulate samples into exact 32 ms frames.
class RecorderProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const config = options.processorOptions;
    if (sampleRate !== config.sample_rate_hz) throw new Error('Unexpected audio sample rate.');
    this.frameSamples = config.sample_rate_hz * config.frame_ms / 1000;
    this.maxSamples = config.sample_rate_hz * config.max_recording_ms / 1000;
    this.sampleWidth = config.sample_width_bytes;
    this.totalSamples = 0;
    this.offset = 0;
    this.stopped = false;
    this.allocate();
    this.port.onmessage = (event) => {
      if (event.data.type === 'stop') this.finish();
    };
  }

  allocate() {
    this.buffer = new ArrayBuffer(this.frameSamples * this.sampleWidth);
    this.view = new DataView(this.buffer);
  }

  emitFrame() {
    this.port.postMessage({ type: 'frame', buffer: this.buffer }, [this.buffer]);
    this.offset = 0;
    this.allocate();
  }

  finish() {
    if (this.stopped) return;
    this.stopped = true;
    // The unused bytes in a new ArrayBuffer are zero. The server trims padding
    // using totalSamples, retaining exact 32 ms framing on the wire.
    if (this.offset) this.emitFrame();
    this.port.postMessage({ type: 'finished', samples: this.totalSamples });
  }

  process(inputs) {
    if (this.stopped) return false;
    const channel = inputs[0]?.[0];
    if (!channel) return true;
    for (const value of channel) {
      const sample = Math.max(-1, Math.min(1, value));
      // Signed PCM16 little-endian is the fixed wire format, not a threshold.
      this.view.setInt16(this.offset * this.sampleWidth,
        Math.round(sample * (sample < 0 ? 0x8000 : 0x7fff)), true);
      this.offset += 1;
      this.totalSamples += 1;
      if (this.offset === this.frameSamples) this.emitFrame();
      if (this.totalSamples >= this.maxSamples) {
        this.finish();
        return false;
      }
    }
    return true;
  }
}

registerProcessor('darija-recorder', RecorderProcessor);
