// Run through playwright-cli run-code --filename. Local browser diagnostic:
// real AudioWorklet/WebAudio, synthetic silent input and mocked WebSocket only.
async (page) => {
  const results = [];
  for (const domain of ['clinic', 'pizza']) {
    await page.goto(`http://127.0.0.1:8000/?domain=${domain}&mode=demo`);
    await page.evaluate(() => {
      const probe = window.__voiceProbe = { controls: [], frames: 0, decoded: 0 };
      navigator.mediaDevices.getUserMedia = async () => {
        probe.inputContext = new AudioContext({ sampleRate: 16000 });
        const destination = probe.inputContext.createMediaStreamDestination();
        probe.inputSource = probe.inputContext.createConstantSource();
        probe.inputSource.offset.value = 0;
        probe.inputSource.connect(destination); probe.inputSource.start();
        await probe.inputContext.resume();
        probe.stream = destination.stream;
        return destination.stream;
      };
      const decode = AudioContext.prototype.decodeAudioData;
      AudioContext.prototype.decodeAudioData = async function (bytes) {
        const buffer = await decode.call(this, bytes);
        probe.decoded++;
        if (probe.delayDecode) await new Promise(resolve => { probe.releaseDecode = resolve; });
        return buffer;
      };
      window.WebSocket = class {
        static OPEN = 1;
        constructor() {
          this.readyState = 1; this.bufferedAmount = 0; probe.socket = this;
          queueMicrotask(() => this.deliver({ type: 'ready' }));
        }
        deliver(value) { this.onmessage?.({ data: value instanceof ArrayBuffer ? value : JSON.stringify(value) }); }
        send(value) {
          if (value instanceof ArrayBuffer) { probe.frames++; return; }
          const control = JSON.parse(value); probe.controls.push(control);
          if (control.type === 'start') queueMicrotask(() => this.deliver({ type: 'started', session_id: 'browser-fixture' }));
        }
        close() { this.readyState = 3; }
      };
      probe.deliverAudio = (id, seconds, incomplete = false) => {
        const size = Math.round(16000 * seconds) * 2;
        const raw = new ArrayBuffer(44 + size); const view = new DataView(raw);
        const tag = (offset, value) => [...value].forEach((character, index) => view.setUint8(offset + index, character.charCodeAt(0)));
        tag(0, 'RIFF'); view.setUint32(4, 36 + size, true); tag(8, 'WAVE'); tag(12, 'fmt ');
        view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
        view.setUint32(24, 16000, true); view.setUint32(28, 32000, true);
        view.setUint16(32, 2, true); view.setUint16(34, 16, true); tag(36, 'data'); view.setUint32(40, size, true);
        probe.socket.deliver({ type: 'audio_start', playback_id: id, bytes: raw.byteLength, action: 'readback' });
        for (let offset = 0; offset < raw.byteLength - (incomplete ? 1024 : 0); offset += 1024)
          probe.socket.deliver(raw.slice(offset, Math.min(offset + 1024, raw.byteLength)));
        probe.socket.deliver({ type: 'audio_end', playback_id: id });
      };
    });
    await page.getByRole('button', { name: 'Start conversation', exact: true }).click();
    await page.waitForFunction(() => current?.phase === 'recording' && __voiceProbe.frames > 2);
    await page.evaluate(() => __voiceProbe.deliverAudio('complete', .2));
    await page.waitForFunction(() => __voiceProbe.controls.some(control => control.playback_id === 'complete'));
    await page.evaluate(() => __voiceProbe.deliverAudio('interrupted', .5));
    await page.waitForFunction(() => current?.botSource != null);
    await page.evaluate(() => __voiceProbe.socket.deliver({ type: 'audio_stop', playback_id: 'interrupted' }));
    await page.waitForTimeout(650);
    await page.evaluate(() => { __voiceProbe.delayDecode = true; __voiceProbe.deliverAudio('late_decode', .2); });
    await page.waitForFunction(() => typeof __voiceProbe.releaseDecode === 'function');
    await page.evaluate(() => {
      __voiceProbe.socket.deliver({ type: 'audio_stop', playback_id: 'late_decode' });
      __voiceProbe.delayDecode = false; __voiceProbe.releaseDecode();
    });
    await page.waitForTimeout(300);
    const beforeFailure = await page.evaluate(() => ({
      decoded: __voiceProbe.decoded, frames: __voiceProbe.frames,
      acknowledgements: __voiceProbe.controls.filter(control => control.type === 'playback_finished'),
      botStopped: current.botSource === null, noPendingPlayback: current.playbackId === null,
    }));
    if (beforeFailure.acknowledgements.length !== 1 || beforeFailure.acknowledgements[0].playback_id !== 'complete' || !beforeFailure.botStopped || !beforeFailure.noPendingPlayback)
      throw new Error(`Playback lifecycle failed for ${domain}: ${JSON.stringify(beforeFailure)}`);
    await page.evaluate(() => __voiceProbe.deliverAudio('incomplete', .2, true));
    await page.waitForFunction(() => current === null && document.getElementById('message').textContent.includes('incomplete'));
    const cleanup = await page.evaluate(async () => {
      const tracksEnded = __voiceProbe.stream.getTracks().every(track => track.readyState === 'ended');
      __voiceProbe.inputSource.stop(); await __voiceProbe.inputContext.close();
      return { tracksEnded, message: document.getElementById('message').textContent };
    });
    if (!cleanup.tracksEnded) throw new Error('Synthetic input track was not released.');
    results.push({ domain, ...beforeFailure, ...cleanup });
  }
  const report = { kind: 'browser_playback_diagnostic', provider_requests: 0,
    input: 'synthetic silence through a real AudioWorklet', transport: 'mock WebSocket',
    playback: 'real decodeAudioData and AudioBufferSourceNode ended event',
    limitations: 'No physical microphone, speaker audibility, ASR, router or TTS validation.', results };
  await page.evaluate(value => { window.__voiceProbeReport = value; }, report);
  return report;
}
