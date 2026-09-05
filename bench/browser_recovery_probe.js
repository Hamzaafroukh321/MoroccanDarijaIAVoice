async (page) => {
  await page.addInitScript(() => {
    window.__recoveryFixture = { sockets: [], tracks: [], permission: 'allow', pending: [] };
    const fixture = window.__recoveryFixture;
    class Context {
      constructor() { this.sampleRate = 16000; this.state = 'running'; this.destination = {}; this.audioWorklet = { addModule: async () => {} }; }
      async resume() {} async close() { this.state = 'closed'; }
      createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    }
    class Worklet {
      constructor() { this.port = { close() {}, postMessage() {}, onmessage: null }; }
      connect() {} disconnect() {}
    }
    window.AudioContext = Context;
    window.AudioWorkletNode = Worklet;
    Object.defineProperty(navigator.mediaDevices, 'getUserMedia', { value: async () => {
      if (fixture.permission === 'deny') throw new DOMException('fixture denial', 'NotAllowedError');
      if (fixture.permission === 'pending') await new Promise(resolve => fixture.pending.push(resolve));
      const track = { stopped: false, onended: null, stop() { this.stopped = true; } };
      fixture.tracks.push(track);
      return { getTracks: () => [track], getAudioTracks: () => [track] };
    } });
    class Socket {
      static OPEN = 1;
      constructor() { this.readyState = 1; this.bufferedAmount = 0; this.sent = []; fixture.sockets.push(this); setTimeout(() => this.emit({ type: 'ready' }), 0); }
      emit(data) { this.onmessage?.({ data: JSON.stringify(data) }); }
      send(data) {
        const value = JSON.parse(data); this.sent.push(value);
        if (value.type === 'start') setTimeout(() => {
          if (fixture.setupFailure) {
            fixture.setupFailure = false;
            this.emit({ type: 'error', message: 'Temporary fixture setup failure', resume_available: true });
            return;
          }
          this.emit({ type: 'started', resumed: Boolean(value.resume_token), session_id: 'fixture' });
          this.emit({ type: 'state', state: value.resume_token ? 'CONFIRMING' : 'LISTENING',
            slots: value.resume_token ? { doctor: 'doctor_a', date: '2026-09-15', time: '10:30' } : {}, pending_proposal: null });
        }, 0);
      }
      close() { this.readyState = 3; }
    }
    window.WebSocket = Socket;
  });
  await page.goto('http://127.0.0.1:8000/?domain=clinic&mode=demo');
  await page.locator('#record').click();
  await page.waitForFunction(() => document.querySelector('#status').textContent === 'Listening');
  await page.evaluate(() => {
    const socket = window.__recoveryFixture.sockets[0];
    socket.emit({ type: 'state', state: 'CLARIFYING', slots: { doctor: 'doctor_a', date: '2026-09-15', time: '10:30' },
      pending_proposal: { state: { doctor: 'doctor_a', date: '2026-09-16', time: '14:00' } } });
    socket.emit({ type: 'error', message: 'The task router is temporarily throttled. Wait before continuing.', recovery: {
      token: 'r'.repeat(43), domain_id: 'clinic', expires_in_seconds: 1800, discarded_pending: true,
      slots: { doctor: 'doctor_a', date: '2026-09-15', time: '10:30' } } });
  });
  await page.locator('#recovery-panel:not([hidden])').waitFor();
  if (await page.locator('#record').innerText() !== 'Start over') throw new Error('Fresh start is not distinguished from resume.');
  if (await page.locator('#proposal-panel').isVisible()) throw new Error('Dropped pending changes are still shown as recoverable.');
  if (!(await page.locator('#slots').innerText()).includes('10:30') || (await page.locator('#slots').innerText()).includes('14:00')) throw new Error('Recovery displayed the held proposal.');
  if (!(await page.locator('#recovery-note').innerText()).includes('Pending changes')) throw new Error('Recovery limit is not disclosed.');
  const stopped = await page.evaluate(() => window.__recoveryFixture.tracks.every(track => track.stopped));
  if (!stopped) throw new Error('Microphone not released after error.');
  await page.setViewportSize({ width: 1200, height: 900 });
  await page.screenshot({ path: 'output/playwright/recovery_desktop.png', fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  if (await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)) throw new Error('Mobile horizontal overflow.');
  await page.screenshot({ path: 'output/playwright/recovery_mobile.png', fullPage: true });

  await page.evaluate(() => { window.__recoveryFixture.permission = 'deny'; });
  await page.locator('#resume').click();
  await page.waitForFunction(() => document.querySelector('#message').textContent.includes('Microphone access was not granted'));
  if (!await page.locator('#resume').isVisible()) throw new Error('Unspent recovery lost after microphone denial.');
  if (await page.evaluate(() => window.__recoveryFixture.sockets.length) !== 1) throw new Error('Denied microphone still sent a recovery request.');

  await page.evaluate(() => { window.__recoveryFixture.permission = 'pending'; });
  await page.locator('#resume').click();
  await page.waitForFunction(() => window.__recoveryFixture.pending.length === 1);
  await page.locator('#record').click();
  if (!await page.locator('#resume').isVisible()) throw new Error('Cancelled microphone setup lost unspent recovery.');
  await page.evaluate(() => { window.__recoveryFixture.pending.splice(0).forEach(resolve => resolve()); window.__recoveryFixture.permission = 'allow'; });
  await page.waitForFunction(() => window.__recoveryFixture.tracks.every(track => track.stopped));

  await page.evaluate(() => { window.__recoveryFixture.setupFailure = true; });
  await page.locator('#resume').click();
  await page.waitForFunction(() => document.querySelector('#message').textContent.includes('Temporary fixture setup failure'));
  if (!await page.locator('#resume').isVisible()) throw new Error('Unconsumed server recovery lost after a recoverable setup failure.');
  await page.locator('#resume').click();
  await page.waitForFunction(() => document.querySelector('#status').textContent === 'Confirming');
  const resumed = await page.evaluate(() => {
    const message = window.__recoveryFixture.sockets[2].sent[0];
    return message.type === 'start' && message.resume_token === 'r'.repeat(43) && Object.keys(message).length === 2;
  });
  if (!resumed || await page.locator('#recovery-panel').isVisible()) throw new Error('Explicit resume did not spend only the selected token.');
  if (await page.locator('#status').innerText() === 'Preferences confirmed') throw new Error('Restored values were presented as confirmed.');
  await page.evaluate(() => window.__recoveryFixture.sockets[2].emit({ type: 'error', message: 'Fixture stop', recovery: {
    token: 's'.repeat(43), domain_id: 'clinic', expires_in_seconds: 1800, discarded_pending: false, slots: { doctor: 'doctor_a' } } }));
  await page.locator('#recovery-panel:not([hidden])').waitFor();
  await page.locator('#record').click();
  await page.waitForFunction(() => window.__recoveryFixture.sockets.length === 4 && document.querySelector('#status').textContent === 'Listening');
  if (await page.evaluate(() => 'resume_token' in window.__recoveryFixture.sockets[3].sent[0])) throw new Error('Start over reused a recovery token.');
  await page.evaluate(() => window.__recoveryFixture.sockets[3].emit({ type: 'error', message: 'Fixture expiry', recovery: {
    token: 't'.repeat(43), domain_id: 'clinic', expires_in_seconds: 0.2, discarded_pending: false, slots: { doctor: 'doctor_a' } } }));
  await page.waitForFunction(() => document.querySelector('#resume').disabled && document.querySelector('#recovery-note').textContent.includes('expired'));
  return { kind: 'real_browser_mocked_microphone_and_socket', provider_calls: 0,
    checks: ['committed details only', 'pending limit disclosed', 'microphone released', 'desktop/mobile layout',
      'microphone denial preserves unspent recovery', 'cancelled setup preserves recovery', 'failed server setup preserves unspent recovery', 'explicit token-only resume',
      'fresh confirmation state', 'start over sends no token', 'expiry disables resume'] };
}
