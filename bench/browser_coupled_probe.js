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
      constructor() { this.port = { close() {}, onmessage: null, postMessage: value => {
        if (value.type === 'stop') setTimeout(() => this.port.onmessage?.({data:{type:'finished',samples:0}}),0);
      } }; }
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
        if (value.type === 'stop') setTimeout(() => this.emit({type:'done',status:'interrupted',slots:{}}),0);
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
  await page.waitForFunction(() => window.__recoveryFixture.sockets.length === 1 && document.querySelector('#status').textContent === 'Listening');
  await page.evaluate(() => window.__recoveryFixture.sockets[0].emit({type:'state',state:'LISTENING',
    slots:{doctor:'doctor_a',date:'2026-09-15',time:'10:00'},
    pending_proposal:{state:{doctor:'doctor_b',date:'2026-09-15',time:'10:00'},
      coupled_slots:['doctor','time'],answered_slots:['doctor'],remaining_slots:['time']}}));
  await page.evaluate(() => window.__recoveryFixture.sockets[0].emit({type:'assistant_text',text:'فاش من ساعة بغيتي؟',action:'ambiguous_value',playback_id:'fixture-question'}));
  await page.locator('#proposal-panel:not([hidden])').waitFor();
  const saved=await page.locator('#slots').innerText();
  const proposed=await page.locator('#proposed-slots').innerText();
  if(!saved.includes('10:00') || proposed.includes('10:00') || !proposed.includes('باء')) throw new Error('Draft inherited the unresolved old time or lost selected B.');
  const note=await page.locator('#proposal-note').innerText();
  if(!note.includes('الساعة') && !note.includes('time')) {
    if(!note.includes('answer:') || !note.includes('have not changed')) throw new Error('Missing unresolved-field disclosure.');
  }
  if(await page.locator('#status').innerText()==='Preferences confirmed') throw new Error('Partial draft was confirmed.');
  await page.setViewportSize({width:1365,height:1100});
  if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)) throw new Error('Desktop overflow.');
  await page.screenshot({path:'output/playwright/coupled_desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)) throw new Error('Mobile overflow.');
  await page.screenshot({path:'output/playwright/coupled_mobile.png',fullPage:true});
  await page.evaluate(() => window.__recoveryFixture.sockets[0].emit({type:'state',state:'CONFIRMING',
    slots:{doctor:'doctor_b',date:'2026-09-15',time:'11:00'},pending_proposal:null}));
  if(await page.locator('#proposal-panel').isVisible() || !(await page.locator('#slots').innerText()).includes('11:00')) throw new Error('Final commit did not clear draft.');
  await page.locator('#record').click();
  await page.waitForFunction(()=>window.__recoveryFixture.tracks.every(track=>track.stopped));
  return {kind:'real_browser_mocked_microphone_and_socket',provider_calls:0,checks:['committed values retained','unresolved inherited time omitted from draft','selected B visible','pending field named','not confirmed','desktop/mobile no overflow','final commit clears draft','microphone released']};
}
