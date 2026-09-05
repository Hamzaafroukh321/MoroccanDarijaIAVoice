'use strict';

const config = JSON.parse(document.getElementById('capture-config').textContent);
const button = document.getElementById('record');
const statusLabel = document.getElementById('status');
const message = document.getElementById('message');
const progressTime = document.getElementById('progress-time');
const playback = document.getElementById('playback');
const recording = document.getElementById('recording');
const savedPath = document.getElementById('saved-path');
const modeSelect = document.getElementById('mode');
const domainSelect = document.getElementById('domain');
const setup = document.getElementById('setup');
const transcript = document.getElementById('transcript');
const slots = document.getElementById('slots');
const proposalPanel = document.getElementById('proposal-panel');
const proposedSlots = document.getElementById('proposed-slots');
const recoveryPanel = document.getElementById('recovery-panel');
const recoveryNote = document.getElementById('recovery-note');
const resumeButton = document.getElementById('resume');
const conversation = document.getElementById('conversation');
const isVoice = session => ['voice', 'demo'].includes(session.mode);
const demoSpeechProvider = {darija_xtts: 'Darija XTTS 2.1 on Hugging Face', azure: 'Azure Speech (Mouna)', elevenlabs: 'ElevenLabs', groq: 'Groq'}[config.demo_tts_provider] || 'Groq';
const demoSupported = config.demo_supported ?? (config.domain_id === 'pizza');
const demoUI = {
  ...(config.domain_id === 'pizza' ? {
    title: 'Let’s order a pizza.',
    note: 'Ask for a pizza, change a detail, then confirm the summary. Pause briefly after each turn. This demo does not place an order.',
    state_title: 'Your order',
    state_empty: 'Each pizza keeps its quantity, size and toppings. Drinks belong to the order.',
    completion_text: 'You confirmed the demo order. No real order was placed.',
  } : {
    title: config.domain_title || 'Your task preferences',
    note: 'Share your preferences, change a detail, then confirm the summary. This demo does not book an appointment.',
    state_title: 'Your preferences',
    state_empty: 'Your requested details will appear here as you speak.',
    completion_text: 'Your demo preferences were confirmed. No appointment was booked.',
  }),
  ...(config.demo_ui || {}),
};
const demoLabels = config.demo_labels || {};
const demoValues = config.demo_values || {};
const demoStateKind = config.demo_state_kind ?? (config.domain_id === 'pizza' ? 'collection_scoped' : 'flat_scoped');
document.getElementById('demo-voice-label').textContent = `${config.demo_asr_label || 'MoulSot'} · Synthetic voice: ${config.demo_tts_voice_label || 'Arabic'}`;
document.getElementById('xtts-license').hidden = config.demo_tts_provider !== 'darija_xtts';
if (Array.isArray(config.domains)) {
  domainSelect.replaceChildren(...config.domains.map(domain => {
    const option = document.createElement('option');
    option.value = domain.id;
    option.textContent = domain.title;
    return option;
  }));
}
domainSelect.value = config.domain_id;
domainSelect.addEventListener('change', () => {
  const address = new URL('/', location.origin);
  address.searchParams.set('domain', domainSelect.value);
  address.searchParams.set('mode', modeSelect.value);
  location.href = address;
});
const voiceOption = modeSelect.querySelector('option[value="voice"]');
voiceOption.disabled = config.voice_issues.length > 0;
const demoOption = modeSelect.querySelector('option[value="demo"]');
demoOption.disabled = !demoSupported;
modeSelect.value = config.initial_mode;
if (modeSelect.selectedOptions[0]?.disabled) modeSelect.value = 'capture';
if (config.bank_target) {
  modeSelect.disabled = true;
  document.getElementById('session-title').textContent = 'Record: ' + config.bank_target;
} else document.getElementById('session-title').textContent = config.domain_title;

const frameBytes = config.sample_rate_hz * config.frame_ms / 1000 * config.sample_width_bytes;
let current = null;
let recordingUrl = null;
let recovery = null;
let recoveryTimer = null;

function clearRecovery() {
  clearTimeout(recoveryTimer);
  recoveryTimer = null;
  recovery = null;
  recoveryPanel.hidden = true;
  button.classList.remove('secondary');
}

function offerRecovery(value) {
  clearRecovery();
  if (!value || value.domain_id !== config.domain_id || modeSelect.value !== 'demo' ||
      typeof value.token !== 'string' || !/^[A-Za-z0-9_-]{20,128}$/.test(value.token) ||
      !Number.isFinite(value.expires_in_seconds) || value.expires_in_seconds <= 0 ||
      value.expires_in_seconds > 1800 || !value.slots || typeof value.slots !== 'object' || Array.isArray(value.slots)) return;
  recovery = { ...value, expiresAt: Date.now() + value.expires_in_seconds * 1000 };
  showSlots(value.slots);
  showRecovery();
  recoveryTimer = setTimeout(() => {
    recovery = null;
    recoveryTimer = null;
    if (!current && modeSelect.value === 'demo') {
      resumeButton.disabled = true;
      recoveryNote.textContent = 'These saved details have expired. Start over to begin a new conversation.';
      button.classList.remove('secondary');
    } else recoveryPanel.hidden = true;
  }, value.expires_in_seconds * 1000);
}

function showRecovery() {
  if (!recovery || current || recovery.expiresAt <= Date.now() || modeSelect.value !== 'demo') {
    recoveryPanel.hidden = true;
    return;
  }
  recoveryPanel.hidden = false;
  resumeButton.disabled = false;
  recoveryNote.textContent = 'Continue with the saved details below, then confirm them again.' +
    (recovery.discarded_pending ? ' Pending changes will need to be repeated.' : '') +
    ' Available for up to 30 minutes while this page and server stay open.';
  button.textContent = 'Start over';
  button.classList.add('secondary');
}

function display(status, text, buttonText, disabled = false) {
  if (!current) { modeSelect.disabled = Boolean(config.bank_target); domainSelect.disabled = false; }
  if (statusLabel.textContent !== status) statusLabel.textContent = status;
  if (message.textContent !== text) message.textContent = text;
  button.textContent = buttonText;
  button.disabled = disabled;
  statusLabel.dataset.active = current?.phase === 'recording' ? 'true' : 'false';
}

const voiceStates = {
  IDLE: ['Ready', 'Start when you are ready.'], LISTENING: ['Listening', 'Speak naturally. Corrections are welcome.'],
  PROCESSING: ['Understanding', 'Updating the task from what you said…'], SPEAKING: ['Speaking', 'You can interrupt to make a correction.'],
  CLARIFYING: ['Please repeat', 'The assistant is asking for clearer input.'], CONFIRMING: ['Confirming', 'Listen to the readback, then confirm or correct it.'],
  DONE: ['Finished', 'The session has ended.'],
};
const progressStages = {
  transcribing: ['Transcribing', 'Turning your audio into text…', 'Still waiting for the transcription. You can end the session.'],
  routing: ['Understanding', 'Working out the details you requested…', 'Still waiting to understand the requested details. You can end the session.'],
  synthesizing: ['Preparing reply', 'Preparing the spoken reply…', 'Still waiting for the spoken reply. You can end the session.'],
};

function showVoiceStatus(session) {
  if (current !== session || session.phase !== 'recording') return;
  const progress = session.progress;
  if (progress?.visible) {
    const [label, normal, waiting] = progressStages[progress.stage];
    display(label, progress.longWait ? waiting : normal, 'End session');
  } else {
    const [label, text] = voiceStates[session.latestState] || voiceStates.LISTENING;
    display(label, text, 'End session');
  }
}

function clearProgress(session, restore = false) {
  if (session?.progress) clearTimeout(session.progress.timer);
  if (session) session.progress = null;
  if (!current || current === session) {
    progressTime.hidden = true;
    progressTime.textContent = '';
  }
  if (restore && session) showVoiceStatus(session);
}

function updateProgress(session, result) {
  if (current !== session || session.phase !== 'recording' ||
      typeof result.progress_id !== 'string' || !result.progress_id) return;
  if (result.active === false) {
    if (session.progress?.id === result.progress_id) clearProgress(session, true);
    return;
  }
  if (result.active !== true || !Object.hasOwn(progressStages, result.stage)) return;
  // A duplicate start notification must not restart the elapsed clock.
  if (session.progress?.id === result.progress_id) return;
  clearProgress(session);
  const progress = {id: result.progress_id, stage: result.stage, started: performance.now(),
    visible: true, longWait: false, timer: null};
  session.progress = progress;
  // State may already say SPEAKING while synthesis has not produced audio.
  // Show the actual stage immediately; delay only the ticking elapsed label.
  showVoiceStatus(session);
  function tick() {
    if (current !== session || session.progress !== progress || session.phase !== 'recording') return;
    const elapsed = Math.max(0, performance.now() - progress.started);
    const elapsedVisible = elapsed >= 1200;
    const longWait = elapsed >= 8000;
    const changed = progress.longWait !== longWait;
    progress.longWait = longWait;
    if (elapsedVisible) {
      progressTime.hidden = false;
      progressTime.textContent = `Waiting · ${Math.floor(elapsed / 1000)} s`;
    }
    // Announce the long wait once, never the ticking seconds.
    if (changed) showVoiceStatus(session);
    progress.timer = setTimeout(tick, 250);
  }
  progress.timer = setTimeout(tick, 1200);
}

function stopBot(session) {
  session.playbackId = null;
  session.incoming = null;
  if (session.botSource) { session.botSource.onended = null; session.botSource.stop(); session.botSource.disconnect(); session.botSource = null; }
}

async function releaseMicrophone(session) {
  stopBot(session);
  session.stream?.getTracks().forEach((track) => { track.onended = null; track.stop(); });
  session.source?.disconnect();
  session.node?.disconnect();
  if (session.node) session.node.port.close();
  if (session.context && session.context.state !== 'closed') await session.context.close();
}

async function fail(session, text) {
  if (current !== session) return;
  clearProgress(session);
  showProposal(null);
  display('Stopping', 'Releasing the microphone…', 'Stopping…', true);
  current = null;
  clearTimeout(session.timer);
  session.socket?.close();
  try { await releaseMicrophone(session); } catch (error) { console.error(error); }
  display(isVoice(session) ? 'Session stopped' : 'Could not record', text, 'Try again');
  if (session.recoveryOffer) offerRecovery(session.recoveryOffer);
  else if (session.resumeSent && !(session.resumeAvailable && !session.resumeAccepted)) clearRecovery();
  showRecovery();
}

function armTimeout(session, milliseconds, text) {
  clearTimeout(session.timer);
  session.timer = setTimeout(() => { void fail(session, text); }, milliseconds);
}

function asWav(chunks, samples) {
  // RIFF/WAVE header offsets are fixed by the file format.
  const dataBytes = samples * config.sample_width_bytes * config.channels;
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  const tag = (offset, value) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
  tag(0, 'RIFF'); view.setUint32(4, 36 + dataBytes, true); tag(8, 'WAVE');
  tag(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, config.channels, true); view.setUint32(24, config.sample_rate_hz, true);
  view.setUint32(28, config.sample_rate_hz * config.sample_width_bytes * config.channels, true);
  view.setUint16(32, config.sample_width_bytes * config.channels, true);
  view.setUint16(34, config.sample_width_bytes * 8, true);
  tag(36, 'data'); view.setUint32(40, dataBytes, true);
  return new Blob([header, new Blob(chunks).slice(0, dataBytes)], { type: 'audio/wav' });
}

function finishInput(session, samples) {
  if (current !== session || session.sentStop) return;
  clearProgress(session);
  showProposal(null);
  session.sentStop = true;
  session.phase = 'saving';
  display(isVoice(session) ? 'Ending session' : 'Saving', isVoice(session) ? 'Saving your conversation…' : 'Finishing your recording…', 'Finishing…', true);
  armTimeout(session, config.stop_timeout_ms, 'The server did not confirm the save. Please try again.');
  if (session.socket.readyState !== WebSocket.OPEN) {
    void fail(session, 'The connection closed before your recording could be saved.');
    return;
  }
  session.socket.send(JSON.stringify({ type: 'stop', samples }));
  void releaseMicrophone(session).catch((error) => { console.error(error); });
}

function requestStop(session) {
  if (session.phase !== 'recording') return;
  clearProgress(session);
  showProposal(null);
  session.phase = 'saving';
  display(isVoice(session) ? 'Ending session' : 'Saving', isVoice(session) ? 'Saving your conversation…' : 'Finishing your recording…', 'Finishing…', true);
  armTimeout(session, config.stop_timeout_ms, 'The microphone did not finish cleanly. Please try again.');
  session.node.port.postMessage({ type: 'stop' });
}

async function start(resume = null) {
  if (resume && (resume !== recovery || resume.expiresAt <= Date.now() || modeSelect.value !== 'demo')) {
    clearRecovery();
    return;
  }
  clearProgress(current);
  const session = { phase: 'starting', chunks: [], sentStop: false, mode: config.bank_target ? 'bank' : modeSelect.value,
    resumeToken: resume?.token || null, resumeSent: false, resumeAccepted: false, resumeAvailable: false };
  if (!resume) clearRecovery();
  recoveryPanel.hidden = true;
  current = session;
  showProposal(null);
  modeSelect.disabled = true; domainSelect.disabled = true; transcript.textContent = ''; slots.replaceChildren(); conversation.replaceChildren(); document.getElementById('conversation-empty').hidden = false; document.getElementById('order-empty').hidden = false;
  playback.hidden = true;
  recording.pause();
  recording.removeAttribute('src');
  recording.load();
  if (recordingUrl) { URL.revokeObjectURL(recordingUrl); recordingUrl = null; }
  display('Microphone access', 'Allow microphone access in your browser to begin.', 'Cancel');
  try {
    session.context = new AudioContext({ sampleRate: config.sample_rate_hz });
    await session.context.resume();
    if (current !== session) { await releaseMicrophone(session); return; }
    if (session.context.sampleRate !== config.sample_rate_hz) throw new Error('This browser cannot record at the required sample rate. Please use Chrome or Edge.');
    session.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: config.channels, sampleRate: config.sample_rate_hz,
        echoCancellation: config.echo_cancellation,
        noiseSuppression: config.noise_suppression, autoGainControl: config.auto_gain_control,
      }, video: false,
    });
    if (current !== session) { await releaseMicrophone(session); return; }
    await session.context.audioWorklet.addModule('/static/recorder-worklet.js');
    session.node = new AudioWorkletNode(session.context, 'darija-recorder', {
      channelCount: config.channels, channelCountMode: 'explicit',
      numberOfInputs: 1, numberOfOutputs: 1, processorOptions: { ...config, max_recording_ms: isVoice(session) ? (session.mode === 'demo' ? config.demo_max_session_ms : config.max_session_ms) : config.max_recording_ms },
    });
    session.node.onprocessorerror = () => { void fail(session, 'Microphone processing stopped. Please try again.'); };
    session.stream.getAudioTracks().forEach((track) => {
      track.onended = () => { void fail(session, 'The microphone was disconnected or its permission was removed.'); };
    });
    session.node.port.onmessage = (event) => {
      if (current !== session) return;
      if (event.data.type === 'frame') {
        if (session.socket.readyState !== WebSocket.OPEN || session.socket.bufferedAmount + frameBytes > frameBytes * config.max_buffered_frames) {
          void fail(session, 'The recording connection could not keep up. Please try again.');
          return;
        }
        if (!isVoice(session)) session.chunks.push(event.data.buffer);
        session.socket.send(event.data.buffer);
      } else if (event.data.type === 'finished') finishInput(session, event.data.samples);
    };
    const address = new URL(config.websocket_path, location.href);
    address.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    address.searchParams.set('domain', config.domain_id);
    address.searchParams.set('mode', session.mode);
    if (config.bank_target) { address.searchParams.set('bank', config.bank_target); address.searchParams.set('overwrite', String(config.bank_overwrite)); }
    session.socket = new WebSocket(address);
    session.socket.binaryType = "arraybuffer";
    armTimeout(session, config.connection_timeout_ms, 'Could not connect to the local server. Check that it is running, then retry.');
    session.socket.onerror = () => { void fail(session, 'Could not connect to the local server. Check that it is running, then retry.'); };
    session.socket.onclose = () => {
      if (current === session) void fail(session, 'The connection closed unexpectedly. Start a new session to reconnect.');
    };
    session.socket.onmessage = (event) => {
      if (current !== session) return;
      if (event.data instanceof ArrayBuffer) {
        if (session.incoming) session.incoming.chunks.push(event.data);
        return;
      }
      let result;
      try { result = JSON.parse(event.data); }
      catch { void fail(session, 'The server returned an unreadable response.'); return; }
      if (isVoice(session) && handleVoiceEvent(session, result)) return;
      if (result.type === 'ready') {
        const startMessage = { type: 'start' };
        if (session.resumeToken) { startMessage.resume_token = session.resumeToken; session.resumeSent = true; }
        session.socket.send(JSON.stringify(startMessage));
      }
      else if (result.type === 'started') {
        if (session.resumeToken && result.resumed !== true) {
          void fail(session, 'The saved details were not restored. Start over to begin a new conversation.');
          return;
        }
        if (result.resumed) { session.resumeAccepted = true; clearRecovery(); }
        clearTimeout(session.timer);
        session.source = session.context.createMediaStreamSource(session.stream);
        session.source.connect(session.node);
        // The processor emits silence to the speakers; the microphone is never monitored.
        session.node.connect(session.context.destination);
        session.phase = 'recording';
        if (isVoice(session)) display('Listening', 'Speak naturally. You can correct an earlier detail or interrupt the assistant.', 'End session');
        else display('Recording', config.bank_target ? config.bank_text : 'Say a short sentence, then stop to check playback.', 'Stop recording');
      } else if (result.type === 'saved') {
        clearTimeout(session.timer);
        current = null;
        session.socket.close();
        recordingUrl = URL.createObjectURL(asWav(session.chunks, result.samples));
        recording.src = recordingUrl;
        savedPath.textContent = result.bank_audio || result.audio;
        playback.hidden = false;
        display('Recording saved', 'Your WAV was saved on this computer. Play it below to check that your voice is clear.', 'Record again');
      } else if (result.type === 'error') {
        session.recoveryOffer = result.recovery;
        session.resumeAvailable = result.resume_available === true;
        void fail(session, result.message);
      }
    };
  } catch (error) {
    const friendly = {
      NotAllowedError: 'Microphone access was not granted. Allow it in your browser, then try again.',
      NotFoundError: 'No microphone was found. Connect one, then try again.',
      NotReadableError: 'The microphone is unavailable. Close other apps using it, then try again.',
    };
    await fail(session, friendly[error.name] || error.message || 'Could not start recording. Please try again.');
  }
}

button.addEventListener('click', () => {
  if (current?.phase === 'recording') requestStop(current);
  else if (current?.phase === 'starting') {
    const session = current; clearProgress(session); current = null; clearTimeout(session.timer);
    session.socket?.close(); void releaseMicrophone(session).catch(console.error); idle(Boolean(session.resumeToken && !session.resumeSent));
  }
  else if (!current) void start();
});
resumeButton.addEventListener('click', () => { if (!current && recovery) void start(recovery); });

window.addEventListener('pagehide', () => {
  if (!current) return;
  clearProgress(current);
  current.stream?.getTracks().forEach((track) => track.stop());
  current.socket?.close();
});

function addMessage(role, text) {
  document.getElementById('conversation-empty').hidden = true;
  const item = document.createElement('li'); item.dataset.role = role;
  const label = document.createElement('small'); label.textContent = role;
  const content = document.createElement('p'); content.dir = 'auto'; content.textContent = text;
  item.append(label, content); conversation.append(item); conversation.scrollTop = conversation.scrollHeight;
}

function idle(preserveRecovery = false) {
  clearProgress(current);
  showProposal(null);
  if (preserveRecovery !== true) clearRecovery();
  const demo = modeSelect.value === 'demo';
  const live = ['voice', 'demo'].includes(modeSelect.value) && !config.bank_target;
  document.getElementById('conversation-panel').hidden = !live;
  document.getElementById('demo-note').hidden = !demo || Boolean(config.bank_target);
  document.getElementById('demo-note-text').textContent = demoUI.note;
  document.getElementById('task-state-title').textContent = demo ? demoUI.state_title : 'Task details';
  document.getElementById('order-empty').textContent = demo ? demoUI.state_empty : 'Your details will appear here as you speak.';
  const issues = demo ? config.demo_issues : config.voice_issues;
  setup.replaceChildren();
  for (const issue of issues) { const item = document.createElement('li'); item.textContent = issue; setup.append(item); }
  document.getElementById('setup-panel').hidden = issues.length === 0;
  if (!config.bank_target) document.getElementById('session-title').textContent = demo ? demoUI.title : config.domain_title;
  if (window.isSecureContext && navigator.mediaDevices?.getUserMedia && window.AudioWorkletNode) {
    if (config.bank_target) display('Ready to record', config.bank_text, 'Record phrase');
    else if (demo && issues.length) display('Voice setup needed', issues.join(' '), 'Complete voice setup', true);
    else if (demo) display('Ready to talk', `Your microphone audio goes to MoulSot; transcripts go to Groq and spoken replies use ${demoSpeechProvider}. Use headphones for clearer interruptions. Each reply may take a few seconds.`, 'Start conversation');
    else if (modeSelect.value === 'voice') display('Ready to begin', 'Audio is sent to the configured ASR provider; transcripts go to Groq for slot routing. The assistant reads your details back before confirmation.', 'Start voice session');
    else display('Ready to record', 'Check your microphone with a short recording. This check stays on your computer.', 'Start recording');
  } else display('Browser unavailable', 'Use Chrome or Edge on localhost or HTTPS to enable microphone capture.', 'Microphone unavailable', true);
  if (preserveRecovery === true) showRecovery();
}
modeSelect.addEventListener('change', idle);
idle();

function showSlots(values) {
  document.getElementById('order-empty').hidden = renderTaskDetails(values, slots) > 0;
}

function renderTaskDetails(values, target) {
  target.replaceChildren();
  if (!values || typeof values !== 'object' || Array.isArray(values)) return 0;
  let rows = 0;
  function row(label, value) {
    const term = document.createElement('dt'); term.textContent = label;
    const detail = document.createElement('dd'); detail.dir = 'auto'; detail.textContent = value;
    target.append(term, detail);
    rows += 1;
  }
  if (demoStateKind === 'collection_scoped' && Array.isArray(values.items)) {
    values.items.forEach((item, index) => {
      const toppings = item.toppings === undefined ? 'toppings not specified' : item.toppings.length ? item.toppings.join(', ') : 'no extra toppings';
      row(`Pizza ${index + 1}`, `Quantity: ${item.quantity ?? 'not specified'} · ${item.size ?? 'size not specified'} · ${toppings}`);
    });
    if (values.drink) row('Drink', values.drink.map(value => value === 'none' ? 'No drink' : value).join(', '));
    return rows;
  }
  const displayValue = value => Object.hasOwn(demoValues, String(value)) ? String(demoValues[String(value)]) : String(value);
  for (const [key, value] of Object.entries(values)) {
    row(demoLabels[key] ?? key.replaceAll('_', ' '), Array.isArray(value) ? value.map(displayValue).join(', ') : displayValue(value));
  }
  return rows;
}

function showProposal(proposal) {
  const values = proposal?.state;
  const visible = values !== null && typeof values === 'object' && !Array.isArray(values);
  proposalPanel.hidden = !visible;
  proposedSlots.replaceChildren();
  document.getElementById('proposal-empty').hidden = true;
  if (visible) document.getElementById('proposal-empty').hidden = renderTaskDetails(values, proposedSlots) > 0;
}

function handleVoiceEvent(session, result) {
  if (result.type === 'state') {
    session.latestState = result.state;
    if (result.state === 'DONE') clearProgress(session);
    showSlots(result.slots);
    showProposal(session.phase === 'recording' && result.state !== 'DONE' ? result.pending_proposal : null);
    showVoiceStatus(session);
  } else if (result.type === 'progress') updateProgress(session, result);
  else if (result.type === 'transcript') addMessage('You', result.text);
  else if (result.type === 'assistant_text') addMessage('Assistant', result.text);
  else if (result.type === 'warning') message.textContent = result.message;
  else if (result.type === 'endpoint') { /* Recorded in the session log. */ }
  else if (result.type === 'audio_start') {
    stopBot(session);
    session.playbackId = result.playback_id;
    session.incoming = { id: result.playback_id, bytes: result.bytes, chunks: [] };
  } else if (result.type === 'audio_stop') {
    if (session.playbackId === result.playback_id) stopBot(session);
  } else if (result.type === 'audio_end') {
    const incoming = session.incoming;
    if (!incoming || incoming.id !== result.playback_id) return true;
    session.incoming = null;
    const blob = new Blob(incoming.chunks, { type: 'audio/wav' });
    if (blob.size !== incoming.bytes) { void fail(session, 'The response audio was incomplete.'); return true; }
    void blob.arrayBuffer().then(bytes => session.context.decodeAudioData(bytes)).then(buffer => {
      if (current !== session || session.playbackId !== incoming.id) return;
      const source = session.context.createBufferSource(); source.buffer = buffer;
      source.connect(session.context.destination); session.botSource = source;
      source.onended = () => {
        source.disconnect(); session.botSource = null;
        if (current === session && session.playbackId === incoming.id && session.socket.readyState === WebSocket.OPEN) {
          session.socket.send(JSON.stringify({ type: 'playback_finished', playback_id: incoming.id }));
          session.playbackId = null;
        }
      };
      source.start();
    }).catch(() => { if (current === session && session.playbackId === incoming.id) void fail(session, 'The assistant audio could not be played.'); });
  } else if (result.type === 'done') {
    clearProgress(session);
    showProposal(null);
    clearTimeout(session.timer); current = null;
    session.socket.close(); void releaseMicrophone(session).catch(console.error);
    showSlots(result.slots);
    const label = result.status === 'demo_completed' ? (config.domain_id === 'pizza' ? 'Demo order confirmed' : 'Preferences confirmed') : result.status === 'completed' ? 'Task confirmed' : result.status === 'handoff' ? 'Handoff requested' : 'Session ended';
    display(label, result.status === 'demo_completed' ? demoUI.completion_text : result.status === 'completed' ? 'Your confirmed details have been saved on this computer.' : 'This task was not marked as successfully completed.', 'Start another session');
  } else return false;
  return true;
}
