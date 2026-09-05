# Darija Voice — Current Build Report

> Latest September 6 code checkpoint: **1001 tests pass, 1 is skipped**. Pizza and
> a fictional clinic preference preview now share transactions, scoped dialogue,
> staged changes, cancellation and versioned confirmation. Both use the same
> MoulSot → Groq → Darija XTTS pipeline. A real-provider synthetic clinic diagnostic
> exposed an ambiguous doctor choice; a validated held draft now preserves the
> time and asks for the doctor. Complete human conversations and native accuracy
> remain unproven. The clinic preview does not check availability or book visits.
> Optional semantic readback grouping now preserves complete summaries while
> reusing unchanged fields. One real XTTS correction render took 2.258 s versus
> an earlier 8.315 s whole render under different load. Grouped playback was
> longer; default remains whole pending quality review. See the bounded
> [grouping experiment](bench/results/readback_grouping_20260905.md).
> Configured flat tasks can now hold clear details while one field is ambiguous,
> with matching-question resolution, cancellation and fresh readback. A real
> two-request draft diagnostic passes; earlier omissions and rejected output are
> retained in the [ambiguity evidence](bench/results/ambiguous_task_proposals_20260905.md).
> A dependent-choice diagnostic exposed a mismatch between the exported JSON
> Schema and local clarification validation. The conditional scope schema and
> bounded retry are fixed. Groq accepted the final inline conditional schema in
> one bounded English engineering case after rejecting a referenced union. See the
> [scope fix](bench/results/clarification_scope_fix_20260905.md).
> A later real-text probe exposed silent first-option selection for clinic and
> renamed counter fields. Generic choice guidance and a narrow configured-enum
> check now prevent the observed form; both failed cases pass on recheck with
> date/time held pending. This is not general Darija accuracy. See local
> [enum evidence](bench/results/enum_alternatives_20260905.md).
> Linked fields can now stay pending across multiple clarification answers.
> Choosing B from A/10 or B/11 holds B and asks time instead of inheriting old10.
> A real three-turn router check passes. One new actual MoulSot synthetic speech
> check omitted B; after a saved-transcript routing refinement it tracked both
> unresolved fields. The transcription limitation remains. See local
> [coupled-choice evidence](bench/results/coupled_choices_20260905.md).
> The supervisor now tolerates Windows status-file locks without stopping healthy
> speech services, while preserving real failure cleanup and STOP. Native Windows
> lock reproduction and lifecycle tests pass. See local
> [supervisor evidence](bench/results/supervisor_status_failure_20260906.md).
> A bounded vocabulary experiment restored a missing name token but failed its
> literal full-label gate; an exploratory negative control was unchanged. Optional
> per-task vocabulary transport is now available only for controlled local demos,
> with limits, forwarding acknowledgment, isolation and setup checks. All shipped
> tasks leave it off. Short-answer ASR/routing variability remains unresolved;
> static validation-stage telemetry and pending-state failure checks are added.
> See [follow-up evidence](bench/results/moulsot_vocabulary_followup_20260906.md).
> Explicit **Continue saved details** now restores validated committed values into
> a new unconfirmed demo after an eligible error. Pending changes are discarded;
> recovery lasts up to 30 minutes while the page and server remain open. Eleven
> browser checks use mocked microphone/socket events; see the local
> [recovery evidence](bench/results/session_recovery_20260905.md).
> A real transport test also exposed a WebSocket compression/size-limit disconnect
> before ASR. That production bug is fixed; the post-fix exchange delivers complete
> reply audio. Its runner found a separate normal-close cleanup race, tracked in
> [transport evidence](bench/results/transport_fix_20260905.md).
> Eight clinic replies and seven pizza replies are now cached. With network access
> blocked, each rendered locally in under 3 ms in one diagnostic. Dynamic replies
> still need synthesis. Concurrent cache misses share generation within a session,
> and cancellation/atomic publication are covered by independent regressions.
> Windows harness pacing averaged 32.098 ms over 125 frames in an offline check;
> this measures scheduling, not microphone or end-to-end latency. See the
> [cache and pacing milestone](bench/results/speech_cache_milestone_20260905.md).
> The latest shared fixes bound repeated-affirmation repair loops, require a fresh
> readback after explicit corrections, prevent stale speech frames from triggering
> later interruptions, and preserve session cleanup across caller cancellation.
> Proposed details are now shown separately while an answer is pending. Browser
> checks pass for desktop/mobile rendering and real WebAudio completion/interruption
> using synthetic input and mocked sockets. These are not human/device quality tests.
> See [conversation lifecycle evidence](bench/results/conversation_lifecycle_20260905.md).
> A new MoulSot-only timing diagnostic took 13.05 s, including 12.04 s waiting
> for the result. Hosted ASR latency remains variable; no ASR speedup is claimed.
> The UI now identifies the active stage immediately, shows elapsed waiting time
> after 1.2 s and a long-wait message after 8 s, with End session still available.
> Blocked-provider cancellation and stale-event cleanup pass across both domains.
> [Slow-turn evidence](bench/results/slow_turn_experience_20260905.md) includes real
> browser timers and synthetic AudioWorklet input; it is not a provider speed test.
> An optional hosted-function profiling patch is prepared and tested locally,
> preserving the baseline deployment. It has not been deployed and supplies no
> hosted timing measurements yet.
> Connection cleanup now covers partial startup, save/close failures and repeated
> cancellation. Real local WebSocket tests exposed and fixed a normal-disconnect
> error misclassification. Invalid voice-provider settings keep microphone capture
> available. See [connection recovery](bench/results/server_connection_recovery_20260905.md).
> Preview overrides now preserve field identities. An independent malformed-profile
> reproduction could previously drop the required doctor field and accept only
> date/time; loading now rejects that configuration. Both shipped profiles pass.
> See [profile identity evidence](bench/results/profile_identity_20260905.md).
> The new offline `check_config.py --demo` command checks merged speech mappings.
> Missing questions, incomplete readbacks and incompatible date/time formats now
> produce setup errors before speaking. See [preflight evidence](bench/results/demo_preflight_20260905.md).
> Invalid model output now uses bounded recovery instead of ending a demo session.
> Saved details and pending proposals survive, stale confirmation is invalidated,
> and valid corrections can continue. Provider outages remain terminal errors.
> See [router recovery evidence](bench/results/router_output_recovery_20260905.md).
> Supplied WAV manifests can now test several turns in one session, with explicit
> live execution and exact playback completion. Actual local TCP tests complete
> request/correction/confirmation in both domains using fixture providers and real
> frame pacing. See [multi-turn evidence](bench/results/multi_turn_replay_20260905.md);
> these are not live MoulSot or human accuracy results.
> A subsequent two-turn recorded-audio replay used actual MoulSot and XTTS and
> exposed a repeated-question bug: answering operations lacked the pending
> question ID. Shared router/state validation now rejects that mismatch before
> mutation. One saved-context Groq replay advanced to readback with the correct
> ID, still unconfirmed. See [pending-answer evidence](bench/results/pending_answer_identity_20260905.md).
> A read-only signal and local endpoint audit of five existing captures found no
> sample-rail clipping and no forced boundary in the documented long order.
> It does not justify gain or endpoint-threshold changes. See
> [capture audit](bench/results/capture_signal_audit_20260905.md).
> Explicit cancellation now clears retained recovery text while preserving saved
> fields. New questions receive their own request context; stale question/request
> IDs cannot clear newer work. Fourteen independent cross-domain tests and two
> bounded English Groq protocol controls pass. These do not establish Darija
> cancellation recognition. See [history cancellation](bench/results/retained_request_cancellation_20260905.md).
> Additional flat domains now load from matching configuration files, with no
> engine registry edit. A temporary service-counter fixture proves configured
> fields, schema, corrections, readback and browser discovery. It also exposed
> and fixed a hidden pizza-rendering assumption for flat fields named `items`.
> See [extension evidence](bench/results/custom_domain_extension_20260905.md) and
> [adding a task](docs/ADDING_A_TASK.md). This is a local English diagnostic,
> not a third deployed voice demo or native speech result.
> AUTONOMOUS_WORK.md records the ongoing campaign and next checks.

The latest latency experiment runs a verified quantization of MoulSot locally on
the RTX 4050 laptop, without changing the engine's Python pins. One GPU API
request took 1.586 s; one request through the actual engine ASR adapter and the
optional JSON bridge took 1.365 s. The same 7.207 s research clip took 5.879 s in
the hosted diagnostic. These are individual measurements, not averages or native
accuracy results. The supervised demo now uses local MoulSot while preserving
the saved hosted configuration. A recorded greeting produced real replies in
pizza and clinic; clinic additionally passed the live TCP WebSocket path. ASR
took 0.483 s and 0.766 s respectively, with cached XTTS replies. These single
exchanges do not prove complete conversations or native task accuracy. See
[local ASR evidence](bench/results/local_moulsot_feasibility_20260905.md) and
[live voice evidence](bench/results/local_voice_live_20260905.md).

Valid restatements after rejecting a complete summary now recover the shared
repair budget. Unchanged removals, unresolved questions and unrelated missing-field
answers cannot bypass the retry limit. See
[repair recovery evidence](bench/results/repair_budget_restatement_20260905.md).

The previous repeated drink question now advances to completed readback in an
actual two-turn recorded-audio replay; it still awaits a separate confirmation.
See [two-turn recovery](bench/results/local_two_turn_recovery_20260905.md).
The replay tools also now support config-added task domains, closing a mismatch
with the engine's extension guide. See
[configured-domain replay](bench/results/configured_domain_replay_20260905.md).

Local ASR performance is not consistently fast: a new synthetic clinic request
and a short correction both hit the 30-second bridge timeout. A bounded model
diagnostic completed with normal text but slow generation. The app now identifies
timeouts and service failures accurately instead of blaming configuration/quota.
See [clinic evidence](bench/results/clinic_sequence_20260905/README.md) and
[failure reporting](bench/results/asr_failure_messages_20260905.md). A complete
clinic human conversation is still unproven. A later CPU run exposed eleven
misrouted as08:00. Shared draft numerals and a conservative time-consistency
guard now address that failure. The final checked synthetic three-turn audio
sequence sets10:30, changes only the time to11:00, completes fresh readback and
confirms separately. Every draft expected state matches; actual MoulSot/Groq/XTTS
were used. See [final clinic evidence and listening excerpt](bench/results/clinic_sequence_20260905/README.md).
Uncached corrected-readback synthesis took8.315s and is the next measured delay
to improve. These synthetic inputs do not establish native accuracy.

> September 5 update: the router now defaults to GPT-OSS 120B, uses strict JSON
> Schema and checks model availability at startup. All three requested candidate
> models passed a live protocol smoke. The 30-case owner-fillable eval and
> MoulSot-only kitchen runner are added; 66 tests pass and one is skipped. See
> [TESTING_NEXT.md](TESTING_NEXT.md). The September 4 snapshot below is historical.

**Snapshot date:** September 4, 2026  
**Status:** Software implementation through M10 is present. Real voice validation and research measurements remain pending.

## 1. What we built

Darija Voice is a browser-based task-completion engine for Moroccan Darija. It collects structured information from a conversation, allows corrections, reads the collected details back, and requires confirmation before completing the task.

Multiple people contribute to one shared task state. The engine does not identify individual speakers or separate overlapping voices.

Two domains are configured:

- **Pizza ordering:** size, toppings, quantity, drinks, phone number, and address.
- **Clinic appointments:** doctor, reason, date, time, patient name, and phone number.

The research tooling is called **The Family Order Test**. It evaluates task completion and endpoint timing, rather than relying solely on transcription spelling accuracy.

## 2. Application and microphone capture

- FastAPI serves a plain HTML/JavaScript interface with pizza/clinic selection.
- A microphone-check mode records locally without needing an ASR service.
- AudioWorklet captures mono audio at 16 kHz and sends PCM16 audio over WebSocket in 32 ms frames.
- The server saves WAV recordings and JSON metadata on disk.
- Final-frame padding is trimmed to preserve the captured sample count.
- Browser playback lets the user inspect the captured audio.
- Recording limits, transport limits, and timeouts come from configuration.
- Voice-task mode displays setup requirements and remains unavailable until its checks pass.

The interface includes session status, recognized transcript text, and current task details. It was inspected in the browser, including domain switching; responsive layout checks were also performed during development.

## 3. Voice engine

### Endpointing

Silero VAD detects speech. A base silence wait can be extended by hesitation markers, continuation markers, or an unfinished digit sequence.

The implementation includes resumed-speech handling, short-noise rejection, a wait cap, a maximum segment length, and cached partial transcript requests. Transcription runs outside the frame-processing loop.

These rules are implemented and tested; improvement on real Darija conversations has not yet been measured.

### Speech recognition and routing

- Hosted MoulSot adapters support its Gradio API and a documented JSON endpoint contract.
- Groq `whisper-large-v3-turbo` provides the fallback ASR path.
- A persistent quota ledger reserves requests before sending them and applies the specified rolling limits across restarts.
- Missing ASR confidence is recorded as unavailable. Groq's derived confidence is labeled as a probability proxy.
- A Groq-hosted LLM returns JSON slot operations only; it does not generate spoken responses.
- Router responses are validated with Pydantic, including slot names, value types, and enum membership.
- Invalid responses are retried once. Invalid batches are never partially applied.

Local MoulSot loading is **not available with the locked dependency versions**. Its Qwen3-ASR runtime requires an additional package/runtime outside that list. The current implementation therefore uses hosted adapters and reports the local compatibility limitation explicitly.

### Normalization and task state

Normalization handles French number expressions, Unicode decimal digits, Moroccan mobile-number formats, dirham aliases, and configured date/time expressions. Darija numeral normalization accepts an owner-supplied dictionary; the reviewed language entries are still missing.

State operations support `set`, `add`, `remove`, and `clear`. Updates are atomic, existing values can be corrected, and missing required slots determine the next question.

Confirmation is tied to the version of the state that was read back. A correction, negation, or interruption invalidates stale confirmation. Filling every required slot alone does not complete a task.

### Voice output and interruption

- An audio-bank recorder uses the existing browser microphone path.
- It refuses configs containing unreviewed language or unresolved placeholders.
- Readbacks concatenate human WAV fragments with the configured 120 ms gaps.
- Phone numbers are read digit by digit; dates and times use recorded digits and separators.
- A restricted TTS interface is implemented for addresses and patient names. No TTS service has been selected or deployed.
- Playback IDs and acknowledgements prevent interrupted audio from being treated as a completed readback.
- Three consecutive speech frames trigger barge-in, retaining the first 96 ms of incoming speech.

### Unusable audio and handoff

The engine flags unusable input when two of three signals fire: low ASR confidence, low token rate, or high energy variation. It asks for clarification and skips routing that segment.

Following Section 6.5 of the specification, the fourth unusable segment triggers handoff after three clarifications. Interrupting the assistant is handled separately from detecting overlapping human speech.

## 4. Benchmark and research tooling

The repository contains **20 pizza scenario stubs and 10 clinic scenario stubs**. They are templates awaiting real recordings and manual annotations, not a completed dataset.

Implemented metrics:

| Metric | What it measures |
|---|---|
| Task Success Rate (TSR) | Exact canonical final-state match in a completed, confirmed session |
| False Cutoff Rate (FCR) | Premature endpoints per gold utterance |
| Correction Success Rate (CSR) | Whether final values reflect annotated corrections |
| Critical Entity Accuracy (CEA) | Exact accuracy for critical slots, individually and as a mean |
| Median Endpoint Latency (MEL) | Median delay between matched gold utterance ends and endpoints |

Reports also include missed boundaries, endpoint coverage, unmatched endpoints, and synthetic output fraction. Undefined metrics remain N/A.

The runner validates annotations and audio, replays recordings through the voice pipeline at real time, and records configuration and source/data hashes. Gold transcripts and speaker labels are not supplied to the engine.

The sweep implementation evaluates **seven silence thresholds × four hold configurations = 28 runs**. It can produce a CSV, an FCR-versus-MEL SVG chart, four baseline comparisons, and measured configuration tuning. Benchmark ASR fallback is disabled so provider labels remain accurate.

**No real benchmark scores, measured tradeoff chart, or tuned endpointing configuration have been produced.** Missing data causes an explicit NOT RUN result. Replay limitations and in-sample tuning conventions are documented in README.md.

## 5. Verification completed

The latest recorded test run reported:

```text
52 passed, 1 skipped in 0.70s
```

The skipped test requires native-speaker-reviewed Darija numeral examples.

Other checks completed during development:

- Both domain configs validate.
- Python modules compile and browser JavaScript passes syntax checks.
- Installed direct dependency versions match the specification.
- The original specification, dependency list, and environment template remain preserved.
- The installed Silero model processes a PCM silence frame.
- Synthetic worklet audio survives WebSocket transmission and WAV saving byte for byte.
- Offline tests cover routing, atomic updates, confirmation, readback assembly, barge-in, clarification/handoff, quota accounting, metrics, and sweep selection.
- A mocked-provider WebSocket session completes greeting → slot updates → acknowledged readback → affirmation → persisted completion.
- A clinic fixture uses the same generic state and readback implementation.
- The live server renders both domains and refuses incomplete voice setup.

Mocked and synthetic checks establish software behavior. They do not establish live ASR accuracy, microphone quality, realistic overlap recovery, or research performance.

## 6. Main implementation files

| Location | Responsibility |
|---|---|
| `engine/server.py` | HTTP/WebSocket application and session lifecycle |
| `engine/pipeline.py` | Capture persistence and voice-session orchestration |
| `engine/endpointing.py` | Silero integration and endpoint rules |
| `engine/stt.py` | Hosted ASR adapters, confidence provenance, and quotas |
| `engine/normalize.py` | Numbers, phones, currencies, dates, and times |
| `engine/router.py` | JSON-only LLM routing and response validation |
| `engine/state.py` | Atomic slot operations and versioned confirmation |
| `engine/responder.py` | Reviewed WAV bank, readback, and restricted TTS |
| `engine/overlap.py` | Unusable-audio heuristics |
| `engine/config.py`, `engine/lexicon.py` | Configuration validation and supplied language markers |
| `configs/` | Schema and pizza/clinic domain settings |
| `web/` | Browser interface, microphone worklet, and playback |
| `scripts/` | Config checking and voice-bank recording |
| `bench/` | Scenario stubs, metrics, replay, and sweep tooling |
| `tests/` | The four specified offline test files |

## 7. What remains before a real demo or research release

1. Configure the Groq key and the chosen hosted ASR provider.
2. Review Darija phrases, marker lists, numeral mappings, and all review flags.
3. Supply clinic doctor/reason choices and matching audio fragments.
4. Record the human WAV banks and configure the restricted TTS endpoint.
5. Validate real microphone capture and corrected voice tasks in both domains.
6. Collect consented recordings and manually annotate all benchmark scenarios, including reference dates and corrections.
7. Run the measured benchmark, baselines, and endpointing sweep.
8. Prepare the real demonstration and failure analysis before making performance claims.

The project retains Python 3.11 and the locked dependencies. It includes no database, telephony integration, speaker diarization, or voice separation. No external publication or dataset upload has been performed.

## 8. How to describe the current progress

> We built a configurable Darija voice task engine with shared task state, correction-aware confirmation, interruptible audio responses, and a benchmark harness. The implementation has 52 passing offline tests. We are preparing reviewed voice assets and real conversational recordings to validate it and measure the endpointing tradeoff.

Credit Atlasia/MoulSot as the underlying ASR. The contribution being evaluated is task handling and endpointing around ASR; a claim of measured improvement is still pending.
