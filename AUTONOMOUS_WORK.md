# DarijaVoice ongoing improvement campaign

User authorization (2026-09-05): research, implement, and test during the user's
holiday; resume every 5 minutes (updated from 30) until the user explicitly says stop. Use agents
for independent research/ideas, implementation, and validation. Keep this file
as the durable queue: move completed items to the log and pick the next item.

Active automation: `darijavoice-research-and-improvement`, current-task heartbeat,
every 5 minutes, no end date. Computer and desktop app must remain running.

Latest user direction: build a reusable Darija voice task engine, not a pizza
product. MoulSot must power the actual speech recognition. The user explicitly
requested persistent work: do not stop after one small fix or nine minutes to
wait for a schedule. Continue to the next meaningful step while work is possible;
the five-minute heartbeat resumes work after a turn ends, not a work-duration cap.

## Current priorities

0. **Objective:** a reusable Darija task engine. Pizza and fictional clinic share
   transactions, clarification, correction and versioned confirmation. Config-added
   flat tasks now work in both the UI and supplied-audio replay. Keep the actual
   MoulSot recognizer and selected Darija XTTS voice; native multi-turn kitchen
   accuracy still needs reviewed human recordings and the owner-supplied cases.

1. **Speech findings and next boundary:** the saved short answer was reused after
   fixing the supervisor crash. Actual MoulSot returned `الطبيب باع`; the first
   router attempt was rejected, while ONE saved-text recheck held B and asked time.
   No behavioral routing fix occurred between those attempts. Do not repeat it
   until a new observed failure/change justifies investigation. Static validation
   stages now classify future rejections; offline pipeline checks preserve coupled
   drafts, current question identity and original requests through rejected answers.

   A frozen four-call local vocabulary experiment restored the missing B token,
   but its exact whole-label gate failed because articles were omitted. A separately
   reported exploratory negative control was unchanged. New optional per-task
   vocabulary transport is implemented and remains OFF in every shipped config.
   One additional live check of the newly implemented adapter/bridge path verified
   context forwarding/hash acknowledgment (1.410s), using saved negative-control
   audio. It was a transport integration check, not another accuracy experiment.
   Evidence: bench/results/moulsot_vocabulary_followup_20260906.md and
   context_bridge_live_22fb1c415702419caeea97ae8bd4d7ea.json. Native benefit remains
   unproven. No more synthetic prompt searches or repeated green context calls.

   **Next independent offline step:** implement the first contract slice from
   bench/results/configured_collections_design_20260906.md. The design audit is
   complete: transactions.py already supports arbitrary collections, but the
   surrounding collection dialogue/router/render/recovery remains pizza-specific.
   A fictional equipment collection with asset/quantity and root return_date is
   blocked by the explicit constructor/profile guards. Add an explicit separate
   configured_collection_scoped adapter/factory in bounded stages, retaining the
   existing pizza path. Start with profile validation and state/kernel fixtures
   under original and renamed IDs, then dialogue/render/recovery. Preserve row
   identity, atomic updates, draft isolation, corrections and fresh confirmation.
   Cross-row coupled dependencies require addresses, so do not silently reuse
   flat coupled_slots. Do not copy pizza logic, alter existing successful voice
   scripts, or claim a shipped third voice demo from fixtures.
   Native speech quality still needs owner-supplied reviewed cases/recordings.

   Completed: explicit coupled_slots in flat clarification, atomic staged answers,
   fresh next-question IDs, original request retained until group completion,
   committed-only recovery and unresolved values omitted from pending preview.
   Real Groq three-turn English check passes B held -> time11 -> B11 readback,
   unconfirmed. 32 independent coupled tests and8 mocked-browser checks pass.
   Evidence: bench/results/coupled_choices_20260905.md. This closes the declared-
   dependency state gap; the model can still omit dependency metadata.

   Completed: the remaining enum alternatives tests exposed silent A selection
   in clinic/custom fields. Generic choice guidance and a conservative exact-enum
   guard corrected the two failing forms. All original seven case contracts have
   passed at least once across separate revisions; not one consistent evaluation
   run or native accuracy. Do not rerun passing cases.
   Previous coupled milestone:10 Groq requests,1 MoulSot ASR,1 synthetic input
   render, cached replies. September6 continuation:6 actual local ASR calls,
   2 Groq calls,0 new TTS. No purchases, model downloads or autonomous publication.

2. **Recovery increment completed:** eligible demo errors offer Continue saved
   details for committed values, with pending changes explicitly discarded and
   fresh readback/confirmation required. Same-process, single-use, 30-minute
   recovery is implemented for clinic and pizza. Faithful held-draft or restart
   recovery would require a new versioned checkpoint contract; do not import
   arbitrary saved reports. See bench/results/session_recovery_20260905.md.

3. **Quality evidence remains open:** reviewed human kitchen conversations and
   owner-supplied native cases are still absent. Optional XTTS semantic grouping
   has saved comparison audio; whole readback remains default until pronunciation
   and joins are reviewed. Do not regenerate passing synthetic sequences or
   tune VAD/gain/timeouts without concrete new evidence. Improve shared task
   behavior from observed failures, with each next step bounded and recorded.

## Current runtime and completed checkpoint

- Main supervisor: deploy/local-moulsot/run_local_demo.py, refreshed hidden with
  --start --hosted-fallback --device cpu at 2026-09-06T00:25:34 Casablanca,
  supervisorPID29736 (launcher wrapper23052).
  Both pages return200, no setup issues, MoulSot on this computer and darija_xtts;
  bench/results/context_refresh_20260906.json records the no-inference checks.
  Context is absent from both shipped task configs. Status-file sharing failures
  now receive bounded retries and independent best-effort publication; they cannot
  kill healthy services. Check updated_at because locked snapshots can be stale.
  The old supervisor22940 failed on WinError5 status replacement; failed snapshot
  and native Windows lock reproduction are preserved in
  bench/results/supervisor_status_failure_20260906.md. Exact lock owner unknown.
  Whole readback remains default; conditional clarification schema, bounded retry
  explicit enum guard, coupled-field staging and saved-detail recovery are active.
  Latest complete suite1001passed/1native-review skip (30.14s), including46 context
  transport/preflight cases, native Windows sharing semantics and coupled-error
  preservation. An initial full run exposed an overly exact timing assertion in
  a transport fixture; deterministic pacing and delayed-ACK controls corrected
  the test without altering production sender behavior. Earlier11 browser
  recovery checks with mocked microphone/socket events. Read
  .local/moulsot/supervisor.json for current status/PIDs; selected_device must
  remain cpu on controlled refresh unless measurements justify a change. The
  launcher's omitted-device default is still cuda. Actual ASR is pinned MoulSot
  Q4_K_M plus Q8 audio projector, not base Qwen or Whisper; .env stays hosted.
- To refresh, create .local/moulsot/STOP, verify stopped state and ports8000/8011/8012
  free, remove only that marker and relaunch hidden in CPU mode. Do not kill an
  individual child: that triggers one hosted recovery. Never stop other user apps.
  Preflight hashes assets/runtime and can take tens of seconds. CPU requires
  3072MiB start RAM/768MiB runtime; its guards do not query or claim GPU memory.
  Optional experimental port8013 is now free; no duplicate model workers.
- Independent agents completed their bounded tasks. Check status before reuse.
- Recorded pizza two-turn replay now resolves questionID1 and reaches complete
  readback, unconfirmed. Evidence: bench/results/local_two_turn_recovery_20260905.md.
  The supplied excerpts alter original timing and have unknown exact human intent.
- GPU clinic attempts timed out; a CPU alternative completed the short correction
  in2.031s. First CPU full transport revealed eleven routed to08:00 and accepted
  by scripted yes. Draft numeral normalization and conservative time consistency
  guard fixed that observed case, including proposed times and safe retry.
- Final checked clinic three-turn replay PASSED expected states/actions through
  actual CPU MoulSot, Groq and Darija XTTS: doctorA/2026-09-08/10:30, change only
  time to11:00, fresh readback, separate confirmation. Report:
  bench/results/voice_transport_a0b25bf0ab4349a59c1b6313c100857a/report.json.
  Saved session8f769b9529d44684b0cfb8b12f71c07d is demo_completed/confirmed=true.
  Inputs and replies are synthetic; device/microphone/native listening untested.
  Do not rerun this passing sequence without a new change/failure.
- Full evidence, failed attempts and an18.533s edited listening excerpt:
  bench/results/clinic_sequence_20260905/README.md. Excerpt removes waits and is
  not latency evidence. Full run66.772s includes playback and source speech.
- Earlier clinic checkpoint:719passed/1skipped,14 known warnings,41.16s. The skipped
  native-reviewed number fixture and base research gates remain pending.

- The user explicitly requested a GitHub snapshot; it was pushed to
  Hamzaafroukh321/MoroccanDarijaIAVoice main atfdec1fe. An earlier clean source-only export
  passed842/1skip after LF checksum correction. Later autonomous changes remain
  local unless separately authorized for publication; no automatic force pushes.

## Operating rules

- Workspace: the DarijaVoice project root (current task working directory).
- Local managed Python: %USERPROFILE%/.codex/environments/darija-voice/Scripts/python.exe.
  Other machines should use a Python 3.11 virtual environment as in README.md.
- Preferred demo voice: Darija XTTS 2.1, author default speaker, HF Space
  medmac01/Darija-Arabic-TTS. MoulSot ASR remains the premise; no silent Whisper
  substitution. Groq router currently openai/gpt-oss-120b.
- Check active agents/processes before starting work. Do not launch duplicate
  servers or concurrent workers that edit the same files. Use a bounded research
  agent and independent validation agent only when useful work can run alongside.
- Keep edits reversible and local. No purchases, payment info, new paid resources,
  external messages, deployment/publication or automatic usage-reset redemption.
- Never print keys. Respect provider quotas; use saved traces/mocks first and
  small live checks only when needed. A scheduled wake is not permission to call
  every provider every time. Back off on quota/network limits without retry loops.
- Keep the main dependency pins and reviewed research mode intact. Scope demo
  improvements explicitly. Label synthetic audio and any new draft Darija; do
  not pretend it is native-reviewed. User will supply the 30-case gold router
  eval; do not replace it with generated cases or make performance claims from
  self-authored fixtures/synthetic audio.
- XTTS base/model output license is non-commercial; retain the license link.
- Finish meaningful steps, run relevant checks, record evidence and update the
  queue, then continue immediately to the next useful step when possible. Do not
  end merely to wait for a scheduled wake. Do not rerun passed checks or create
  busywork merely to fill time. Pause only for user stop, a real external/input
  blocker, or runtime/usage constraints; make independent progress when blocked.
- If blocked on user-only recordings, language review, payment or external quota,
  continue other independent work. Record exact blockers and remain quiet while
  unchanged. Do not claim continuous execution if app/host/usage is unavailable.

## Evidence and baseline

- Historical clinic checkpoint: **719 passed, 1 skipped**, 14 existing warnings, 41.16 s.
  Config-added domains now validate in supplied-audio replay; twelve independent
  cases close the old pizza/clinic-only harness limitation. Static ASR failure
  messages and status categories distinguish timeouts, busy/rate limits and
  service errors, with eighteen new regressions. Ledger failures are separate
  from verified budget exhaustion. Read associated evidence in bench/results.
  Both local demo pages return 200, report MoulSot on this computer and Darija
  XTTS, and have no setup issues. Final supervised refresh ready/local/CPU at12:58
  Casablanca; still ready at13:04. Independent agents completed their bounded
  tasks. Next priority is uncached TTS delay and the uncovered scoped clarification
  branch above. Earlier runtime PIDs and next-step statements below are history.
- 2026-09-05: local MoulSot is integrated into the supervised live demo. Both
  domains have recorded-greeting reply evidence, with an actual TCP WebSocket
  clinic exchange and complete reply WAVs. A shared 512-token output ceiling
  replaces the short probe's 128-token limit for uploads up to 20 seconds;
  completed long output and truncation rejection are covered offline. This
  establishes output handling, not long-audio recognition quality. Current live
  state is in .local/moulsot/supervisor.json; earlier PIDs below are historical.
- 2026-09-05: accepted restatements of a rejected complete summary now reset the
  shared repair budget while still requiring a fresh readback. Independent audit
  caught no-op clears/removes, missing-field restatements and allocator-only
  create/delete batches; they remain bounded. Final full suite568passed/1skipped.
  Evidence: bench/results/repair_budget_restatement_20260905.md. Running server
  refreshed: listener3180, wrapper33604; server_restatement_20260905_115136 logs.
  Both domain pages200 and Groq startup model-list check passed. .env unchanged.
- 2026-09-05: one public human-reference Moroccan Casablanca sample processed by
  actual hosted MoulSot in5.879s. Original/converted hashes and provenance checked,
  literal differences retained; unknown training overlap, research-only audio,
  no native accuracy claim. Local quantized MoulSot then loaded successfully and
  completed a GPU API request in1.586s on the same sample. See linked evidence
  above; no hosted retry loop or production provider replacement.

- 2026-09-05: Darija XTTS integration completed; 90 tests pass, 1 language fixture
  skipped. Live recorded-greeting transport passed; no full order proven by it.
- Real failure evidence: bench/results/demo_session_d0a4e4c564834d4e999086eddf355e46.json
  and demo_session_c7e29d7e525340baa3f47e43480bff61.json.
- Initial research agent output: bench/results/conversation_research_20260905.md.
- Initial validation agent output: bench/results/regression_audit_20260905.md.

## Completed work log

- 2026-09-06: fixed supervisor observability failure without hiding actual service
  errors; restored the local CPU demo; reused the failed transport attempt's
  synthetic audio; recorded ASR/routing variability; tested symmetric vocabulary
  with a frozen negative control. Implemented opt-in local per-task vocabulary,
  bounded config/wire formats, matching acknowledgment, setup rejection, research
  gating and isolated telemetry. All defaults remain off. Full1001/1skip and one
  actual integration request pass. Evidence linked above; no native accuracy claim.

- 2026-09-05 retained-history cancellation: separate request IDs allow explicitly
  forgetting resolved recovery text while preserving committed data. Old question
  IDs remain stale; cancellation requires fresh readback before confirmation.
  New questions now receive their own text/ID instead of inheriting a resolved
  earlier request. Existing unresolved questions retain their original context.
  Fourteen independent cross-domain tests pass. Two bounded English Groq controls
  (0.953s pizza/0.797s clinic) returned correct cancellation IDs and preserved saved
  fields; no ASR/TTS or Darija accuracy claim. Evidence:
  bench/results/retained_request_cancellation_20260905.md.
- Reusability extension: additional flat domains load from matching safe-ID base
  and preview config files; no engine registry edit. Domain selector discovers
  valid configs and preserves mode. Startup ignores malformed/unmatched extra
  configs consistently; direct filename/ID mismatch is rejected. Profile UI is
  validated, custom collection flags fail clearly. A flat field named items had
  triggered pizza readback/renderer assumptions; explicit state-kind dispatch now
  fixes it. Nine independent temporary third-domain tests and real-browser
  controlled rendering/navigation pass. No extra shipped domain or provider
  calls. Guide: docs/ADDING_A_TASK.md; evidence:
  bench/results/custom_domain_extension_20260905.md.
- Latest full suite544passed/1skipped/14knownwarnings in30.63s; JS syntax and
  offline clinic preflight pass. Skipped native-number fixture remains owner-held.
  Test browser closed. Current listener1668/wrapper13928; logs
  bench/results/server_domains_20260905_112349.stderr.log and .stdout.log.
  Startup model-list validation passed and both domain pages returnedHTTP200.
  This is startup evidence, not another provider audio conversation. The single
  five-minute heartbeat remains active and unchanged.

- 2026-09-05 current run: inspected all five existing captures without changing
  them. Added reusable offline signal/local VAD diagnostics with13 tests. No
  rail clipping; the documented long order produced two unforced endpoints.
  No evidence justified amplification or threshold changes. See
  bench/results/capture_signal_audit_20260905.md and capture_endpoint_audit_20260905.json.
- Replayed two original audio excerpts through actual MoulSot/Groq/XTTS once:
  two ASR calls (4.117s/5.530s), three router attempts, one new56-character XTTS
  generation. WAV delivery passed; order remained unconfirmed. Found matching
  answer ops committed while their question ID stayed unresolved, repeating the
  exact prompt. Original evidence preserved in recorded_two_turn_20260905.md.
- Fixed pending-answer identity in the shared dialogue and router, using each
  adapter's scope matcher. Missing/stale IDs reject atomically and use existing
  bounded router retry; correct IDs require fresh readback. Eighteen independent
  regressions pass across both domains. One router-only saved-context check took
  1.844s, returned the correct ID and advanced to unconfirmed readback. No ASR/TTS
  rerun, no native accuracy claim. Full suite521passed/1skipped in38.96s.
  Evidence: bench/results/pending_answer_identity_20260905.md.
- Restarted the validated local demo: listener18744, wrapper41308; logs
  bench/results/server_identity_20260905_105229.stderr.log and .stdout.log.
  Startup model-list check passed; both domain pages returnedHTTP200. These are
  startup checks, not a new audio conversation. Schedule remains one active
  five-minute heartbeat with persistent-work instructions and no end date.
- Independent first-turn proposal audit found no missing staging instruction:
  repeated pizza wording has uncertain item association. Original request is
  retained in later context, and partial proposals are supported. Do not force
  an invented order or spend quota replaying this unchanged ambiguity.
- Previously completed preflight, profile identity, slow-stage feedback,
  cancellation/cleanup and manifest replay remain documented in BUILD_REPORT.md
  and their individual evidence files. Do not rerun unchanged green checks.
- Local hosting remains unproven: latest read-only snapshot RTX4050 6GB had
  ~3.15GB free VRAM, below published MoulSot weight size before overhead; RAM
  ~4.53GB free. No unrelated app was stopped and no model/package installed.
  Earlier feasibility analysis remains deploy/moulsot-space/LOCAL_FEASIBILITY.md.
  Hosted profiling patch remains optional and undeployed. ASR result-wait spans
  queue/network/inference and cannot be attributed solely to queueing.

- Campaign initialized; immediate investigation running. Automation ID and
  subsequent changes will be recorded here after scheduler confirmation.
- 2026-09-05 initial run completed: two independent agents audited session traces
  and conversational repair sources. Code adds demo-only typed intents, a guard
  against classified mixed-item flattening/confirmation, missing-slot repair,
  three-failure cap, progress-sensitive counter reset and explicit router-failure
  errors. HTTP 429 stops without an immediate retry. Research mode unchanged.
- Verification: **99 passed, 1 skipped**. Two live Groq transcript replays passed
  (existing mixed-pizza request rejected without mutations; existing greeting
  requests size). Evidence: bench/results/demo_repair_live_regression_20260905.json.
  No full multi-item order, real microphone usability or kitchen-ASR score proven.
- Live check found and fixed an overly strict non-task validator: the model can
  correctly mark a clear but unsupported mixed-item request unclear=false. No-op,
  no-confirmation guarantees remain required. A 429 during diagnosis was respected.
- Next implementation should use the research/validation agent reports above;
  don't repeat the same provider probes. Broader regression tests can use traces
  and mocked providers first. Restart the app only after relevant checks pass.
- 2026-09-05 next run: shipped isolated demo order schema v2. Stable monotonic
  pizza IDs, atomic create/edit/delete batches, per-item quantity/size/toppings,
  order drinks, item-aware questions/readback/UI, deep snapshots and contextual
  router input. Main research TaskState and strict dependencies remain unchanged.
- Independent review caught and fixed unknown-topping removal inventing a plain
  pizza and unrelated drink edits clearing item ambiguity. Ambiguity now blocks
  bare-yes acceptance/stale callbacks until an original item changes (or first
  items are supplied for an empty order). No live provider calls by agents.
- Verification: **142 passed, 1 skipped**, Node syntax check and DOM rendering
  diagnostic passed. The skipped native-number fixture is still user-owned.
  Mocked full voice pipeline covers two pizzas, correcting second with yes,
  fresh completed readback, then separate confirmation; not microphone evidence.
- Live diagnostics: clean English two-item control passed (1.281s router call).
  Original garbled Darija and later explicit Darija both returned ambiguous/no
  changes. Two HTTP400 JSON-validation failures occurred at the original budget;
  demo now uses 2048 output tokens and GPT-OSS low reasoning, after checking Groq
  primary docs. A subsequent request produced valid JSON, but remains ambiguous;
  token exhaustion was NOT proven as the original 400 cause. No accuracy claims.
  Reports: multi_item_live_control_20260905.json, multi_item_live_replay_20260905.json,
  multi_item_live_explicit_20260905.json, multi_item_view_check_20260905.json.
- Tools for targeted reuse: `python -m bench.multi_item_replay` (one real request;
  flags --explicit / --synthetic) and `node bench/check_order_view.cjs` (offline).
  Do not rerun all live probes every wake. Draft phrases added for item labels,
  plain toppings and ambiguity; synthetic TTS remains Darija XTTS. Full multi-item
  live audio conversation and TTS latency still need validation.
- App restarted after checks; old logs archived in bench/results. Single existing
  half-hour heartbeat stays active and uses the updated priorities above.
- 2026-09-05 05:02 UTC run: introduced typed scoped clarifications for item target,
  order details, unsupported bottle size/drink/menu, and unintelligible speech.
  Spoken repair can offer the configured choices for the affected field. A bare
  yes, wrong item/field, stale pending ID or unrelated new clarification cannot
  resolve the original issue. Redundant explicit choices require a fresh readback.
  Independent agent review found/fixed wrong-field and wrong-item scope bypasses.
- Added bounded retained request + actual assistant reply as dialogue roles, safe
  schema-validation diagnostics without inputs/credentials, and offline full voice
  tests for clarification, partial recovery, request truncation and confirmation.
  Research routing remains unchanged. New response phrases are draft Darija.
- Live evidence is MIXED, not a success claim: a diagnostic with extra issue fields
  parsed pizzas but silently omitted bottle size, illustrating unstable behavior.
  Explicit runtime scope rules then produced correct drink-size questions. A short
  cola response resolved that scope but dropped all earlier pizza details. Medium
  reasoning and history roles did not fix this. Keep low reasoning; no measured
  benefit justified switching. One early schema ValidationError was retained;
  later calls were valid. No ASR/TTS calls this run, no full conversation proven.
- New diagnostics: bench/ambiguity_probe.py (one opt-in provider call),
  bench/clarification_replay.py (two, or one with --followup-only). Each writes a
  timestamped report so failed attempts survive. Do not run them automatically
  on every wake. A mocked pipeline passing is not evidence of model recovery.
- Verification after scoped work: **185 passed, 1 skipped**. App restarted after
  checks. Next run should progress the explicit recovery failure, not rerun passed
  tests or change providers merely to fill the schedule.
- 2026-09-05 05:46 UTC run: added validated, uncommitted pizza proposals for initial
  drink-size/unsupported-drink clarification. Committed values/version/item IDs
  remain unchanged while staged. Resolution applies saved proposal + new answer
  atomically and only once, then requires full completed readback. Wrong IDs,
  failed batches, unrelated edits and replacement proposals cannot leak a commit.
- Explicit discard_clarification abandons only pending work, preserves committed
  order/allocator and invalidates old playback acknowledgements. Commit/discard
  clears retained reconstruction text even if the resulting order is incomplete.
  Router sees preview IDs separately from committed state; saved traces keep both.
- A live response produced a valid drink clarification and staged pizzas but a
  redundant intent=task label. Added conservative normalization ONLY when committed
  ops are empty, affirmation/negation false, and resolve/discard IDs null. Infer the
  lower-privilege intent from the validated clarification, then run all validators.
  It cannot authorize mutation/confirmation; normalization is recorded in calls.
- **Live regression passed:** clarification_live_20260905_055148.json contains two
  actual router calls: original saved long Darija request stages large-cheese and
  medium-beef pizzas, then 'كوكا' applies both plus cola and produces full readback.
  Call times 2062ms and 985ms. This is one interpreted development fixture with a
  synthetic follow-up, not native gold, microphone evidence or an accuracy score.
  Failed pre-normalization attempt is preserved in 20260905_054932 report.
- Verification: **202 passed, 1 skipped**. Added full mocked VoiceSession outcomes
  for complete/incomplete proposal commit and discard, stale acknowledgements and
  fresh confirmation. Agent report: staged_proposal_design_review_20260905.md.
  No ASR/TTS provider calls this run. Restarted app after checks; single heartbeat
  remains active. Next priority is actual audio transport, not more prompt tweaks.
- 2026-09-05 06:25 UTC run interrupted by user feedback: user perceives no visible
  improvement and asks what happened during absence. Explained concrete changes
  and unproven live usability. Shift progress evidence toward playable artifacts.
- Added bench/audio_turn.py: one documented real recording through current MoulSot,
  router and Darija XTTS; complete WAV length/format verified, reply and combined
  conversation saved. No browser, simulated ACK, microphone capture or device-playback
  claims. Agent inventory report audio_fixture_inventory_20260905.md documents
  source/hash and lack of native ground truth. No ASR fallback used.
- Output recorded_order_20260905_062914_reply.wav says two large cheese pizzas and
  water, then asks confirmation. Timings 4860ms ASR,1375ms router,6344ms TTS including
  cache lookup. Report recorded_order_20260905_062914.json and combined conversation
  WAV retain the evidence. This single request gives useful latency evidence, not
  accuracy or proof of a completed order. No engine changes/tests claimed this run.
- Audio audit found old xtts_transport_reply_0/1.wav are only726-byte PCM tails,
  not valid WAVs. Do not use them as proof of playback. Any WebSocket harness must
  aggregate all binary frames, verify declared byte count and parse full RIFF before
  playback/acknowledgement. Valid new audio artifacts do not have this defect.
- User redirected work toward the reusable engine and requested shorter idle gaps.
  Updated the existing heartbeat to every5minutes, preserved its identity, and
  rewrote its durable prompt to require persistent work, shared multi-domain core,
  actual MoulSot, agent collaboration and visible/audible results. No duplicates.
- Extracted engine/transactions.py with generic root fields and named stable-ID
  collections, atomic rollback, existing normalization, limits and exclusive values.
  Both TaskState (including clinic) and DemoOrderState now call this same executor.
  Domain-specific pizza field projection is explicit in demo transaction_schema.
  Agent independently implemented 23 executor tests and reviewed the integration.
- Extracted engine/confirmation.py: both task shapes now share completed-readback,
  exact-version, interruption and correction guards. Fixed flat TaskState acceptance
  surviving negation and redundant correction retaining an old readback latch.
  Cross-domain tests use fictional canonical clinic inputs, no native/speech claims.
- Verification: 232 passed, 1 skipped (native-number fixture). No ASR/TTS/router calls
  for this refactor. Existing real MoulSot audio evidence remains the prior recorded
  diagnostic, not a new cross-domain voice result. Review caught renamed collection
  breaking pizza dialogue: adapter now rejects incompatible schemas immediately.
  Session traces do not persist the stable-ID allocator for restoration; there is
  no supported restore path, and one must not be added using only saved row IDs.
- Restarted the checked local server after the shared-core checks. Listener PID
  38664, wrapper14272; GET demo page returned200. Startup model-list check passed
  for openai/gpt-oss-120b. Logs: server_shared_*.stdout/stderr.log. No inference
  calls were made by this startup check.
- 2026-09-05 clinic milestone: extracted ScopedDialogue lifecycle; implemented
  configurable ScopedTaskState; integrated strict scoped flat router, shared
  pending history, voice pipeline selection and fictional clinic profile/UI.
  Two agents implemented/reviewed separate boundaries and cross-domain tests.
  Full suite264passed/1skipped; JSsyntaxcheckpassed. Browser snapshot confirms
  clinic demo enabled with fictional/no-booking copy and actualMoulSot disclosure.
- Live synthetic diagnostic clinic_audio_turn_20260905_065311.json: XTTSinput
  doctorB+10:30; actualMoulSot heardunspecifieddoctor. Router rejected mixed time
  edit+doctorclarification. 065522 report capturedresponse shape. Fixed strictly
  lower-privilege staging (1..40validated unrelatedops, noyes/no/unclear/resolve/
  discard/existingproposal; nevercommit). Replay065655 reusedoriginalASR; actual
  router+XTTS nowaskdoctorchoices while10:30remainsheld. AudiofilesverifiedfullWAV.
  Stage times ASR5797ms, successfulrouter1375ms,TTS5219ms in separate runs.
  No nativeaccuracy/microphone/deviceplayback/completedclinicconversation claims.
  Details: bench/results/clinic_preview_milestone_20260905.md. Server refreshed
  afterchecks: listener35340, wrapper9480. Next actualWebSocket/VADtransportprobe.
- Actualtransport firstattempt a4626a... failed1009beforeASR. Pinnedwebsockets13.1
  checkscompressedwirelengthbeforeinflate; deflatecanexpand1024bytePCM. Agent
  reproducedfiveephemeralloopbackcontrols; productionrun_server nowdisablesdeflate
  whilekeeping1024bound. Sixregressionsincludeproductionkwargsand1025rejection.
- Secondtransport00aef... deliversactualSilero/MoulSot/Groq/cachedXTTSreply, all167
  replyframes170710bytesverified andsimulateddurationACKprocessed. Servertrace
  demo_session_500d6e435fe24f58be0b33f92245c22e.json: ASR3563ms/router1000ms/cache16ms;
  4594ms endpoint-eventtoaudio-ready,5953ms source-endtoaudio. Time10:30heldpending
  doctorchoice. Reporttopstatusfailed due normal-close senderrace AFTERdone; preserve
  it. Agentfixingrunneroffline; do notrerunprovidersjusttochangestatus. Measured
  source3.95saudiotook5.56stotransmit onWindows; don'tclaimgenuine32mspacing.
- AddedTTSper-rendercachehit/miss/elapsedtelemetry tosavedvoice sessions andaudio
  runner. Existingcachebehaviornowmeasurable, notanewclaimedcacheoptimization.
  Fullsuite270passed/1skipped; targetedTTStelemetry32passed. Productionlistener41076,
  wrapper9788. Detailsandlimits: bench/results/transport_fix_20260905.md.
- Runnercleanupfixed: cancels/joinsPCM sender beforestop; rechecksstopafterpacedwait;
  earlycleanclosuresremainfailures. Threeofflineprotocolregressionspass. Original
  livereportunchanged; noextrainference. Finalfullsuite273passed/1skipped. Server
  remainslistener41076. Heartbeatactiveevery5min, no duplicate orschedulechange.
  Nextboundedstep: exactstaticreplycachepreparation, thencache-missdedup/cancellation
  validation; fixWindows harnesspacing separatelybeforeclaimingreal-timemetrics.
- 2026-09-05 speech preparation milestone: nine bounded real XTTS fragment
  generations, 287 input characters total. Eight clinic and seven pizza exact
  replies are now available. Network-blocked verification made zero requests;
  maximum local render times were 2.4432 ms and 2.5214 ms respectively. Dynamic
  readbacks still require synthesis; this is not end-to-end latency or quality.
- Added bench.prepare_speech dry-run/explicit-execute planner with whole-reply
  budgets, validation, deduplication and stop-on-first-provider-failure. Added
  per-instance cache-miss coalescing, atomic validated publication, cancellation
  isolation and shared shielded close cleanup. Independent audit caught and
  verified fixes for cancelled/concurrent close and API/revision key collisions.
- Timing now uses perf_counter for fast TTS renders. Older zero/16 ms readings
  used the coarse Windows monotonic clock; historical reports remain unchanged.
  Another agent fixed harness cumulative frame pacing and long-stall resets.
  Offline Windows125-frame check averaged32.098ms; maximum47.001ms, no resets.
  No provider calls for pacing, no microphone/device playback claim.
- Full suite317passed/1skipped. Cache and pacing evidence:
  bench/results/speech_cache_milestone_20260905.md. Corrected README's old claim
  based on truncated transport WAVs. Refreshed verified server: listener14348,
  wrapper7364, logsserver_cache_20260905_083812.*. Pizza/clinic pages return200;
  startup model-list check passes (no inference). Existing five-minute heartbeat
  remains active without duplication. Next priority is updated above.
- 2026-09-05 conversation lifecycle: independent agents reproduced an endless
  bare-no/repeated-yes repair loop and stale two-frame barge-in history leaking
  across normal playback completion. Fixed shared dialogue repair accounting,
  aligned explicit redundant corrections to fresh readback, and clear barge-in
  history at playback allocation/completion. Close cancellation now waits for
  shared cleanup and persists the session once. Canonical fixtures in both domains.
- Added actual browser AudioWorklet/WebAudio diagnostic with synthetic silence
  and mocked sockets: full WAV ACK, interrupted output without ACK, late decode
  suppression and incomplete-WAV cleanup pass. No physical microphone or speaker
  quality claim. Rechecked after UI changes; browser sessions closed afterward.
- Proposed-details UI uses the same renderer as committed values and stays
  separate until clarification resolves. Four desktop/mobile scenarios pass
  resolution/discard/end cleanup without overflow; screenshots visually checked.
  Evidence: browser_pending_details_20260905.json and output/playwright/proposed_*.
- Added request-local MoulSot upload/submission/result-wait timings and payload
  duration/size, including failure/cancellation. Session transcript_requests now
  distinguish partial/final waiters and reuse without raw audio/transcript telemetry.
  One actual existing synthetic3.95s WAV: MoulSot13049ms, upload858ms, submission154ms,
  resultwait12037ms. Same unspecified-doctor transcript; no fallback/router/TTS.
  No latency improvement or native accuracy claimed; no repeated probe afterward.
- Full suite345passed/1skipped. Latest local listener13940, wrapper34820;
  server_lifecycle_20260905_085851 logs, both domain pages200, startup modelscheck
  passed. Details: bench/results/conversation_lifecycle_20260905.md. Single
  five-minute heartbeat remains active, unchanged. Next concrete priority updated.
- 2026-09-05 slow-turn milestone: shared pipeline emits per-stage progress with
  fresh IDs and closed/stale completion guards. UI immediately distinguishes
  transcription, routing and voice preparation; elapsed appears after1.2s,
  long-wait text after8s. End session stays enabled, timer cleanup suppresses
  late events, and ticking seconds are outside live announcements.
- Thirteen new blocked-provider lifecycle cases pass across both domains.
  Browser tests use real >=8s waits and synthetic AudioWorklet input with mocked
  sockets: stale finish ignored, latest state restored, stop sent once, input
  tracks ended and late errors ignored. Playback regression still passes. Both
  desktop/mobile waiting screenshots visually reviewed. No provider requests.
- Prepared optional hosted profiling.patch/app.profiled.py using primary sources:
  MOULSOT_PROFILE=1, request-local IDs, body/parse/model-call host-wall timings,
  safe stdout records. Original app.py/source.json and API/model/duration unchanged;
  no deployment/model imports. Ten extracted-function tests and patch check pass.
  Exact queue time/GPU kernel time remain unmeasured; correlation limits documented.
- Final full suite368passed/1skipped. Local server refreshed: listener41016,
  wrapper2620, server_progress_20260905_091625 logs; model-list startup check and
  both domain pages200. Evidence: bench/results/slow_turn_experience_20260905.md.
  Latest actual MoulSot result is unchanged from prior turn; no new inference
  this milestone. Single five-minute heartbeat remains active without edits.
- 2026-09-05 server recovery: fixed provider leaks during allocation, guaranteed
  all registered resource closers run after failures/cancellation, and removed
  the second close after peer disconnect. Actual offline Uvicorn/WebSocket tests
  reproduced the disconnect error in both domains; now interrupted sessions save
  correctly. Invalid demo provider keeps capture usable and reports setup issues.
- Full suite401passed/1skipped; 21 connection lifecycle, four actual loopback and
  eight setup checks. Real frontend setup recovery verified with isolated invalid
  HTML fixture; actual .env unchanged. No inference calls this milestone.
  Evidence: bench/results/server_connection_recovery_20260905.md.
- Refreshed local server: listener8140, wrapper40548; both domain pages200 and
  startup model-list check passed. Logsserver_cleanup_20260905_093651.*.
- Local MoulSot hardware/dependency feasibility checked without installation:
  RTX4050 6GB currently lacks free memory for weights; no project GPU worker to
  reclaim. Separate environment/JSON sidecar is the prospective path, not yet
  proven. Details in deploy/moulsot-space/LOCAL_FEASIBILITY.md. Existing single
  five-minute heartbeat remains ACTIVE with persistent-work instructions.
- 2026-09-05 profile identity milestone: independent audit reproduced malformed
  doctor->date override silently dropping the required doctor and accepting only
  date/time. Fixed selected-field uniqueness, source identity, override targets
  and immutable IDs. Fifteen independent regressions pass; shipped profiles work.
  Full suite416passed/1skipped, no inference. Evidence:
  bench/results/profile_identity_20260905.md. Broader semantic checks remain next.
- Current local server refreshed after profile fix: listener12836, wrapper27912;
  server_profile_20260905_094503 logs. Both page requests succeeded and startup
  model-list HTTP200 check passed. Automation viewed through the app; the single
  ACTIVE five-minute heartbeat and persistent-work prompt remain unchanged.
- 2026-09-05 merged demo preflight: reproduced missing clinic question, omitted
  required readback field, wrong format and pizza implicit-field omission. Added
  shared rendering-contract checks plus offline check_config.py --demo; helpful
  page/socket setup failures allocate no providers. Independent optional-field
  control confirms absent optionaltime is skipped and suppliedtime is spoken.
- Trace a673ff... exposed terminal handling of model validation despite ASR success.
  Added RouterOutputError for all-output-validation exhaustion and bounded shared
  repair retaining state/proposals. HTTP/protocol/mixed/timeouts stay terminal;
  researchbehavior unchanged. Twenty independentcross-domaincases pass; old200
  regression updated toexactnewcategory. Fullsuite452passed/1skipped.
  Evidence: demo_preflight_20260905.md and router_output_recovery_20260905.md.
- Refreshed server afterthosechanges: listener30820, wrapper5100,
  server_recovery_20260905_100446 logs, bothpages200/startupmodelscheckpassed.
  No liveinference thismilestone. Next multi-turnharness work underway:
  conversation_research owns bench/voice_transport.py + bench/voice_manifest.py;
  regression_audit owns tests/test_voice_manifest.py; rootownsproduction exact
  playback_completeevent andtests/test_playback_complete.py. Currentserver predates
  the completionevent andneedsrefresh onlyafternewintegration passes.
- 2026-09-05 multi-turn replay: sharedsingle/multiWAVrunner withdefaultoffline
  manifestvalidation, explicitexecution, exactserverplayback_completeevent,
  per-turnsource/state/action/provenancechecks. Independenttests caughtACK-send
  completionraceandterminalstoprace; fixed withdeferredACKcompletionandnaturaldone.
  Finaltailtruncationrecordedaccurately; earlierterminalstillfailsunplayedturns.
- Thirtymanifesttests +sixservercompletiontests pass. AddedactualTCP/UvicornASGI
  three-turnrequest/correction/confirmationtests inbothdomains, withfakeproviders
  andreal32msFramePacer. Bothsaveconfirmeddemo_completed; exactly3ASR/3routerfixture
  callsand4completeWAVreplies. Initialfasttestfloodstalledsocketclosure, sofixture
  nowusesrealpacing; nofailurehiddenwithproductionchanges. No liveinference.
- Fullsuite490passed/1skipped. Evidence: bench/results/multi_turn_replay_20260905.md
  and multi_turn_offline_20260905.json. ExistingclinicWAVmanifestcheckedoffline;
  clinic_manifest_validation_20260905.json network_used=false. Noaudio regenerated.
- Currentserverafterfinalvalidation: listener27000, wrapper34932;
  server_multiturn_20260905_102305 logs. BothpagesHTTP200/startupmodelscheckpassed.
  Bothagentsfinished; nofileownershipremains. Singlefive-minuteheartbeatunchanged.
  Nextread-onlycapturequalityinvestigationupdatedabove; no repeatedproviderprobe.
