# Current project status

Darija Voice is a reusable voice task engine. Pizza orders and a fictional clinic
preference preview share task transactions, clarification, corrections, pending
changes, cancellation and fresh confirmation. The clinic demo does not book visits;
the pizza demo does not place orders.

## Implemented

- Browser microphone capture, end-of-turn detection, progress feedback, audio
  replies, interruption handling and resource cleanup.
- Actual MoulSot recognition, using a hosted Space or the optional local bridge.
  Groq interprets structured task operations; it does not replace MoulSot with Whisper.
- Optional task-specific vocabulary for controlled local MoulSot demo experiments,
  with strict bounds and an exact forwarding acknowledgment. All shipped tasks
  leave it off; native benefit is unproven. Hosted and reviewed research paths
  reject enabled nonempty vocabulary before inference.
- Local supervision survives status-file sharing locks without shutting down
  healthy speech services. Real service failures and STOP still close owned jobs.
- Synthetic Darija XTTS demo voice, local reply caching and optional semantic
  readback grouping. Whole summaries remain the default pending quality review.
- Speech timings separate cache reads, remote submission/wait/download and local
  conversion/assembly. Shared synthesis work is recorded once, with references
  from waiting renders; incomplete/cancelled work stays explicitly identified.
- Configured flat tasks, pizza collections and an opt-in configurable collection
  adapter with validated operations, correction handling and separate pending/
  committed values. A collection can contain repeated rows plus shared root fields;
  its router, readback, UI and recovery use the configured field names.
- Linked choices across configured collection rows and shared root fields hold
  all declared addresses in one draft until explicitly answered, with fresh
  question IDs and masked unresolved previews. Exact
  configured enum/integer answers can resolve the current linked question locally;
  the same matcher handles flat clinic coupled questions. Evidence separates
  these lookups from Groq requests; it does not guess dates, times or ASR aliases.
- Recognized router schema-configuration errors stop immediately with a static
  setup message instead of retrying an identical request or asking for repeated
  speech. Other error shapes keep the existing retry behavior.
- Configured collection rejections identify bounded static validation rules,
  separating shape/coherence from field, value and address failures without
  including rejected values or arbitrary property names in diagnostics.
- Router duration telemetry uses a high-resolution performance clock with explicit
  provenance, preserving submillisecond local lookup measurements on Windows.
- Conditional clarification schema: ambiguous values require an affected field.
  Dependent alternatives are not silently selected or committed.
- Explicit **Continue saved details** after an eligible provider failure. Recovery
  restores only validated committed values, discards pending changes, and requires
  a new readback/confirmation. It lasts up to 30 minutes while the page and server
  stay open; it does not survive a reload or restart.
- Offline tests, router probes, supplied-audio replay and real-browser diagnostic
  scripts. See [adding a task](ADDING_A_TASK.md) and [voice replay](../bench/VOICE_REPLAY.md).

## Verification and limits

The latest September 6, 2026 local suite passed **1325 tests, with 1 skipped native-review
fixture**. Eleven browser recovery checks passed using mocked microphone/socket
events. These establish engineering behavior, not native speech accuracy.
Nine additional mocked-browser collection checks cover distinct rows, pending
previews, configured labels and desktop/mobile layout. Temporary English equipment
fixtures test original and renamed field IDs; they are not a shipped voice domain.
Cross-row and root linkage now share the same explicit address-coverage contract.
Thirty-four new offline cases cover root/row scopes, held partial answers, atomic
completion, cancellation and committed-only recovery. Fifteen additional mocked
browser checks verify root/row preview separation and desktop/mobile display.
These checks do not establish that a model always identifies the dependency.
One frozen English live opening for the extension failed local response validation
and left the saved state unchanged. The diagnostic did not capture the raw model
output, so its specific failed rule is unresolved; no repeat attempt was made.
Future collection parse failures now expose bounded static rule codes. Thirty-two
new offline tests verify classification and exclusion of rejected data from
diagnostics. This instrumentation does not recover or fix that historical response.
A historical successful clinic run spent 8.315 seconds rendering a new whole
summary after a time-only correction. The new TTS subphase records passed ten
offline tests, including shared requests, errors, cancellation and detached
session snapshots. No new live synthesis was performed to repeat this timing;
whole-summary mode and pronunciation review requirements remain unchanged.
The new collection router passed one real English API compatibility check after
fixing an overlapping-union HTTP400 rejection. Its strict wire schema constrains
structure and configured names; local parsing validates root/row relationships
before atomic state application. This check made no speech calls.
Ten further mocked-browser checks validate linked-row preview masking. A real
English router diagnostic first omitted linkage, then declared it after a generic
prompt refinement but repeated the question after an exact answer. Replaying its
saved successful opening with the local exact-answer path now advances from
equipment to quantity and reaches the correct unconfirmed readback with zero new
API calls. This fixes that scoped text flow; broader model interpretation and
native spoken accuracy remain unproven.
The same saved-opening replay now advances a clinic doctor choice to its linked
time question without an API request. A separate two-call actual MoulSot check
on a frozen short synthetic doctor-name clip failed: vocabulary context inserted
both doctor options instead of producing one exact alias. Context remains off;
this result does not establish a native speech improvement.
A separate offline audit using the actual installed Silero VAD retained the short
clip as one segment, discarding its first 32 ms. Whether those samples contain
important speech remains unreviewed; the original full-file recognition failure
was independent of endpointing. No VAD or gain change follows from this result.
The router clock correction passed 6 new tests and 65 relevant tests total; this
is measurement validation, not evidence of faster voice conversations.
The earlier published source-only Git export passed 842 tests with the same single skip
after aligning the profiling artifact checksum with Git's canonical LF files.
No local recordings, `.env` file or model downloads were included in that export.

A subsequent local fix prevents silent first-option selection in a narrowly
recognized configured-enum alternative. Two previously failing English text cases
now ask a choice question and hold independent date/time details. Linked fields
can also stay pending across partial answers, then commit together with fresh
confirmation. This closes the reproduced state-contract gap when coupling is
declared; the router must still recognize the relationship.

A new actual MoulSot synthetic-input check omitted one option name. It asked a
choice question without committing anything, but initially missed the linked time.
One recheck of the saved transcript after generic prompt refinement tracked both
fields. That is bounded diagnostic evidence, not native accuracy or a repaired ASR.

A short synthetic follow-up was transcribed as `الطبيب باع`; its first routing
attempt was rejected, while one saved-text diagnostic later held doctor B and
asked for the linked time. No routing behavior changed between those attempts,
so variability remains. Static validation-stage telemetry now identifies future
rejection sites. Offline pipeline checks preserve coupled drafts and the original
request through rejected answers, without silently committing or confirming.

One frozen vocabulary experiment restored a missing B token but failed its
predeclared exact whole-label gate. A separately reported exploratory negative
control was unchanged. Vocabulary transport is implemented for controlled tests,
not enabled as an accuracy fix. The full regression run also exposed a test's
timing assumption about sender shutdown; deterministic pacing and delayed-ACK
coverage corrected the test without changing production transport behavior.

A bounded synthetic clinic sequence used actual MoulSot, Groq and Darija XTTS to
collect preferences, correct a time, read the result back and accept a separate
confirmation. Its scripted audio and playback acknowledgements do not establish
real kitchen or microphone reliability. Earlier failed attempts were retained in
local diagnostics. One real Groq dependent-choice check accepted the corrected
inline schema; that is compatibility evidence, not a model accuracy benchmark.

Native-reviewed Darija cases, human conversation evaluations, pronunciation review
and consistent end-to-end latency measurements remain incomplete. This is a
development build; research acceptance and production readiness are not claimed.

## Reproduce locally

Use Python 3.11 and the pinned [requirements](../requirements.txt), create a local
`.env` from [.env.example](../.env.example), then follow the [README](../README.md).
Run `python -m pytest -q` for offline checks. Live diagnostics require explicit
provider configuration and consume provider quota when executed.

Recordings, provider credentials, model downloads, generated reports and browser
artifacts are excluded from Git. Historical report links into `bench/results/`
refer to private local evidence; the source probes and tests are included so new
results can be generated. External model weights are not redistributed here.
