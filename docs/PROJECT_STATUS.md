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
- Configured flat tasks, pizza collections and an opt-in configurable collection
  adapter with validated operations, correction handling and separate pending/
  committed values. A collection can contain repeated rows plus shared root fields;
  its router, readback, UI and recovery use the configured field names.
- Conditional clarification schema: ambiguous values require an affected field.
  Dependent alternatives are not silently selected or committed.
- Explicit **Continue saved details** after an eligible provider failure. Recovery
  restores only validated committed values, discards pending changes, and requires
  a new readback/confirmation. It lasts up to 30 minutes while the page and server
  stay open; it does not survive a reload or restart.
- Offline tests, router probes, supplied-audio replay and real-browser diagnostic
  scripts. See [adding a task](ADDING_A_TASK.md) and [voice replay](../bench/VOICE_REPLAY.md).

## Verification and limits

The latest September 6, 2026 local suite passed **1071 tests, with 1 skipped native-review
fixture**. Eleven browser recovery checks passed using mocked microphone/socket
events. These establish engineering behavior, not native speech accuracy.
Nine additional mocked-browser collection checks cover distinct rows, pending
previews, configured labels and desktop/mobile layout. Temporary English equipment
fixtures test original and renamed field IDs; they are not a shipped voice domain.
Cross-row coupled choices are not supported by the new collection adapter yet.
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
