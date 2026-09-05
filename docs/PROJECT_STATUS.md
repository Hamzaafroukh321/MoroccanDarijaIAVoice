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
- Synthetic Darija XTTS demo voice, local reply caching and optional semantic
  readback grouping. Whole summaries remain the default pending quality review.
- Configured flat tasks and pizza collections with validated operations,
  correction handling and separate pending/committed values.
- Conditional clarification schema: ambiguous values require an affected field.
  Dependent alternatives are not silently selected or committed.
- Explicit **Continue saved details** after an eligible provider failure. Recovery
  restores only validated committed values, discards pending changes, and requires
  a new readback/confirmation. It lasts up to 30 minutes while the page and server
  stay open; it does not survive a reload or restart.
- Offline tests, router probes, supplied-audio replay and real-browser diagnostic
  scripts. See [adding a task](ADDING_A_TASK.md) and [voice replay](../bench/VOICE_REPLAY.md).

## Verification and limits

The September 5, 2026 local suite passed **842 tests, with 1 skipped native-review
fixture**. Eleven browser recovery checks passed using mocked microphone/socket
events. These establish engineering behavior, not native speech accuracy.
The source-only Git export also passed all 842 tests with the same single skip
after aligning the profiling artifact checksum with Git's canonical LF files.
No local recordings, `.env` file or model downloads were included in that export.

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
