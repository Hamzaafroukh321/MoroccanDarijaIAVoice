# Replaying a supplied voice conversation

The manifest runner sends one to five supplied WAV recordings through one local
demo session, waiting for each verified reply and matching server completion
before sending the next. Pizza, clinic and valid configured demo tasks use the same runner. It generates no
input audio and does not replace MoulSot with another recognizer.

Use mono, 16 kHz, 16-bit PCM WAVs. Each file must contain one turn, be at most
20 seconds long, and have no internal pause that creates a second speech endpoint.
Combined source audio is limited to 60 seconds; the whole diagnostic is limited
to 180 seconds, or the lower configured session limit. All inputs are checked
before connecting. A malformed later file prevents the entire run from starting.

Create a JSON manifest next to your recordings, for example:

```json
{
  "domain": "clinic",
  "turns": [
    {"audio": "request.wav", "source_kind": "recorded"},
    {"audio": "correction.wav", "source_kind": "recorded"},
    {"audio": "confirmation.wav", "source_kind": "recorded"}
  ]
}
```

These are placeholder filenames, not included recordings or reviewed evaluation
cases. Use `synthetic_diagnostic` for generated input. `recorded` means recorded
audio; it does not imply native review or verified expected wording.

For another configured task, replace `clinic` with its safe lowercase task ID,
such as `service-desk`. Both `configs/<task-id>.json` and its valid demo profile
must exist, and the base configuration's `domain_id` must match the filename.
Validation rejects unsafe IDs, missing configurations and invalid/missing profiles
before connecting. Adding a task does not require editing the replay runner.

Validate locally first:

```powershell
python -m bench.voice_transport --manifest path/to/conversation.json
```

The default manifest operation does not open a socket or use a provider. To run
the recordings through the local server's actual MoulSot/router/TTS pipeline:

```powershell
python -m bench.voice_transport --manifest path/to/conversation.json --execute
```

The original single-WAV interface also accepts configured task IDs. Use
`--audio path/to/request.wav --domain service-desk --source-kind recorded --check`
for local validation. Its legacy execution behavior is unchanged: omitting
`--check` runs the single-WAV diagnostic. Manifest execution always requires
`--execute`.

Execution uses the configured services and their quotas. Input/time limits are
not strict provider-request limits: the router may retry, and a voice reply may
need multiple synthesis fragments. The harness does not reconnect, retry the
conversation or switch recognizers. It stops on a provider error, unexpected
endpoint/termination, or failed expectation. Do not schedule repeated live runs
merely to obtain a faster result.

A turn may include `expected_action` and `expected_slots`. The latter is the
entire expected committed state, not a partial match. Supply these from what you
intended in that recording. For pending changes, committed values can remain
unchanged while the proposed details await clarification. Assertions are stored
as diagnostic, unreviewed expectations; the runner does not turn them into a
native accuracy benchmark.

The runner saves complete reply WAVs, source hashes, per-turn observations and
failure details. Playback ACKs are simulated after WAV validation and a duration
wait. An exact playback ID links the ACK to the server's completion event, so
duplicate state/completion events cannot consume another turn. This verifies
transport and task behavior against supplied assertions; it does not test the
physical microphone, speakers, WebAudio or a human listener's pronunciation verdict.

For a final accepted/handoff reply, the runner waits for natural completion rather
than sending a competing stop. If that completion arrives before all conservative
tail silence has been sent, the report records the truncated tail and actual
frame count. The source audio and reply must still be complete; early termination
with later supplied turns remains a failure.

A passed transport run is not necessarily a completed task. Inspect the final
server status and state; ask for an `accepted` final action only when that is what
the supplied conversation should achieve. The fictional clinic still does not
check availability or book appointments, and the pizza demo places no order.


## Interpret speech timing

Use the matching server session report for component durations. `tts_calls` has
one entry per completed, failed or cancelled render, with `render_id`, overall
`elapsed_ms`, cache counts and `part_timings`. Each part describes its own cache
read/validation, optional shared-producer wait and PCM conversion. Render-level
phases cover any grouped-mode legacy-cache probe and final WAV assembly.

`tts_producer_calls` records each dispatched synthesis producer once. A missed or
coalesced part references its `producer_id`; IDs are local to that voice-bank
instance/session, not global identifiers. XTTS phases distinguish reference
lookup, API submission, combined remote waiting, result download, audio validation
and cache writing. `remote_wait` combines provider queueing, generation and SSE
transport; the API does not establish their separate costs. Other TTS APIs that
return audio directly have one aggregate `provider_request` phase.

These spans overlap: a render's `inflight_wait` includes its producer's work.
Do not add waiter and producer durations or count shared remote work once per
waiting render. Cache-only renders create no new producer. A cancelled render
may leave shared work running for another waiter. A saved producer marked
`in_progress` has no finalized `elapsed_ms`; its snapshot must not be treated as
a completed request. Reports deep-copy these records, so later producer completion
does not rewrite a previously saved snapshot. All phase clocks use `perf_counter`.

Synthesis finishes before the current pipeline emits assistant text and audio.
A short text-to-audio-delivery interval therefore does not measure the earlier
synthesis wait. Keep generated playback length separate from audio-ready latency,
and keep the harness's simulated ACK separate from real device playback.
Historical reports without these phases remain incomplete evidence. Capture the
new data on the next otherwise-needed uncached reply; do not repeatedly synthesize
an unchanged prompt just to search for a faster result.

## Interpret ASR runtime evidence

Each `stt_calls` entry includes `runtime_provenance`. Its `configured` fields
identify that attempt's provider and transport. Only the exact owned local
endpoint can attach a `launch_snapshot`: the supervisor verified the pinned
decoder/projector hashes and extracted runtime at startup, then checked the
owned model's reported identity before starting the bridge and engine.

The snapshot contains a fresh run ID, UTC verification time, model/conversion
identity, runtime release and hashes. `selected_device` describes the launcher's
CPU/CUDA policy; `actual_device` is null because tensor placement is not measured.
There is no large-file hashing during transcription.

`response_binding=matched` means the response's bridge acknowledgment matches
the expected launch snapshot. It does not attest to runtime memory, current
file contents or recognition accuracy. A matched response can still have
`ok=false` if its transcription or context is invalid; inspect both fields.
HTTP failures and attempts without a response cannot have matched binding.
Missing, malformed or mismatching optional metadata does not change whether a
valid transcript succeeds. `bridge_reported` is a separate bounded label, not
independent verification.

Standalone bridges without launch metadata remain unverified. Hosted, custom
and redirected endpoints cannot claim local launch evidence, and a fallback
attempt gets its own provider record. Old reports lacking these fields cannot
be retroactively assigned verified runtime details from a newer run.
