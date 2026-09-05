# MoulSot profiling plan — local design, not deployed

Reviewed 2026-09-05. An optional local implementation now accompanies this plan:
`profiling.patch` and `app.profiled.py`. Baseline `app.py` and `source.json` remain
unchanged. No inference requests, dependency changes, model changes, or deployment
were performed for this profiling work.

The artifact records its baseline SHA256. `MOULSOT_PROFILE=1` enables structured
start/end records on standard output; without that flag, no profile records are
emitted. Ten tests execute only extracted functions with fake model/audio/logging
dependencies. They verify unchanged results, API/decorator/model settings, safe
metadata, early returns, exceptions, distinct IDs and contained logging failures.
Patch application checks pass. These are local checks, not hosted measurements.

The local `app.py` places the entire `transcribe` function under `@spaces.GPU(duration=60)`. Its body parses/normalizes audio, selects language, calls `asr.transcribe(...)`, then extracts the text. The model is already loaded at module scope. Keep that structure and the existing return values, Gradio inputs/outputs, decorator duration, and launch options unchanged. The directory has no requirements manifest; record installed package versions when profiling is eventually enabled rather than assuming current documentation matches the deployed version.

## Smallest useful patch

Add standard-library timing and structured logging inside the existing function, enabled only by an environment flag such as `MOULSOT_PROFILE=1`:

- Allocate a fresh server-local random `profile_id` and record UTC entry time for log ordering.
- Use `time.perf_counter()` for elapsed durations. Record `body_ms` from function-body entry through exit, including existing early returns and exceptions.
- Measure `parse_ms` around `_parse_audio_any`, and `model_call_wall_ms` around the existing `asr.transcribe` call. Record input sample rate/count and duration after parsing; do not serialize the array.
- Emit a small start record and a final record in `finally`, carrying the same ID, outcome, elapsed fields, and exception **class** if applicable. Never log exception text, stack locals, transcript, audio, upload path, headers, IP address, username, or credentials. Logging failures must not replace the transcription result or exception.
- Keep the instrumentation local to each invocation. Avoid a module-level “current request” record because calls may overlap.

Suggested final record fields: `event`, `profile_id`, `started_utc`, `outcome`, `body_ms`, `parse_ms`, `model_call_wall_ms`, `sample_rate_hz`, `samples`, `audio_duration_ms`, `error_type`. Use null for stages never entered. A killed worker can leave only a start record; absence of a finish is not zero latency.

Function-body timing begins after the ZeroGPU wrapper reaches the decorated body. It cannot measure upload, Gradio preprocessing/queueing, or the wrapper's preceding allocation wait. Hugging Face documents that the decorator obtains a GPU for the call and releases it afterwards; keep the existing placement and duration. [ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu)

`model_call_wall_ms` is host-observed API-call latency, including library preprocessing/decoding and synchronization performed by the library. It is **not GPU kernel time**. Do not add CUDA synchronization in the minimal patch: that can perturb behavior. A separate, explicitly labelled GPU profiling experiment would need appropriate synchronization/events, since CUDA operations are asynchronous. [PyTorch CUDA timing semantics](https://docs.pytorch.org/docs/main/notes/cuda.html#asynchronous-execution)

## Correlation and outside time

Currently the client receives a Gradio event ID from the submission POST and uses it for the result GET. The prediction function has no request parameter or documented direct access to that per-call ID. [Gradio HTTP API](https://www.gradio.app/main/guides/querying-gradio-apps-with-curl)

A generated server-local ID joins only the server's own start/end records. UTC timestamps can associate an isolated diagnostic tentatively; they cannot prove a match under concurrent traffic, and client/server clocks may differ. `gr.Request.session_hash` is session/page-load identity, not a unique inference ID. Do not invent `request.event_id` or depend on undocumented queue internals. [Gradio Request API](https://gradio.app/main/docs/gradio/request)

If exact client/server pairing becomes necessary, a separate small follow-up can test Gradio's injected `request: gr.Request` with a dedicated random diagnostic header, accepting only a validated 32-character hexadecimal value and logging only that value. Keep the two component inputs and output unchanged. First verify header propagation through the deployed Gradio/ZeroGPU combination using mocked/local dispatch; public Request documentation establishes header access, but does not prove this deployment forwards a custom header through every layer. If unavailable, record the call as uncorrelated rather than falling back to user identifiers.

Compare matched **client overall ASR duration** with server body duration only as a combined outside-body residual. Do not label the residual “queue time.” The client SSE/result-wait interval starts after submission, while server execution may already have begun; results can also be buffered before the GET. Therefore `SSE wait − model_call_wall_ms` is not an exact queue measurement and can be misleading or negative. Upload, submission, framework work, allocation, networking, result serialization, and timing overhead remain partly combined.

## Local validation before any deployment

Use AST checks or an extracted function with fake `spaces.GPU`, audio parser, model, clock, and logger; do not import the full Space module because that loads model weights. Verify unchanged outputs/exceptions, identical model-call arguments and call count, logging-off behavior, early returns, failures, concurrent invocation IDs, and absence of sensitive fields. Record the source hash and package versions with future observations. This plan alone supplies no hosted profiling measurements or accuracy evidence.
