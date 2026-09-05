# Local quantized MoulSot experiment

The supervised live demo can now run MoulSot on this computer. It leaves saved
credentials, the hosted `.env` configuration and engine Python pins unchanged;
only the supervised engine child's environment selects local recognition.

`prepare.py` prints the download plan by default. Explicit `--download` fetches
four pinned public assets (2.109 GB total), verifies exact sizes and SHA256, and
extracts each official runtime archive into a separate directory under
`.local/moulsot`. No model is executed by this script. Failed partial downloads
are never loaded. The artifacts are ignored by Git.

The decoder and audio projector are the same revision of a third-party
[MoulSot v0.3 quantization](https://huggingface.co/mradermacher/moulsot.v0.3-GGUF),
`ca66fea7f3db516212720fd005a0d70a57213de8`. They total 1,463,115,808 bytes before
runtime buffers. Quantization can affect recognition; numerical equivalence and
native accuracy have not been established.

The portable Windows CUDA 12.4 binaries and matching DLLs are from official
[llama.cpp b10809](https://github.com/ggml-org/llama.cpp/releases/tag/b10809).
Its [multimodal documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
lists Qwen3-ASR support but calls audio support experimental. The converter uses
`qwen3vl` for the text decoder and a separate audio projector; that decoder label
does not by itself imply the wrong model.

Before inference, inspect binary help, current free RAM/VRAM, and runtime logs.
Require the audio projector to load correctly. Bound context, generation length,
threads, elapsed time and memory. Keep the audio encoder on CPU initially if GPU
headroom is insufficient. Stop only the experiment's own process on failure.
Record cold load separately from warm inference, preserve raw output, and do not
compare unlike timings as a speedup. No production provider change follows merely
from a successful model load.

## Optional engine bridge

`bridge.py` adapts the already running loopback model API on port 8011 to the
engine's existing multipart JSON ASR contract. It does not start the model or
change the production provider. Run explicitly with one worker, for example:

```powershell
python -m uvicorn bridge:app --app-dir deploy/local-moulsot --host 127.0.0.1 --port 8012 --workers 1
```

POST one `file` WAV to `/transcribe`. Supported input is complete, nonempty
16 kHz mono PCM16 up to 20 seconds, with bounded upload/response bytes and a
30-second request timeout and a shared 512-token output ceiling. Busy requests
return 429. Truncated output is rejected. The bridge checks exact
model identity, normal completion and the Arabic ASR prefix; it never invents
confidence or repairs transcript wording. `/health` checks the adapter only,
not model readiness. No credentials are sent to the loopback upstream.

The existing `SpeechToText` adapter can use endpoint
`http://127.0.0.1:8012/transcribe` with `stt.moulsot_protocol=json` in an explicitly
selected configuration. The normal demo configuration remains unchanged.
`warm_probe.py --audio <wav> --gpu-projector --engine-bridge` performs one bounded
actual engine-adapter request through an in-process ASGI bridge and real TCP
model server, then stops the model. This is not browser/microphone validation.

The busy reservation is per bridge process, so use one worker. Cancellation
closes the upstream request but cannot guarantee immediate GPU-work termination.
The experimental runner separately enforces its own process deadline/resource
guards. The supervised launcher below now owns the service lifecycle.

Independent offline bridge tests cover upload bounds, complete long responses
and truncation rejection. See the live evidence below for the latest full-suite
checkpoint. A real adapter request returned a transcript in
1.365 s, with null confidence and quantized-MoulSot provenance. See the
[full evidence](../../bench/results/local_moulsot_feasibility_20260905.md).

## Supervised live demo

Use the project's Python environment. Run `run_local_demo.py` without arguments
for read-only preflight. With `--start --hosted-fallback`, it starts hidden local
services on ports 8011/8012 and the normal demo on port 8000. Existing listeners
are rejected; it never kills them to claim a port. The supervisor verifies pinned
assets and extracted runtime files before model allocation. It uses the normal
engine runner, including its tested WebSocket frame limits and compression setting.

`--device cuda` remains the default. Opt into the same quantized MoulSot on CPU
with `--device cpu`; preflight still runs without `--start`:

```powershell
python deploy/local-moulsot/run_local_demo.py --device cpu
python deploy/local-moulsot/run_local_demo.py --device cpu --start --hosted-fallback
```

CPU mode puts both the decoder and audio projector on CPU using `--device none`,
`-ngl 0` and `--no-mmproj-offload`, with four threads. It requires 3072 MiB free
RAM before launch and stops its owned services below 768 MiB. It does not query
GPU memory or apply a GPU memory guard. CUDA mode retains the existing launch
requirements of 2560 MiB RAM and 1900 MiB GPU memory, and running floors of
768 MiB RAM and 256 MiB GPU memory. Both modes use the same ports and refuse
existing listeners; changing device requires stopping the current supervisor.

One CPU comparison of the existing 2.219-second synthetic clinic correction
finished recognition in 2.031 seconds, after the GPU path had timed out on that
input. This is a single diagnostic under shared-machine load, not a sustained
latency or native accuracy claim. See the [CPU report](../../bench/results/local_moulsot_cpu_b85ca16c39184b54946b0bbd4b8e7136/report.json).

The child environment sets `MOULSOT_PROTOCOL=json` and the local bridge endpoint;
the saved `.env` is unchanged. If an owned service exits or the resource guard
fires after readiness, the optional fallback restores the original hosted
MoulSot engine once. No GPU restart or provider-inference retry loop is created.
Current state is `.local/moulsot/supervisor.json`, including real supervisor PID,
child PIDs, logs, `selected_device` and whether hosted recovery occurred. The
selected device describes the local configuration; `mode` separately records
whether the supervisor is currently local or has recovered to hosted MoulSot.

Status snapshots are atomic and best effort. Brief Windows sharing locks receive
at most three attempts with 150 ms total retry delay per destination. Persistent
status-write failures do not stop healthy speech services or activate hosted
fallback: the supervisor keeps checking its owned processes, resource guards and
STOP marker. Current and per-run JSON snapshots are independent, and stderr logs
an outage/recovery transition instead of repeating a warning every poll. Check
`updated_at` before relying on a snapshot; a locked destination can be stale.
Available snapshots expose `status_publication_unavailable` and a cumulative
`status_publication_failures` count. Real service failures still close the owned
Windows job even when final status cannot be published.

Stop using Ctrl+C when foreground, or create `.local/moulsot/STOP` when hidden.
The supervisor closes its Windows job and all owned descendants; this can
interrupt an active conversation before its final report saves. Remove that
marker explicitly before starting again. `--max-seconds N` bounds a test run;
polling/network waits can add a few seconds. No other user processes are stopped.
Use the usual `python -m engine.server` afterward to run the saved hosted setup.

`warm_probe.py --audio <owner-wav> --gpu-projector --voice-pipeline` is a separate
one-input experiment with real VAD/VoiceSession/router and configured XTTS,
simulated playback ACKs and complete reply WAVs. It stops its model afterward.
Do not run it alongside the supervised model on the same ports.

The live clinic WebSocket exchange and browser configuration now pass; see
[live evidence](../../bench/results/local_voice_live_20260905.md). Native
multi-turn accuracy and sustained resource stability are still open.

## Optional vocabulary experiment

Ordinary sessions send no vocabulary context. For controlled **demo-only**
experiments, the selected task's base configuration may include this inside `stt`:

```json
"moulsot_context": {
  "enabled": false,
  "terms": ["الطبيب ألف", "الطبيب باء"]
}
```

These example terms are draft synthetic-clinic labels, not reviewed language.
Use a small symmetric vocabulary of task options, never an expected answer or
conversation history. There is no global vocabulary environment variable or
browser-supplied session prompt. Current shipped configs omit this setting.
Terms are copied from the selected configuration for each adapter instance.

Explicitly enabling a nonempty list requires demo mode, MoulSot as the only
recognizer, fallback disabled, JSON protocol, and the exact endpoint
`http://127.0.0.1:8012/transcribe`. Unsupported targets fail setup before inference;
reviewed research sessions reject enabled nonempty context. The page labels an
enabled experiment. Empty or disabled context keeps existing requests unchanged.

The configuration accepts at most eight distinct, trimmed, nonempty terms,
totalling at most 160 characters when joined by `، `. Control characters and
angle brackets are rejected. The bridge accepts one WAV and at most one bounded
UTF-8 `context` form field. It prepends that string as a system message while
keeping the audio payload and decoding settings unchanged. Responses acknowledge
the exact forwarded string with `context_sha256`; the adapter rejects missing or
mismatched acknowledgment. This proves forwarding, not model compliance or
improved recognition. Ordinary transcript wording is never repaired from hints.

Local call evidence records the terms, hash, experimental flag and `applied`
acknowledgment, including failures. Controlled tests cover the adapter, multipart
bridge, isolated tasks and mocked model transport without additional inference.
The original one-clip experiment restored an omitted name token but failed its
literal whole-label gate; a separate negative control was unchanged. This is
insufficient to enable vocabulary in ordinary sessions or claim native accuracy.
See local [research and evidence](../../bench/results/moulsot_vocabulary_followup_20260906.md).
