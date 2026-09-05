# MoulSot ZeroGPU startup repair

An optional, **not deployed** profiling variant is now available as
`profiling.patch` and `app.profiled.py`. It preserves the model, API, GPU duration
and existing return behavior, and enables host-wall timing logs only with
`MOULSOT_PROFILE=1`. The original `app.py` below remains the baseline. Read
[PROFILING_PLAN.md](PROFILING_PLAN.md) for validation and correlation limits.

This directory contains a corrected copy of the duplicated Space's `app.py`.
The source commit and hashes are recorded in `source.json`; `zerogpu.patch` shows
the complete changes. AtlasIA's original copyright/license header is retained.
These files belong to the hosted Space, not the local engine's pinned runtime.

Applied to the hosted Space as commit
`5d92fc83d3104f3f6cbd43ac4b1d425ddaccda22` on September 5, 2026.
The remote file was checked against this local copy and matches exactly.

The copied app loaded the model without importing `spaces`, had no GPU-decorated
function, and explicitly selected CPU. Gradio later imported `spaces` after CUDA
initialization, causing the reported startup failure.

Changes:

- Import `spaces` before Gradio, Torch and Qwen ASR.
- Load the model on CUDA at module scope, using ZeroGPU's startup emulation.
- Decorate `transcribe` with `@spaces.GPU(duration=60)` to allocate GPU time.
- Pass `theme` and `css` to `launch`, as required by the Space's Gradio 6 runtime.

The existing Space requirements already include `spaces`; no dependency change
is needed for this repair. Keep its hardware set to ZeroGPU.

Apply by replacing the root `app.py` in `Viego09/MoulSot.v0.3` with this file and
committing. Hugging Face will restart the app. A factory rebuild is unnecessary
unless a later error points to stale dependencies.

Validation: Python compilation and AST checks for import order, CUDA model
placement, and the decorated inference function passed. The hosted runtime then
reached RUNNING on ZeroGPU, and `/gradio_api/info` returned HTTP 200 with the
`/transcribe` endpoint. Its included approximately 21-second `audio.wav` sample
transcribed successfully through the browser, producing Darija/French text.
This is a deployment/inference smoke, not the planned family/kitchen test or a
measured accuracy result.

Reference: [Hugging Face ZeroGPU setup and model loading](https://huggingface.co/docs/hub/spaces-zerogpu).

The initial 120-second ceiling was subsequently reduced to 60 seconds: the
provider rejected a short test with “180s requested vs. 177s left” after applying
its quota accounting. Short recordings had completed in a few seconds. This
reduces the maximum permitted GPU run time; it does not reset or bypass quota.
Long recordings that need more than 60 seconds of inference will time out and
need a separately assessed duration budget. The latest commit is in `source.json`.
