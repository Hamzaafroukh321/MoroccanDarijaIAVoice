# Test in this order

Updated September 5, 2026. Real Darija accuracy and kitchen robustness are pending.

Latest demo checkpoint: **202 tests pass, 1 language fixture is skipped**. Separate
pizzas, scoped clarification and pending pizza drafts are implemented. One live
saved-transcript replay now preserves both pizzas after the short drink answer.
The next check is a bounded MoulSot–router–Darija XTTS audio conversation. See
AUTONOMOUS_WORK.md for the active queue; the reviewed research eval below remains
separate from the synthetic demo and still needs the owner's cases.

## 1. Router configuration — implemented

The local `.env` now contains `GROQ_ROUTER_MODEL=openai/gpt-oss-120b`.
Both domain configs use this default; the environment overrides them. Restart the
server after changing the model. No credential is included in this document.

Requests use strict JSON Schema generated from the response model, plus local
semantic validation. Transcripts are sent as user messages. At startup, Groq's
models endpoint must list the configured model; authentication/network failures
also stop startup when a key is configured. Capture-only startup without a key
is still supported with a warning.

The original Llama shutdown date was August 16, 2026, following the June 17
announcement. See [Groq deprecations](https://console.groq.com/docs/deprecations).

## 2. Owner-supplied 30-case router eval — ready to fill

Fill `bench/router_cases/pizza.json`: 10 straightforward, 10 corrections,
10 messy/ambiguous cases. Every entry has a suggested English intent, initial
state, and proposed expected operations/flags. Supply your natural Darija
transcript, edit the gold answer to match your wording, then set `reviewed: true`.
These are templates, not completed cases. You can also send the 30 lines with their
case IDs in chat for insertion. No Darija sentences were generated on your behalf.

Use fictional contact details. For correction cases, the initial state matters.
Drinks and toppings are list slots: setting cola means `"value": ["cola"]`.
For list additions/removals, a single item or list is accepted by the engine.

```powershell
.venv\Scripts\Activate.ps1
python bench/eval_router.py --check
python bench/eval_router.py
```

The runner calls the three requested models, saves each result immediately under
`bench/results/router_eval_*.json`, and reports exact ordered operations plus
three flags, valid-response rate, retries, provider failures, category results,
and successful-call p50/p95 latency including retries. Confidence is not scored
against a made-up gold value. Requests respect the persistent shared quota ledger.
An exhausted budget is recorded as a failure; it is never reset by the runner.
Incomplete runs are labeled and do not receive full-eval summary scores.

This is a live API text eval, without audio. Owner review provides the gold labels.
Model choice remains provisional until reviewed cases are run. Compare correctness
first, then latency; repeat finalists before claiming a stable speed advantage.

### Live protocol evidence (one English case, not an accuracy benchmark)

| Model | Strict-schema request | End-to-end routing time |
|---|---|---|
| `openai/gpt-oss-120b` | Passed | 781 ms |
| `openai/gpt-oss-20b` | Passed | 422 ms |
| `qwen/qwen3.6-27b` | Passed | 2360 ms |

Evidence: `bench/results/router_probe_3e254bf0203149988b7416d7f0d1df77.json`.
All produced the expected size operation without retry in this run. One request
per model cannot establish a latency distribution or Darija capability.

Qwen initially rejected the system-only conversation because it requires a user
message. The final router fixes that. Current [Groq structured-output docs](https://console.groq.com/docs/structured-outputs)
list strict support for GPT-OSS and Qwen 3.8, not Qwen 3.6; nevertheless Qwen 3.6
accepted this exact strict request in the live test. That observed acceptance is
not a claim of documented guaranteed support for every schema.

To repeat only the protocol check:

```powershell
python bench/eval_router.py --probe
```

## 3. MoulSot kitchen smoke — Space running; waiting for recording

The duplicate [Viego09/MoulSot.v0.3](https://huggingface.co/spaces/Viego09/MoulSot.v0.3)
is running on ZeroGPU. Its inherited startup error was repaired by importing
`spaces` first, placing the model on CUDA, and decorating inference. The repair is
recorded in `deploy/moulsot-space/`; no paid hardware was selected.
Its included 21-second audio example transcribed successfully after the repair.
The family/kitchen smoke remains pending; the included example is not a substitute.

`MOULSOT_ENDPOINT` in the local `.env` is now
`https://viego09-moulsot-v0-3.hf.space`; the local server was restarted with it.
The current adapter expects an accessible endpoint with the source Space's
Gradio `transcribe` API. Authenticated private Spaces need an authentication
integration before this runner can access them.

Record about three minutes of family speech with the TV on, with participants
agreeing to the recording/upload. Include an order, a correction, pauses in a
phone number, and natural code switching. No prompt bank or full transcript
annotation is required for this qualitative smoke. The local pizza microphone
check now permits up to four minutes; save a 16 kHz mono PCM16 WAV.

```powershell
python bench/smoke_moulsot.py bench/recordings/kitchen.wav --check
python bench/smoke_moulsot.py bench/recordings/kitchen.wav
```

`--check` only validates the input and endpoint setting; it does not prove endpoint
availability. The actual run uploads the whole WAV to MoulSot and allows up to
600 seconds for its queue/inference. Groq fallback is always disabled, regardless
of `STT_PRIMARY`. JSON output records the transcript, audio hash/duration, provider,
elapsed time, and call outcome. The human verdict starts as `pending`.

Listen while reading the transcript. Check whether it retains the order details,
corrections, and digits, and whether it invents words during noise. Decide what
needs investigation before recording the bank. A single recording is an early
feasibility check; it does not establish general accuracy or live endpoint timing.

## 4. Language, voice bank and full conversation — deferred

After the MoulSot smoke is reviewed:

1. Fill and review Darija number words first, then enable the missing language fixture.
2. Use actual regional pizza-call phrasing; aim for short prompts around eight words.
3. Review every marker and prompt before clearing review flags.
4. Record the bank in one room/session at a consistent mic distance. Say digits
   with level, continuing intonation so concatenated phone numbers sound natural.

The current pizza manifest has **66 clips**, including both `digit_0.wav` and
`number_0.wav`: 6 prompts, 7 responses, 6 carriers, 10 digits, 21 numbers (0–20),
15 enum values and one connector.

Address TTS selection is deferred. The current complete delivery-order flow still
requires an address readback and blocks without it. The router and ASR smoke do
not need TTS. Before a complete spoken delivery demo, implement an explicit
address confirmation alternative or configure TTS; do not silently omit a critical
address while reporting the full order confirmed.

## Verification

66 offline tests pass, one native-Darija numeral fixture remains skipped.
Coverage includes strict request structure, model override, startup failure,
truncation/refusal, gold-data review gates, scoring, WAV checks and no-fallback
MoulSot failures. The live three-model probe is separate evidence.
