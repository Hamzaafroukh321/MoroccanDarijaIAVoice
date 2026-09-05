# Darija Voice

Source snapshot: reusable task engine with pizza and clinic demos, MoulSot speech
recognition, Groq structured task routing, and the selected Darija XTTS voice.
See [current capabilities and limitations](docs/PROJECT_STATUS.md) and
[how to add a task](docs/ADDING_A_TASK.md).

Raw recordings, credentials, downloaded models, browser artifacts and generated
`bench/results/` evidence stay local. Links into that results directory in the
development reports refer to local diagnostics and are not included in this repository.

A browser microphone task-completion engine for Moroccan Darija, with one shared
state across speakers and a reproducible evaluation harness: **The Family Order
Test**. Pizza ordering is the first domain; clinic appointments exercise the same
engine with dates, times and patient names.

**Implementation: M3–M10 code is now in place. Research acceptance remains
pending.** Diagnostic timings are not validated accuracy benchmarks. Human language
review, audio-bank recordings, provider setup, and annotated scenario recordings
are still needed for the reviewed voice task and benchmark. A separate synthetic
pickup demo and a fictional clinic preference preview are available for trying
the interaction now. Diagnostic measurements are recorded separately from
research acceptance results.

## Run

Next testing sequence: **router → owner-supplied 30-case text eval → MoulSot kitchen
smoke → language review and recording**. See [TESTING_NEXT.md](TESTING_NEXT.md) for
commands and the September 5 live provider check. Address TTS work is deferred.

Python **3.11** is required. The dependency list remains exactly as specified.
On Windows:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Fill in provider settings in .env before starting a voice demo.
python scripts/check_config.py configs/pizza.json
python scripts/check_config.py configs/clinic.json
python -m engine.server
```

For another machine, create a Python 3.11 environment and install
`python -m pip install -r requirements.txt`. Open [the local app](http://127.0.0.1:8000).
The **microphone check** works independently of external services. The **voice
task** becomes selectable when its setup checks pass. Domain selection switches
between pizza and clinic; task state and recorded responses are session-local.

Inspect an existing recording without uploading it:

```powershell
python -m bench.inspect_audio path/to/recording.wav
python -m bench.inspect_audio path/to/recording.wav --vad --domain pizza --output path/to/report.json
```

The optional VAD pass uses installed local Silero and the demo's endpoint settings.
Reports contain signal levels, exact zero/rail fractions and detected boundaries;
they do not measure intelligibility or recommend automatic gain/threshold changes.
Up to five mono16kHz PCM16 WAVs (180 seconds combined) are checked before local
model loading. Source files are preserved; ASR, router and TTS are not called.

## Try a live conversation

Open [the pizza conversation demo](http://127.0.0.1:8000/?domain=pizza&mode=demo),
click **Start conversation**, and allow the microphone. Use headphones, speak
Darija, then pause about 1.4 seconds. The assistant asks for missing details,
reads back quantity/size/toppings/drink, and accepts confirmation only after the
readback finishes. Correct a detail to get an updated summary. **End session**
releases the microphone. Sessions are limited to five minutes.

While a turn is pending, the status identifies transcription, task interpretation
or voice preparation. Elapsed waiting time appears after a short delay, followed
by a “Still waiting” message for longer waits. **End session** stays available;
these indicators show progress without promising a completion time. See the
[slow-turn checks](bench/results/slow_turn_experience_20260905.md).

If the router returns an invalid interpretation, the demo preserves your details,
asks the current question and lets you continue. Repeated failures use the bounded
handoff policy. Provider outages and rate limits still produce a distinct error.
[Recovery evidence](bench/results/router_output_recovery_20260905.md) covers both
domains with offline model responses; it is not a speech accuracy measurement.

An invalid voice-provider setting shows a demo setup issue while microphone check
remains available. Ending or disconnecting a voice session cancels pending local
work and closes its provider clients; normal disconnects are saved as interrupted.
[Connection recovery checks](bench/results/server_connection_recovery_20260905.md)
include actual local WebSocket transport with offline providers.

The [clinic conversation preview](http://127.0.0.1:8000/?domain=clinic&mode=demo)
uses the same transactions, clarification, staged changes and confirmation
lifecycle for a fictional doctor, date and time. It does not check availability
or book appointments.

Preview field overrides preserve the domain's slot IDs. Duplicate fields, renamed
IDs and overrides for excluded fields are rejected before a conversation. This
prevents a malformed profile from dropping a required choice; see
[profile validation evidence](bench/results/profile_identity_20260905.md).

Before changing a demo profile, validate its merged configuration offline:

```powershell
python scripts/check_config.py configs/pizza.json --demo
python scripts/check_config.py configs/clinic.json --demo
```

This checks task-field coverage, questions, labels, readback coverage and date/time
format compatibility. It loads local environment settings but makes no provider
requests. A missing question or omitted readback field produces an actionable
setup error before speaking. It does not verify credentials, voice quality or
native language; `--ready` remains the separate reviewed research-bank check.
See [demo preflight evidence](bench/results/demo_preflight_20260905.md).

This mode uses the configured MoulSot Space with no Whisper fallback, the Groq
router, and the speech provider selected by `DEMO_TTS_PROVIDER`. The preferred
voice is **Darija XTTS 2.1**, selected after the owner's listening comparison.
The voice is a synthetic Arabic preview; its Moroccan pronunciation and the
draft phrases need your review. Shared ZeroGPU queues/quota can delay or block
turns. No actual pizza order is placed.

Set `DEMO_TTS_PROVIDER=darija_xtts` in `.env` and restart. This uses the author's
[hosted demo](https://huggingface.co/spaces/medmac01/Darija-Arabic-TTS), currently
running `medmac01/darija_xtt_2.0/model_2.1.pth` with its default speaker reference
and temperature 0.65. This is the newer checkpoint auditioned here, not the older
`xtt2_darija_v0.1`. No Azure or ElevenLabs subscription is needed. `HF_TOKEN` is
optional but recommended for authenticated shared GPU quota; both ASR and TTS
now depend on Hugging Face availability. Hosted deployments can change upstream.
The adapter converts 24 kHz output into the app's 16 kHz WAV, caches replies,
bounds each request to 120 seconds, and reports failure without voice fallback.
It uses existing dependencies. The base XTTS [Coqui Public Model License](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt)
limits the model and output to non-commercial use; include the license link with
shared generated audio. The demo remains synthetic and excluded from research
evaluation. A preferred sample is not a comprehensive pronunciation evaluation.

The latest [two-turn recorded-audio diagnostic](bench/results/recorded_two_turn_20260905.md)
delivered complete reply WAVs through actual MoulSot, Groq and XTTS. It exposed a
repeated-question bug, subsequently fixed and checked with one saved-context
router replay; the conversation remained unconfirmed. Earlier transport reports
retain their failures for compression/cleanup diagnosis; some old audio artifacts
contain only final PCM fragments and are not playback evidence. See
[transport history](bench/results/transport_fix_20260905.md).
Playback acknowledgements in transport diagnostics are simulated. A separate
[browser lifecycle check](bench/results/conversation_lifecycle_20260905.md) now
exercises real WebAudio decoding and completion events with synthetic silence and
a mocked socket in both domains. Physical microphone and speaker quality still
need verification.

The [multi-turn replay guide](bench/VOICE_REPLAY.md) explains how to test a request,
correction and confirmation in one session using supplied WAVs. Manifests validate
locally by default and require `--execute` to contact the local voice server.
Each turn waits for a complete reply and matching server acknowledgement; optional
state/action assertions remain diagnostic. Offline local TCP tests complete this
three-turn sequence in both domains with fixture providers, without establishing
live MoulSot accuracy or device playback quality.

Both domains support explicit cancellation of pending changes and of retained
request text after a question is resolved. Saved values survive cancellation;
new questions use their own request context and stale IDs cannot clear newer
work. See [cancellation evidence](bench/results/retained_request_cancellation_20260905.md).
Its two live router controls used English fixtures, so Darija cancellation
recognition remains unverified.

Add another flat task with matching base and preview configs; no engine registry
edit is needed. The Domain selector discovers validated files automatically.
See [adding a task](docs/ADDING_A_TASK.md) and the
[temporary third-domain evidence](bench/results/custom_domain_extension_20260905.md).
This extension does not provide arbitrary collection adapters or external actions.

Prepare exact common replies before a demo with a bounded generation budget:

```powershell
python -m bench.prepare_speech --domain clinic
python -m bench.prepare_speech --domain clinic --execute --max-misses 3 --max-characters 100
```

The first command only plans. Execution skips valid cached clips, stops on the
first provider failure and never calls ASR or the router. It does not precompute
dynamic readbacks. Within a voice session, concurrent requests for the same missing
clip share one generation; validated WAV files are published atomically. There is
no cross-process request coalescing. Change `tts_cache_revision` in the demo voice
settings if an upstream checkpoint or default speaker changes: remote freshness
cannot be inferred from an unchanged endpoint. See the
[cache preparation results](bench/results/speech_cache_milestone_20260905.md).

If the provider reports **ZeroGPU runs limit exceeded**, browser sign-in alone
does not authenticate this Python server. Create a Hugging Face Read token and
save `HF_TOKEN=...` in the local `.env`, then restart the server. The adapter sends
it only to HTTPS `*.hf.space` Gradio endpoints, never to Groq or custom services.
Authenticated requests use the account's quota, which is still limited; wait for
the reset if that quota is also exhausted. The app now displays recognized
ZeroGPU limit messages without exposing raw provider errors or credentials.
See [Hugging Face API authentication](https://huggingface.co/docs/hub/spaces-api-endpoints).

For Mouna, create an Azure **Speech** resource using the **Free F0** tier. From
its **Keys and Endpoint** page, save the key and matching region in your local
`.env` (do not commit or paste the key in chat):

```dotenv
DEMO_TTS_PROVIDER=azure
AZURE_SPEECH_KEY=your_resource_key
AZURE_SPEECH_REGION=your_resource_region
```

For example, use `westeurope` only if that is your resource's region. Restart
the server afterward. The demo stays disabled until the Azure key and region are
configured. The REST adapter uses escaped SSML and 16 kHz mono PCM WAV, caches
repeated replies, and counts all generated audio as synthetic. It reports quota
and authentication failures without retry loops or switching to another voice.
The F0 tier currently includes 500,000 neural TTS characters per month and has a
20 requests/minute limit. Real Darija pronunciation and latency still require a
live listening test with the owner's Azure resource; offline adapter tests do
not establish voice quality. See [Azure pricing](https://azure.microsoft.com/en-us/pricing/details/speech/)
and [Speech quotas](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-services-quotas-and-limits).

The previous Saudi preview remains available with `DEMO_TTS_PROVIDER=groq`
(also the fallback configuration when this environment variable is absent).
It uses Groq Orpheus `fahad` and requires acceptance of the model terms.

For the optional Moroccan voice, the adapter also supports ElevenLabs **Ghizlane —
Warm, Natural and Encouraging** (`OfGMGmhShO8iL9jCkXy8`) with
`eleven_multilingual_v2`. After verifying account access, set
`ELEVENLABS_API_KEY` and `DEMO_TTS_PROVIDER=elevenlabs` in `.env`, then restart.
The key needs Text to Speech access only. The provider returns 16 kHz PCM, which
the adapter wraps as WAV for the existing player; no additional audio dependency
is needed. Speech failures do not silently switch back to the Saudi voice.
Library-voice API access depends on the ElevenLabs plan. A working website preview
does not prove API access. Existing research audio-bank requirements still apply.

Demo settings live in `configs/demo/voice.json`. Draft number mappings and phrases
apply only to this opt-in mode; research review flags and WAV-bank gates remain.
Common speech is cached locally. Microphone segments go to MoulSot, transcripts
go to the Groq router, and response text goes to the selected speech provider. Session JSON is saved
as `bench/results/demo_session_*.json`, with `evaluation_eligible: false` and
synthetic output accounting. It is excluded from research session naming.

The live provider transport smoke used an excerpt of the owner's recording:
MoulSot transcription → size update → synthesized toppings question. Its playback
acknowledgements were simulated; see `bench/results/demo_live_transport_check.json`.
Offline tests additionally cover correction, mandatory readback acknowledgement,
confirmation, provider failure, speech caching and streaming WAV decoding.

## What is built and verified

| Milestone | Implementation | Remaining acceptance |
|---|---|---|
| M3 | Silero VAD, base wait, three added holds, partial-tail caching, minimum speech duration and bounded segments | Real Darija endpoint timing check |
| M4 | Hosted MoulSot JSON/Gradio adapters, Groq fallback, persistent shared quota ledger | Live provider/recorded-speech validation; local MoulSot is incompatible with the pins |
| M5 | Numeral/phone/currency/date normalization, JSON router, atomic state operations, versioned confirmation | Reviewed Darija numeral dictionary and live router validation |
| M6 | Reviewed WAV-bank recorder, readback assembly, restricted TTS interface, playback acknowledgements, barge-in | Owner voice bank and chosen TTS endpoint; real corrected voice order |
| M7 | Two-of-three unusable-audio signals, clarification and handoff | Real overlapping-human-speech recovery |
| M8 | Five metrics, strict scenario loader, real-time replay and provenance; 20 pizza stubs | Consented audio and manual gold annotations |
| M9 | 28 configurations, CSV/SVG generation, four baselines and measured-only tuning | Run on completed recordings; no chart or tuning is fabricated |
| M10 | Clinic config, ten scenario stubs, generic date/time/name readback and state test | Owner doctor/reason choices, audio, and real clinic flow |

Verification currently includes **99 passing offline tests and one explicitly
skipped test** for missing native-speaker-reviewed Darija numeral fixtures. It also
includes a mocked-provider WebSocket smoke: streamed greeting → slot updates →
readback acknowledgement → affirmation → persisted completed session. This tests
the transport and state contract, not real ASR quality or acoustic playback.

```powershell
python -m pytest -q
```

Tests live only in the four files specified by the build contract. They use no
live APIs and no new test framework.
The installed Silero model was also smoke-tested on a real PCM silence frame.

## Configure speech services

Copy `.env.example` to `.env` and set `GROQ_API_KEY`. Keep the key out of configs and
source control. Choose `STT_PRIMARY=moulsot` or `STT_PRIMARY=groq`.

For the official hosted MoulSot Space:

```dotenv
MOULSOT_ENDPOINT=https://atlasia-moulsot-v0-3.hf.space
```

`stt.moulsot_protocol` defaults to `gradio`; the adapter uses upload, queued
inference and SSE completion through httpx. Its API shape was checked against the
Space's published API metadata. To use an operator-hosted endpoint, set the
protocol to `json`: it receives multipart `file` containing a WAV and returns
`{"text": "...", "confidence": null}`. Confidence is optional, and a supplied
confidence must be finite and between zero and one.

**Local MoulSot compatibility:** the current model is Qwen3-ASR and documents
`qwen_asr`, while the locked dependencies pin Transformers 4.46.0 and do not include
that package. Local loading therefore produces an explicit setup error; it does
not download an incompatible model or silently install another runtime. The
original `.env.example` comment is preserved from the specification, but its
blank-endpoint/local-load assumption is not currently implementable with the pins.
See the [MoulSot model card](https://huggingface.co/atlasia/moulsot.v0.3).

The router defaults to `openai/gpt-oss-120b`. Override it using
`GROQ_ROUTER_MODEL` in `.env`; restart after edits. Startup checks every configured
domain router against Groq's models endpoint and refuses to start on unavailable
models, invalid authentication, or an unsuccessful availability check. Without a
key, microphone capture remains available and startup logs a warning.

Router requests use `json_schema` with `strict: true`, generated from the Pydantic
response model with allowed slot IDs. The transcript is a separate user message.
Local validation still checks operations, values, and contradictory flags;
truncated/refused responses are rejected. Groq speech uses
`whisper-large-v3-turbo` and verbose transcription responses. Missing confidence
is recorded as **unavailable**, never invented. Groq's exponentiated mean token
log-probability is explicitly labeled as a geometric-mean probability proxy; it
is not a calibrated arithmetic mean token confidence. These distinctions matter
when interpreting overlap results. See [Groq speech documentation](https://console.groq.com/docs/speech-to-text)
and [supported models](https://console.groq.com/docs/models).

The quota ledger under `bench/results/groq_usage.json` reserves requests before
sending them, applies all four specified rolling windows, survives restarts and
uses a process lock. Router calls conservatively share request limits with ASR.
It cannot account for other applications using the same account. Failed requests
remain counted; no automatic quota-reset or bypass occurs.

```powershell
python -m engine.stt path-to-16khz-mono.wav --domain pizza
python -m engine.pipeline --domain pizza --text "two large pizzas" --text "quantity three"
```

The typed-transcript command is an M5 diagnostic. It shows state/action changes
and cannot claim that a spoken readback has been completed.

## Review and record the voice bank

1. Replace every `[[TODO_DARIJA: ...]]` / owner-choice placeholder in the configs.
   Supply reviewed Darija numeral mappings; the implementation never invents them.
2. Verify the marker lists in `engine/lexicon.py`, then change its review flag.
3. Verify all review flags, slot prompt/carrier phrases, and
   `audio_output.fragments`. Clinic enum choices must have matching value WAV
   entries. Set review flags false only after the content was actually checked.
4. Start the local server and run:

```powershell
python scripts/record_audio_bank.py configs/pizza.json
python scripts/record_audio_bank.py configs/clinic.json
python scripts/check_config.py configs/pizza.json --ready
```

The script refuses unreviewed configs. It lists every prompt/carrier/response and
required value fragment, then opens the existing browser microphone recorder for
each phrase. WAVs must be 16 kHz mono PCM16. Existing files are kept unless
`--overwrite` is explicitly supplied. No microphone package is added.

Readback uses human fragments separated by the configured 120 ms gap. Phones are
read digit by digit; dates use ISO year–month–day order with recorded separators;
times use hour/minute digits. Quantities and enum values are prerecorded. There
is no TTS fallback for prompts, responses, digits, quantities or enum values.

For free-text addresses and patient names only, configure
`audio_output.tts_enabled` and `tts_endpoint`. This operator-provided endpoint must
accept JSON `text`, `slot`, `language`, and `spell`, and return a PCM16 mono 16 kHz
WAV. `spell: true` requires spelling the address, not reading it as an ordinary
sentence. No TTS service has been selected or deployed by this repository.
Generated synthetic sample counts and interrupted playback are logged separately.
The reported synthetic fraction uses generated output samples, including silence
gaps; it is not an estimate of what was actually heard before barge-in.

## Voice-session behavior

The seven states are IDLE, LISTENING, PROCESSING, SPEAKING, CLARIFYING, CONFIRMING
and DONE. Any speaker can update the same task. The router validates the entire
operation batch before state mutation. Filling slots never finalizes a task:
affirmation is accepted only after the matching readback finishes. Corrections,
negation and barge-in invalidate stale confirmation.

VAD remains active during playback. Three consecutive speech frames stop the
response, and those first 96 ms are retained as the start of the incoming turn.
Playback carries an ID so late acknowledgements cannot confirm interrupted audio.
Partial ASR requests run outside the frame loop, are cached by audio content, and
use bounded waits. Frame arrival and new speech continue while routing is pending.

Unusable audio needs two of low confidence, low token rate and high RMS variation.
Missing confidence does not fire the low-confidence signal. After **exceeding**
three clarifications, the fourth unusable segment hands off, following Section
6.5. Barge-in alone is not evidence of simultaneous human overlap.

Local capture WAVs and session JSON are under `bench/recordings/` and
`bench/results/`. A stopped/disconnected task is marked interrupted, not completed.
The server binds to loopback by default; no telephony, database or diarization is
included.

## Collect and run the benchmark

The 30 scenario JSON files are **stubs**. The owner must supply consented recordings,
transcripts, real gold states, utterance boundaries and correction events. Record
at least the diversity and real-room conditions specified in [SPEC.md](SPEC.md).
Do not mark `annotations_reviewed` true before listening and annotating.

Each scenario needs `reference_date` in YYYY-MM-DD form, the date on which relative
date expressions should be interpreted. A correction event has:

```json
{"utterance_index": 1, "slot": "size", "value": "small"}
```

That utterance must also have `is_correction: true`. Use the last annotated value
per corrected slot for correction success. Recordings live at the paths declared
in the scenarios, such as `bench/recordings/pizza/s01.wav`.

```powershell
python bench/run_bench.py --domain pizza --check
python bench/run_bench.py --domain pizza
python bench/sweep_endpointing.py --domain pizza --check
python bench/sweep_endpointing.py --domain pizza
python bench/run_bench.py --domain clinic
```

Missing data produces an explicit NOT RUN report and exit code 2. No gold
transcript, speaker label, or correction annotation is fed to the voice engine.
The replay runs at real time and simulates the actual duration of response WAVs;
it does not accelerate away partial-ASR or playback timing. The recorded input
starts after the greeting. This fixed recording replay cannot adapt the humans'
responses to a changed prompt schedule; report that limitation with results.

Metrics are TSR/CSR/CEA in percent, FCR as cutoffs per gold utterance, and MEL in
milliseconds. Undefined denominators produce null/N/A. TSR additionally requires
a completed, confirmed session; merely filling slots in an interrupted task earns
no success. The report also includes per-slot CEA, missed boundaries, unmatched
endpoints, coverage and synthetic output fraction.

Endpoint matching uses the earliest uncompleted gold utterance touched by a
segment. A false cutoff does not consume its gold boundary. A merged segment can
consume at most one boundary; remaining boundaries are reported missed. MEL alone
must always be read alongside coverage. These matching conventions resolve gaps
in the initial spec and are documented rather than hidden in the implementation.

The sweep evaluates seven silence thresholds × four hold modes, writes a CSV and
standalone SVG, then runs the Groq ablation and writes all four baseline results.
There is **no ASR fallback inside benchmark runs**, so an ASR label cannot conceal
a different provider. Provider transport failures invalidate scoring. Input config,
engine hashes, audio hashes and annotation hashes accompany results.

Tuning chooses the lowest FCR with full boundary coverage and MEL within the
configured ceiling, breaking ties by TSR then latency. If no setting qualifies,
no tuning is written. The default ceiling is a configurable research decision,
not a measured optimum. Selection and evaluation on the same scenarios are
explicitly labeled **in-sample**. Checkpoints preserve partial sweep work.

## Sharing and remaining decisions

Text-only router evaluation requires no audio annotations, but does require the
owner's transcripts and reviewed expected operations/flags. It makes live Groq
calls; it is not an offline/local-inference benchmark. See
`bench/router_cases/pizza.json` and `bench/eval_router.py`. The MoulSot qualitative
smoke runner (`bench/smoke_moulsot.py`) explicitly disables Groq fallback.

[QUESTIONS.md](QUESTIONS.md) records the remaining owner inputs and implementation
conventions. [SPEC.md](SPEC.md) preserves the original source verbatim.

Credit **Atlasia / MoulSot v0.3** as the underlying ASR. This project evaluates and
improves task handling around ASR; it does not claim to beat the ASR model. A
LinkedIn development update can show the implemented flow and test evidence now.
A performance post or completed voice demo needs real recordings, the measured
baselines, the tradeoff chart, and an honest failure example first.
