# Darija Voice Task-Completion Engine — Complete Build Specification

**Version:** 1.0
**Status:** Authoritative. This document is the single source of truth.
**Audience:** An AI coding agent (Codex) and the human project owner.

---

## 0. RULES FOR THE CODING AGENT — READ FIRST

These rules override any instinct you have. Violating them breaks the project.

1. **Build only what is in this document.** Do not add features, endpoints, models, abstractions, UI elements, or "nice to haves" that are not written here. If you think something is missing, add a line to `QUESTIONS.md` and continue with what is specified.
2. **Do not add dependencies.** The complete dependency list is in Section 4.3. If you believe another library is required, stop and write it in `QUESTIONS.md`.
3. **Never write Darija text yourself.** You do not speak Darija. Every Darija string lives in `configs/<domain>.json` or `engine/lexicon.py`, and every one of them is marked `"NEEDS_HUMAN_REVIEW": true` until the project owner verifies it. If you need a new Darija phrase, insert the literal placeholder `"[[TODO_DARIJA: description of what this should say]]"` and never guess.
4. **Do not implement speaker diarization or voice separation.** It is explicitly out of scope. See Section 6.3.
5. **Do not implement telephony, SIP, Twilio, or phone numbers as a channel.** Browser microphone only.
6. **Do not add a database.** State lives in memory during a session and is written to JSON files on disk.
7. **Do not use text-to-speech as the primary output.** Pre-recorded WAV files are primary. TTS is a fallback only, and only where Section 9.2 permits it.
8. **Follow the milestone order in Section 14.** Do not start milestone N+1 before milestone N passes its Definition of Done.
9. **Every threshold, timeout, and constant must come from a config file.** No magic numbers in code. If this document gives a default, put it in the config with that default.
10. **Write tests as specified in Section 12.** Do not invent additional test frameworks.
11. When this document and your training data disagree, **this document wins.**

---

## 1. What This Project Is

### 1.1 One-sentence description

A general-purpose engine that lets a Moroccan Darija speaker complete a structured task by voice, in a realistic messy environment (multiple family members talking in turns, background noise, French/Darija code-switching, mid-sentence corrections), plus a benchmark that measures how well it does.

### 1.2 The two deliverables

| Deliverable | What it is |
|---|---|
| **The Engine** | Domain-agnostic. Handles listening, knowing when someone finished, collecting information across multiple speakers, corrections, normalizing numbers, confirming, and responding. |
| **The Benchmark** | A named, reusable test suite ("The Family Order Test") with 4 metrics, 20 scenarios, and a published results table. |

The pizza ordering domain is **test case #1**, not the product. The clinic appointment domain is **test case #2** and exists to prove the engine is genuinely general.

### 1.3 What this project is NOT

- Not a new speech recognition model. It consumes MoulSot v0.3.
- Not a new language model. It consumes a hosted LLM as a router only.
- Not a phone call system.
- Not a system that separates simultaneously overlapping voices.
- Not a chatbot. It fills slots and completes a task. It does not have open conversation.

### 1.4 The intellectual contribution

Three things, in order of importance:

1. **Darija-aware endpointing.** Existing voice agents decide "the human finished talking" using silence thresholds tuned on English. This project introduces hesitation-hold and continuation-hold rules driven by Darija discourse markers, and measures the improvement.
2. **Task-level evaluation instead of Word Error Rate.** Darija has no standardized orthography, so WER penalizes valid spelling variation. This project measures whether the *task* succeeded, not whether letters matched.
3. **Published failure analysis.** A documented list of exactly where a state-of-the-art Darija ASR breaks in a realistic multi-speaker task, which nobody has published.

---

## 2. Core Design Decisions (Locked)

These have been decided. Do not revisit them.

| Decision | Choice | Reason |
|---|---|---|
| Who is speaking | **Ignored.** One shared task state that any voice can modify. | A real shop employee does not care who said "extra cheese." Diarization is a research problem and would kill the project. |
| Simultaneous overlapping speech | **Detected and handled politely, not resolved.** | Separating overlapping voices is out of scope. Saying "one at a time please" is what a human employee does and is a better demo moment. |
| Primary voice output | **Pre-recorded human WAV files.** | Free, instantly authentic Darija, zero latency, and it removes the weakest link in the entire stack. |
| ASR | **MoulSot v0.3** primary, **Groq Whisper large-v3-turbo** fallback | MoulSot is state of the art for Darija and handles code-switching. Groq is a cheap, always-available fallback. |
| LLM role | **Slot-delta router only.** Outputs JSON. Never generates spoken text. | Removes hallucination risk entirely. The bot can only ever say approved lines. |
| Turn state | **Explicit state machine**, not free conversation | Predictable, testable, measurable. |

---

## 3. Architecture

### 3.1 Data flow

```
Browser mic (16 kHz mono PCM)
        │  WebSocket, 32 ms frames
        ▼
┌─────────────────────────────────────────────┐
│ endpointing.py                              │
│  Silero VAD per frame                       │
│  + Darija hesitation-hold                   │
│  + Darija continuation-hold                 │
│  + digit-sequence hold                      │
│  → emits SEGMENT when the human is finished │
└─────────────────────────────────────────────┘
        │ audio segment (bytes)
        ▼
┌─────────────────────────────────────────────┐
│ stt.py                                      │
│  MoulSot v0.3  →  transcript + confidence   │
│  (fallback: Groq whisper-large-v3-turbo)    │
└─────────────────────────────────────────────┘
        │ raw transcript
        ▼
┌─────────────────────────────────────────────┐
│ overlap.py                                  │
│  Is this unusable (overlap / noise)?        │
│  If yes → CLARIFY, skip the rest            │
└─────────────────────────────────────────────┘
        │ clean transcript
        ▼
┌─────────────────────────────────────────────┐
│ normalize.py                                │
│  Darija numerals, French numerals,          │
│  phone formats, money, dates                │
└─────────────────────────────────────────────┘
        │ normalized transcript
        ▼
┌─────────────────────────────────────────────┐
│ router.py   (LLM, JSON out only)            │
│  input: slot definitions + current state    │
│         + normalized transcript             │
│  output: list of slot operations            │
└─────────────────────────────────────────────┘
        │ ops: [{op, slot, value}]
        ▼
┌─────────────────────────────────────────────┐
│ state.py                                    │
│  Apply ops to task state                    │
│  Decide next action:                        │
│    ask missing slot / readback / done       │
└─────────────────────────────────────────────┘
        │ action id
        ▼
┌─────────────────────────────────────────────┐
│ responder.py                                │
│  Look up pre-recorded WAV for action id     │
│  Stream to browser                          │
│  Barge-in: stop instantly if VAD fires      │
└─────────────────────────────────────────────┘
```

### 3.2 Module responsibilities (exact)

| File | Responsible for | Must NOT do |
|---|---|---|
| `engine/server.py` | FastAPI app, WebSocket endpoint, session lifecycle | Any audio logic |
| `engine/pipeline.py` | Orchestrating the flow above, holding the state machine | Any I/O with external APIs |
| `engine/endpointing.py` | VAD + all hold rules, emitting segments | Transcription |
| `engine/stt.py` | Calling MoulSot / Groq, returning `(text, confidence)` | Normalization |
| `engine/overlap.py` | Deciding if a segment is unusable | Fixing it |
| `engine/normalize.py` | Numbers, phones, money, dates → canonical form | Slot logic |
| `engine/router.py` | Building the LLM prompt, calling it, parsing/validating JSON | Applying ops |
| `engine/state.py` | Task state, applying ops, deciding next action | Talking |
| `engine/responder.py` | Audio bank lookup, streaming, barge-in | Deciding what to say |
| `engine/lexicon.py` | All Darija marker word lists, one place | Logic |
| `engine/config.py` | Loading + validating domain configs against schema | Everything else |

---

## 4. Tech Stack (Locked)

### 4.1 Runtime

- Python **3.11**
- Node not required. Frontend is plain HTML/JS, served statically by FastAPI.

### 4.2 Services

| Service | Purpose | Cost |
|---|---|---|
| MoulSot v0.3 (`atlasia/moulsot.v0.3`) | Primary ASR | Free (self-hosted or HF Space) |
| Groq API, `whisper-large-v3-turbo` | Fallback ASR | Free tier; paid is $0.04/audio hour |
| Groq API, an instruct model | JSON slot router | Free tier |
| Silero VAD | Voice activity detection | Free, local |

Groq free tier limits to respect: 20 requests/minute, 2,000 requests/day, 7,200 audio seconds/hour, 28,800 audio seconds/day, 25 MB max upload. The engine must not exceed these; implement a simple client-side rate limiter in `stt.py`.

### 4.3 Complete dependency list

`requirements.txt` — this is the whole list. Do not add to it.

```
fastapi==0.115.0
uvicorn[standard]==0.32.0
websockets==13.1
numpy==2.1.0
torch==2.5.1
torchaudio==2.5.1
silero-vad==5.1.2
transformers==4.46.0
soundfile==0.12.1
httpx==0.27.2
pydantic==2.9.2
python-dotenv==1.0.1
jsonschema==4.23.0
pytest==8.3.3
```

If a needed package is genuinely absent, write it in `QUESTIONS.md`. Do not install it.

### 4.4 Environment variables

`.env.example` must contain exactly:

```
GROQ_API_KEY=
STT_PRIMARY=moulsot          # moulsot | groq
MOULSOT_ENDPOINT=            # leave blank for local transformers load
LOG_LEVEL=INFO
```

---

## 5. Repository Structure (Exact)

Create this tree. Do not add directories.

```
darija-voice-engine/
├── README.md
├── SPEC.md                        # this document, copied in
├── QUESTIONS.md                   # agent writes blockers here
├── requirements.txt
├── .env.example
├── .gitignore
│
├── configs/
│   ├── schema.json                # JSON Schema for domain configs
│   ├── pizza.json                 # domain 1
│   └── clinic.json                # domain 2
│
├── engine/
│   ├── __init__.py
│   ├── server.py
│   ├── pipeline.py
│   ├── endpointing.py
│   ├── stt.py
│   ├── overlap.py
│   ├── normalize.py
│   ├── router.py
│   ├── state.py
│   ├── responder.py
│   ├── lexicon.py
│   └── config.py
│
├── audio/
│   ├── pizza/                     # pre-recorded WAV, 16 kHz mono
│   └── clinic/
│
├── web/
│   ├── index.html
│   ├── app.js
│   └── recorder-worklet.js
│
├── bench/
│   ├── scenarios/
│   │   ├── pizza/                 # 20 scenario JSON files
│   │   └── clinic/                # 10 scenario JSON files
│   ├── recordings/                # WAV per scenario
│   ├── run_bench.py
│   ├── metrics.py
│   ├── sweep_endpointing.py
│   └── results/
│
├── scripts/
│   ├── record_audio_bank.py
│   └── check_config.py
│
└── tests/
    ├── test_normalize.py
    ├── test_state.py
    ├── test_endpointing.py
    └── test_router_schema.py
```

---

## 6. The Engine — Detailed Behaviour

### 6.1 State machine

States, exactly these seven:

```
IDLE → LISTENING → PROCESSING → SPEAKING → LISTENING …
                        ↓
                   CLARIFYING → LISTENING
                        ↓
                   CONFIRMING → DONE
```

| State | Entered when | Behaviour |
|---|---|---|
| `IDLE` | Session opens | Waiting for user to press start |
| `LISTENING` | Start pressed, or bot finished speaking | VAD running, buffering audio |
| `PROCESSING` | Endpoint emitted | STT → overlap → normalize → router → state |
| `SPEAKING` | An action was chosen | Streaming a WAV. **Barge-in enabled.** |
| `CLARIFYING` | Overlap or unusable segment detected | Play the "one at a time" line, return to LISTENING |
| `CONFIRMING` | All required slots filled | Play full readback, wait for yes/no |
| `DONE` | User confirmed | Write final state to `bench/results/`, close |

**Barge-in rule:** While in `SPEAKING`, if VAD reports speech for 3 consecutive frames (≈96 ms), stop audio playback immediately and transition to `LISTENING`. This is mandatory.

### 6.2 Endpointing — the core contribution

This is the most important algorithm in the project. Implement it exactly.

**Base loop**, per 32 ms frame:

```
prob = silero_vad(frame)
is_speech = prob > vad_threshold          # default 0.5
```

On a speech → silence transition, start a silence timer and compute a **required wait**:

```
required_wait = base_silence_ms                    # default 900

# Run STT on the buffer so far to inspect the tail (cheap: reuse last result
# if the buffer has not grown, otherwise call STT once)
tail = last_two_tokens(partial_transcript)

if tail matches HESITATION_MARKERS:
    required_wait += hesitation_hold_ms            # default 600

if tail matches CONTINUATION_MARKERS:
    required_wait += continuation_hold_ms          # default 900

if buffer tail contains an in-progress digit sequence:
    required_wait += digit_hold_ms                 # default 700

required_wait = min(required_wait, max_wait_ms)    # default 2500
```

Emit the segment when `silence_duration >= required_wait`.
If speech resumes before that, cancel the timer and keep buffering.

Also enforce `min_speech_duration_ms` (default 250) — segments shorter than this are discarded as noise blips.

**Why `base_silence_ms` defaults to 900 and not 500:** 500 ms is the common English-tuned default. The hypothesis this project tests is that Darija speakers pause longer, especially before naming an item or a number. The benchmark sweep (Section 11.4) is what proves or disproves this. Do not hardcode 900 as truth — it is a starting default that the sweep will replace with a measured value.

**All six parameters live in the domain config** under `endpointing`. See Section 7.3.

### 6.3 Multi-speaker handling

**There is one task state per session. Any utterance from any voice can modify it.**

Do not attempt to identify speakers. Do not attempt voice embeddings. Do not attempt diarization.

Practical consequence: if the mother says "wahed large" and the brother says "zid fromage," these are two segments that produce two sets of operations against the same state object. This is correct and intended behaviour.

### 6.4 Corrections

Corrections are detected by the router LLM, not by keyword matching in code. The router receives the current state and must output `set` or `remove` operations that overwrite it.

The Darija correction markers in `lexicon.py` are passed into the router prompt as hints. They are **not** used for programmatic matching.

The state layer must accept an operation that overwrites an already-filled slot without complaint. Never say "you already told me that."

### 6.5 Overlap detection

Implement three cheap heuristics in `overlap.py`. A segment is flagged unusable if **any two** fire:

1. **Low ASR confidence.** Mean token confidence below `overlap.confidence_floor` (default 0.45).
2. **Abnormal token rate.** Fewer than `overlap.min_tokens_per_second` tokens per second of speech (default 1.2). Long noisy audio producing few words indicates unusable input.
3. **High energy variance.** Frame-level RMS standard deviation above `overlap.rms_std_ceiling` (default 0.18, computed on normalized float32 audio).

When flagged, transition to `CLARIFYING` and play the "one at a time" line. Do not attempt to recover content from the segment. Increment the session's `clarify_count`.

If `clarify_count` exceeds 3 in one session, play the handoff line and go to `DONE` with `status: "handoff"`.

### 6.6 Confirmation and readback

Never finalize on the last slot being filled. Always:

1. Enter `CONFIRMING`
2. Play the readback, assembled by concatenating pre-recorded WAV fragments (see Section 9.3)
3. Wait for a yes/no
4. On yes → `DONE`. On no → return to `LISTENING` and let corrections flow

Any slot marked `"critical": true` must additionally be read back **digit by digit** (phone numbers) or **spelled out** (addresses) during the readback.

---

## 7. Domain Config Format

### 7.1 Principle

The engine knows nothing about pizza. Everything domain-specific lives in one JSON file. Adding a new domain must require **zero changes to `engine/`**.

### 7.2 JSON Schema

Write this to `configs/schema.json` and validate every config against it at load time.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["domain_id", "language", "slots", "endpointing", "overlap", "responses"],
  "properties": {
    "domain_id": { "type": "string" },
    "language": { "type": "string", "enum": ["ary"] },
    "needs_human_review": { "type": "boolean" },
    "slots": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["id", "type", "required", "critical", "prompt_audio"],
        "properties": {
          "id": { "type": "string" },
          "type": {
            "type": "string",
            "enum": ["enum", "enum_list", "integer", "text", "phone", "address", "date", "time"]
          },
          "required": { "type": "boolean" },
          "critical": { "type": "boolean" },
          "multi": { "type": "boolean", "default": false },
          "values": { "type": "array", "items": { "type": "string" } },
          "aliases": { "type": "object" },
          "min": { "type": "integer" },
          "max": { "type": "integer" },
          "prompt_audio": { "type": "string" },
          "prompt_text_ary": { "type": "string" },
          "readback_audio": { "type": "string" },
          "ask_order": { "type": "integer" }
        }
      }
    },
    "endpointing": {
      "type": "object",
      "required": ["base_silence_ms", "hesitation_hold_ms", "continuation_hold_ms",
                   "digit_hold_ms", "max_wait_ms", "min_speech_duration_ms", "vad_threshold"],
      "properties": {
        "base_silence_ms": { "type": "integer" },
        "hesitation_hold_ms": { "type": "integer" },
        "continuation_hold_ms": { "type": "integer" },
        "digit_hold_ms": { "type": "integer" },
        "max_wait_ms": { "type": "integer" },
        "min_speech_duration_ms": { "type": "integer" },
        "vad_threshold": { "type": "number" }
      }
    },
    "overlap": {
      "type": "object",
      "required": ["confidence_floor", "min_tokens_per_second", "rms_std_ceiling", "max_clarifies"],
      "properties": {
        "confidence_floor": { "type": "number" },
        "min_tokens_per_second": { "type": "number" },
        "rms_std_ceiling": { "type": "number" },
        "max_clarifies": { "type": "integer" }
      }
    },
    "responses": {
      "type": "object",
      "required": ["greeting", "clarify_overlap", "confirm_prefix", "confirm_question",
                   "accepted", "handoff", "not_understood"],
      "additionalProperties": {
        "type": "object",
        "required": ["audio", "text_ary", "needs_human_review"],
        "properties": {
          "audio": { "type": "string" },
          "text_ary": { "type": "string" },
          "needs_human_review": { "type": "boolean" }
        }
      }
    }
  }
}
```

### 7.3 Full pizza config

Write this to `configs/pizza.json`.

**Every `text_ary` value below is a DRAFT written by a non-native speaker and must be replaced by the project owner before recording audio.** They are all marked `needs_human_review: true`.

```json
{
  "domain_id": "pizza",
  "language": "ary",
  "needs_human_review": true,

  "slots": [
    {
      "id": "size",
      "type": "enum",
      "required": true,
      "critical": true,
      "multi": false,
      "ask_order": 1,
      "values": ["small", "medium", "large"],
      "aliases": {
        "small": ["[[TODO_DARIJA: small]]", "petite", "petit", "small"],
        "medium": ["[[TODO_DARIJA: medium]]", "moyenne", "moyen", "medium"],
        "large": ["[[TODO_DARIJA: large]]", "grande", "grand", "large"]
      },
      "prompt_audio": "ask_size.wav",
      "prompt_text_ary": "[[TODO_DARIJA: what size would you like?]]",
      "readback_audio": "rb_size.wav"
    },
    {
      "id": "toppings",
      "type": "enum_list",
      "required": false,
      "critical": false,
      "multi": true,
      "ask_order": 2,
      "values": ["cheese", "mushroom", "olive", "pepper", "onion", "tuna", "chicken", "beef"],
      "aliases": {
        "cheese": ["fromage", "[[TODO_DARIJA: cheese]]"],
        "mushroom": ["champignon", "[[TODO_DARIJA: mushroom]]"],
        "olive": ["olives", "[[TODO_DARIJA: olive]]"],
        "pepper": ["poivron", "[[TODO_DARIJA: pepper]]"],
        "onion": ["oignon", "[[TODO_DARIJA: onion]]"],
        "tuna": ["thon", "[[TODO_DARIJA: tuna]]"],
        "chicken": ["poulet", "[[TODO_DARIJA: chicken]]"],
        "beef": ["viande", "[[TODO_DARIJA: beef]]"]
      },
      "prompt_audio": "ask_toppings.wav",
      "prompt_text_ary": "[[TODO_DARIJA: what would you like on it?]]",
      "readback_audio": "rb_toppings.wav"
    },
    {
      "id": "quantity",
      "type": "integer",
      "required": true,
      "critical": true,
      "min": 1,
      "max": 20,
      "ask_order": 3,
      "prompt_audio": "ask_quantity.wav",
      "prompt_text_ary": "[[TODO_DARIJA: how many?]]",
      "readback_audio": "rb_quantity.wav"
    },
    {
      "id": "drink",
      "type": "enum_list",
      "required": false,
      "critical": false,
      "multi": true,
      "ask_order": 4,
      "values": ["cola", "fanta", "water", "none"],
      "aliases": {
        "cola": ["coca", "cola", "coca-cola"],
        "fanta": ["fanta"],
        "water": ["[[TODO_DARIJA: water]]", "eau"],
        "none": ["[[TODO_DARIJA: nothing]]", "rien"]
      },
      "prompt_audio": "ask_drink.wav",
      "prompt_text_ary": "[[TODO_DARIJA: any drinks?]]",
      "readback_audio": "rb_drink.wav"
    },
    {
      "id": "phone",
      "type": "phone",
      "required": true,
      "critical": true,
      "ask_order": 5,
      "prompt_audio": "ask_phone.wav",
      "prompt_text_ary": "[[TODO_DARIJA: what is your phone number?]]",
      "readback_audio": "rb_phone.wav"
    },
    {
      "id": "address",
      "type": "address",
      "required": true,
      "critical": true,
      "ask_order": 6,
      "prompt_audio": "ask_address.wav",
      "prompt_text_ary": "[[TODO_DARIJA: where should we deliver?]]",
      "readback_audio": "rb_address.wav"
    }
  ],

  "endpointing": {
    "base_silence_ms": 900,
    "hesitation_hold_ms": 600,
    "continuation_hold_ms": 900,
    "digit_hold_ms": 700,
    "max_wait_ms": 2500,
    "min_speech_duration_ms": 250,
    "vad_threshold": 0.5
  },

  "overlap": {
    "confidence_floor": 0.45,
    "min_tokens_per_second": 1.2,
    "rms_std_ceiling": 0.18,
    "max_clarifies": 3
  },

  "responses": {
    "greeting":         { "audio": "greeting.wav",   "text_ary": "[[TODO_DARIJA: hello, what would you like to order?]]", "needs_human_review": true },
    "clarify_overlap":  { "audio": "clarify.wav",    "text_ary": "[[TODO_DARIJA: sorry, one at a time please]]",          "needs_human_review": true },
    "confirm_prefix":   { "audio": "confirm_pre.wav","text_ary": "[[TODO_DARIJA: so your order is]]",                     "needs_human_review": true },
    "confirm_question": { "audio": "confirm_q.wav",  "text_ary": "[[TODO_DARIJA: is that correct?]]",                     "needs_human_review": true },
    "accepted":         { "audio": "accepted.wav",   "text_ary": "[[TODO_DARIJA: perfect, your order is registered]]",    "needs_human_review": true },
    "handoff":          { "audio": "handoff.wav",    "text_ary": "[[TODO_DARIJA: I'll pass you to a colleague]]",         "needs_human_review": true },
    "not_understood":   { "audio": "notund.wav",     "text_ary": "[[TODO_DARIJA: sorry, could you repeat?]]",             "needs_human_review": true }
  }
}
```

### 7.4 Clinic config

`configs/clinic.json` follows the identical schema. Slots:

| Slot | Type | Required | Critical |
|---|---|---|---|
| `doctor` | enum | yes | yes |
| `reason` | enum | no | no |
| `date` | date | yes | yes |
| `time` | time | yes | yes |
| `patient_name` | text | yes | yes |
| `phone` | phone | yes | yes |

This domain exists specifically because dates and times behave differently from item lists. If the engine needs code changes to support it, the abstraction in Section 7.2 is wrong and must be fixed rather than special-cased.

---

## 8. Lexicon

`engine/lexicon.py`. All lists are drafts requiring owner verification. Include both Arabic script and Latin (Arabizi) forms — Moroccans write both, and ASR output may use either.

```python
# EVERY LIST BELOW NEEDS HUMAN REVIEW BY A NATIVE DARIJA SPEAKER.
# The coding agent must not add, remove, or "correct" entries.

NEEDS_HUMAN_REVIEW = True

# Speaker is hesitating — they have NOT finished. Extend the wait.
HESITATION_MARKERS = [
    "اااه", "eeeh", "eh", "ahh",
    "زعما", "zaama", "z3ma",
    "يعني", "y3ni", "yaani",
    "mmm", "emm", "hmm",
    "euh", "ben", "alors",
    "[[TODO_DARIJA: add more hesitation fillers]]",
]

# Sentence ends on a connector — more is coming. Extend the wait.
CONTINUATION_MARKERS = [
    "و", "ou", "w", "wa",
    "زيد", "zid",
    "مع", "m3a", "m3ak",
    "ديال", "dyal", "d",
    "et", "avec", "plus",
    "[[TODO_DARIJA: add more trailing connectors]]",
]

# Hints passed to the router LLM so it recognises overwrites.
CORRECTION_MARKERS = [
    "لا لا", "la la", "lala",
    "ماشي", "machi",
    "بدل", "bdel",
    "سمح ليا", "sme7 liya", "smehli",
    "non", "pardon",
    "[[TODO_DARIJA: add more correction phrases]]",
]

AFFIRM_MARKERS = [
    "ايه", "iyeh", "ah", "wah",
    "واخا", "wakha",
    "صافي", "safi",
    "oui", "ok", "d'accord",
    "[[TODO_DARIJA: add more yes words]]",
]

NEGATE_MARKERS = [
    "لا", "la",
    "ماشي", "machi",
    "non",
    "[[TODO_DARIJA: add more no words]]",
]
```

---

## 9. Audio Output

### 9.1 Recording the voice bank

`scripts/record_audio_bank.py` reads a domain config, lists every `prompt_audio`, `readback_audio`, and `responses[*].audio` filename, prints the corresponding `text_ary`, and records from the default microphone into `audio/<domain>/<filename>`.

Requirements for all recordings: **16 kHz, mono, 16-bit PCM WAV.**

The script must refuse to run if any string in the config still contains `[[TODO_DARIJA` or if `needs_human_review` is `true`.

### 9.2 When TTS is permitted

TTS is permitted **only** for slot values that cannot be pre-recorded exhaustively: free-text addresses and patient names during readback.

For everything else — every prompt, every response, every enum value, every digit 0-9, every number 0-20 — pre-record it.

If TTS is used, log it in the session record so the benchmark can report what fraction of output was synthetic.

### 9.3 Assembling the readback

The readback is a **concatenation of pre-recorded fragments**, never a synthesized sentence.

Record these fragments per domain:
- One WAV per digit 0-9 (for phone numbers, read one digit at a time)
- One WAV per integer 1-20 (for quantities)
- One WAV per enum value across all slots
- One connector WAV (equivalent of "and")
- The `readback_audio` carrier phrase per slot

Concatenate with 120 ms of silence between fragments. This sounds slightly robotic and that is acceptable and honest. Do not attempt prosody smoothing.

---

## 10. The Router LLM

### 10.1 Contract

The router is called once per usable segment. It receives the slot definitions, the current state, and the normalized transcript. It returns JSON only.

### 10.2 Exact system prompt

Store this in `engine/router.py` as a constant. Do not modify it at runtime except by substituting the marked placeholders.

```
You are a slot-filling router for a voice assistant that operates in Moroccan
Darija. You do not talk to the user. You only output JSON.

TASK
Given the current task state and a new utterance, output the list of operations
that should be applied to the state.

SLOT DEFINITIONS
{slot_definitions_json}

CURRENT STATE
{current_state_json}

NEW UTTERANCE (already number-normalized)
{transcript}

DARIJA HINTS
Correction phrases (the speaker is overwriting an earlier value):
{correction_markers}
Agreement phrases: {affirm_markers}
Refusal phrases: {negate_markers}

RULES
1. Output ONLY a JSON object. No prose, no markdown, no code fences.
2. Never invent a value that is not in a slot's "values" list for enum slots.
3. If the utterance corrects an earlier value, emit a "set" operation. It is
   correct and expected to overwrite a filled slot.
4. If you cannot map the utterance to any slot, output an empty ops list and
   set "unclear" to true.
5. Multiple speakers contribute to one shared state. Do not track who spoke.
6. Do not ask questions. Do not generate any Darija text.

OUTPUT SCHEMA
{
  "ops": [
    { "op": "set" | "add" | "remove" | "clear",
      "slot": "<slot id>",
      "value": <string | integer | array | null> }
  ],
  "confidence": <float 0.0 to 1.0>,
  "unclear": <boolean>,
  "is_affirmation": <boolean>,
  "is_negation": <boolean>
}
```

### 10.3 Validation

Parse the response with `pydantic`. If parsing fails, or an `op` references an unknown slot, or an enum value is not in `values`, **discard the entire response** and treat the segment as `not_understood`. Retry once, then give up. Never partially apply an invalid response.

---

## 11. The Benchmark — "The Family Order Test"

### 11.1 Metrics (exact definitions)

Implement these in `bench/metrics.py`. Report all five. Reporting only the first four is misleading.

| # | Metric | Definition |
|---|---|---|
| 1 | **Task Success Rate (TSR)** | Percentage of scenarios where the final state exactly equals the gold state after canonical normalization. Binary per scenario. |
| 2 | **False Cutoff Rate (FCR)** | Number of endpoints emitted before a gold-annotated utterance boundary, divided by total gold utterances. Lower is better. |
| 3 | **Correction Success Rate (CSR)** | Of scenarios containing at least one annotated correction event, the percentage where the final state reflects the corrected value. |
| 4 | **Critical Entity Accuracy (CEA)** | Per slot with `critical: true`, exact-match percentage. Report per-slot and as a mean. |
| 5 | **Median Endpoint Latency (MEL)** | Median milliseconds from the gold utterance end timestamp to the emitted endpoint. |

**Metric 5 is mandatory.** FCR can be driven to zero by waiting forever, which produces an unusable agent. The headline result of this project is the **FCR-vs-MEL tradeoff curve**, not a single number.

### 11.2 Scenario file format

`bench/scenarios/pizza/s01.json`:

```json
{
  "id": "s01",
  "domain": "pizza",
  "title": "Single speaker, clean, complete order",
  "difficulty": "easy",
  "audio": "recordings/pizza/s01.wav",
  "speakers": 1,
  "background_noise": "none",
  "utterances": [
    { "start_ms": 0, "end_ms": 3400, "speaker": "A",
      "transcript_ary": "[[TODO_DARIJA]]", "is_correction": false }
  ],
  "gold_state": {
    "size": "large",
    "toppings": ["cheese"],
    "quantity": 1,
    "drink": ["cola"],
    "phone": "0612345678",
    "address": "[[TODO: real address]]"
  },
  "corrections": []
}
```

`utterances[].end_ms` is the **gold utterance boundary** used to compute FCR and MEL. The project owner must annotate these by listening. This is the most tedious part of the project and it cannot be skipped or automated.

### 11.3 The 20 pizza scenarios

Create scenario stubs for all twenty. The owner records the audio and fills the gold states.

**Easy (1 speaker, quiet, no corrections)**

| ID | Title |
|---|---|
| s01 | Complete order in one breath |
| s02 | Complete order, slot by slot, answering each question |
| s03 | Order with a long pause in the middle of naming toppings |
| s04 | Order where the phone number is said as a continuous digit string |
| s05 | Order where the phone number is said in grouped pairs |

**Medium (2 speakers, mild noise, corrections and code-switching)**

| ID | Title |
|---|---|
| s06 | Two speakers alternating; one gives size, other gives toppings |
| s07 | Speaker corrects the size mid-order ("la la, machi large") |
| s08 | Speaker corrects the quantity after the readback begins |
| s09 | Toppings named in French, everything else in Darija |
| s10 | Price/quantity said as a French numeral inside a Darija sentence |
| s11 | Speaker adds a topping after saying they were finished |
| s12 | TV audible in background throughout |

**Hard (3 speakers, real kitchen noise, layered corrections)**

| ID | Title |
|---|---|
| s13 | Three speakers each contributing different slots |
| s14 | Two people disagree on size; last statement wins |
| s15 | Long hesitation before naming a topping ("eeeh... zaama...") |
| s16 | Sentence ends on a connector, then continues after 1.2 s |
| s17 | Address given with landmarks, not a street number |
| s18 | Speaker changes their mind twice on the same slot |

**Chaos (overlap present)**

| ID | Title |
|---|---|
| s19 | Two speakers overlap once; engine must clarify and recover |
| s20 | Everyone talks at once for 3 s; engine must clarify, then complete |

For s19 and s20, the gold state is still a **complete correct order.** The measure is whether the engine recovers, not whether it parsed the overlap.

### 11.4 The endpointing sweep

`bench/sweep_endpointing.py` runs the full scenario set across a grid and writes a CSV:

- `base_silence_ms` ∈ {300, 500, 700, 900, 1100, 1300, 1500}
- holds enabled ∈ {all off, hesitation only, continuation only, all on}

That is 28 runs. Output columns: `base_silence_ms, holds, TSR, FCR, CSR, CEA, MEL`.

The deliverable chart is **FCR on the y-axis against MEL on the x-axis**, one line per hold configuration. The claim the project makes is that the Darija hold rules move the curve down and to the left compared to plain silence thresholding.

### 11.5 Baselines to compare against

Run the same scenarios through:

1. MoulSot v0.3 + plain 500 ms silence threshold (the "English default" baseline)
2. MoulSot v0.3 + tuned `base_silence_ms`, no hold rules
3. MoulSot v0.3 + tuned + all hold rules (the full system)
4. Groq `whisper-large-v3-turbo` + all hold rules (ASR ablation)

Report all four in the results table.

---

## 12. Testing

`pytest`. Only these four test files. Unit tests only, no integration tests against live APIs.

| File | Must cover |
|---|---|
| `test_normalize.py` | Darija numerals, French numerals inside Darija sentences, Moroccan phone formats (06/07 + 8 digits, +212 form), dirham amounts |
| `test_state.py` | Applying each op type, overwriting a filled slot, rejecting out-of-enum values, next-action selection, readiness for confirmation |
| `test_endpointing.py` | Each hold rule fires on synthetic marker tails; `max_wait_ms` cap is respected; `min_speech_duration_ms` discards blips |
| `test_router_schema.py` | Valid JSON parses; malformed JSON is discarded; unknown slot is discarded; out-of-enum value is discarded |

Mock all network calls. Tests must run offline in under 30 seconds.

---

## 13. Data Collection Protocol

The owner performs this. The coding agent does not.

1. **Consent.** Every person recorded signs or verbally records a consent statement covering research use and public dataset release. Keep the consent recordings.
2. **Realism is mandatory.** Record in a kitchen or living room, with normal background: TV, other conversation, dishes. Do not use a studio or a headset microphone. The mess is the scientific content.
3. **Diversity.** Target at least 8 distinct speakers across the 20 scenarios, mixed gender, mixed age. Include at least one speaker over 55 and one under 15 if possible.
4. **Device.** Record on a phone, held at normal conversational distance. Resample to 16 kHz mono.
5. **Annotation.** For every scenario, listen and mark `end_ms` for each utterance, plus flag correction events. Budget roughly 15 minutes of annotation per scenario.
6. **Release.** Publish the audio and annotations on Hugging Face under a permissive license once consent permits. If any speaker declines, exclude their scenario from the public release but keep it in the private results.

---

## 14. Milestones and Definition of Done

Work in this order. Do not skip ahead.

### M1 — Skeleton
Repo tree, `requirements.txt`, config loader with schema validation, `pizza.json` loading successfully, FastAPI serving `web/index.html`.
**Done when:** `python -m engine.server` starts and `scripts/check_config.py configs/pizza.json` passes.

### M2 — Audio in
AudioWorklet capturing 16 kHz mono, streaming 32 ms frames over WebSocket, server writing them to a WAV file.
**Done when:** speaking into the browser produces a playable WAV on disk.

### M3 — Endpointing
Silero VAD integrated, all four hold rules implemented, parameters read from config.
**Done when:** `test_endpointing.py` passes and the server logs a segment boundary at the expected time on a hand-made test clip.

### M4 — STT
MoulSot v0.3 wired in with Groq fallback and rate limiting.
**Done when:** speaking a sentence prints a transcript to the log.

### M5 — Normalize + Router + State
Number normalization, the LLM router with strict validation, state application, next-action selection.
**Done when:** `test_normalize.py`, `test_state.py`, `test_router_schema.py` all pass and a typed transcript produces the right state change.

### M6 — Voice out
Audio bank recorded by the owner, `responder.py` playing WAVs, barge-in working, readback concatenation working.
**Done when:** a full pizza order can be completed end to end by voice, including a correction.

### M7 — Overlap and clarify
`overlap.py` heuristics, `CLARIFYING` state, handoff after 3 clarifies.
**Done when:** deliberately talking over the bot triggers the clarify line and the session still completes.

### M8 — Benchmark
20 scenarios recorded and annotated, `run_bench.py`, `metrics.py`, results table.
**Done when:** `python bench/run_bench.py --domain pizza` prints all five metrics.

### M9 — Sweep
`sweep_endpointing.py`, 28 runs, CSV, the FCR-vs-MEL chart.
**Done when:** the chart exists and the tuned configuration is written back into `pizza.json`.

### M10 — Generality proof
`clinic.json` written, 10 clinic scenarios recorded, full flow working.
**Done when:** the clinic domain works **with zero changes to any file under `engine/`.** If any engine change was required, the abstraction failed and must be fixed before this milestone can close.

---

## 15. What Gets Published

1. **The repo**, with the engine, both configs, the benchmark, and the results.
2. **The dataset** on Hugging Face: audio plus annotations, consent permitting.
3. **The write-up**, structured as:
   - The problem: a real Moroccan family ordering, not one person in a studio
   - The baseline result: MoulSot with English-default endpointing, and its failure count
   - Every failure mode, listed and explained
   - The fixes: hold rules, shared state, readback, digit-by-digit confirmation
   - The FCR-vs-MEL tradeoff chart
   - The final result
   - The generality proof: same engine, clinic config, one config file changed
4. **The video**: real family, real kitchen noise, subtitled in Darija and English, including one failure case shown honestly.

**Framing rule:** the write-up positions this as *"I used MoulSot in a realistic setting, here is where it struggles, here is a fairer way to measure it, here is what I built to fix it."* It never claims to beat MoulSot. MoulSot's authors are cited and credited throughout.

---

## 16. Open Questions for the Owner

The coding agent appends to this list in `QUESTIONS.md`. The owner answers before the affected milestone starts.

1. Verify or replace every `[[TODO_DARIJA]]` placeholder in `configs/` and `engine/lexicon.py`.
2. Confirm the hesitation and continuation marker lists are complete for the target region (Casablanca / Rabat-Salé).
3. Choose the engine name.
4. Decide whether MoulSot runs locally or via a Hugging Face Space, which determines `MOULSOT_ENDPOINT`.
5. Confirm the 8-speaker recruitment target is achievable.
