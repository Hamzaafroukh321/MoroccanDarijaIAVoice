# Remaining owner inputs and implementation decisions

The request to build M3 through M10 authorized continuing implementation past
pending manual milestone gates. The gates are still tracked as unverified; no
human recordings, reviewed translations or benchmark measurements are invented.

## Required for the real voice demo

September 5 update: Groq authentication and all three candidate router protocol
checks passed; 120B is the default, with strict JSON Schema and startup model
validation. The owner will supply the 30 Darija eval transcripts. Current next
steps are in [TESTING_NEXT.md](TESTING_NEXT.md). The duplicated MoulSot Space now
runs on ZeroGPU after its startup repair; the real kitchen recording has not been supplied.
Voice-bank and address TTS work are deferred until the MoulSot smoke is reviewed.

1. `GROQ_API_KEY` is now configured locally and authenticated successfully.
2. Use hosted MoulSot for the kitchen smoke, without Groq fallback. MoulSot's current Qwen3-ASR architecture requires
   a local package/runtime outside the locked dependency list. The repository
   supports its hosted Gradio API and a documented operator-hosted JSON endpoint;
   local loading is explicitly unavailable under the existing pins.
3. Review the original Darija marker lists in `engine/lexicon.py` and all config
   placeholders. Supply a native-speaker-reviewed numeral dictionary. That
   missing language fixture is the single skipped unit test.
4. Supply clinic doctor/reason choices and matching value-fragment recordings.
   The current choices are labeled placeholders, not invented clinicians.
5. Record the human WAV banks using `scripts/record_audio_bank.py`. Prompts,
   carrier phrases, enum values, numbers, digits and separators all need review.
6. Select/configure the restricted TTS endpoint for address spelling and patient
   names. Its JSON-to-WAV contract is documented in README.md. No new TTS service
   or dependency was silently selected.
7. Run a real microphone check and full corrected pizza/clinic voice sessions.
   Offline tests and mocked transport checks do not establish acoustic quality.

## Required for research acceptance

- Record and manually annotate 20 pizza and 10 clinic scenarios. Each JSON is an
  explicit stub with `annotations_reviewed: false`; no fabricated gold states.
- Confirm eight-speaker recruitment, consent and release permissions as described
  in SPEC.md. Public dataset upload has not been performed.
- Supply each recording's reference date so relative dates are reproducible.
- Run the benchmark and sweep after preflight passes. No measured CSV/chart,
  optimized config, baseline score or real generality result exists yet.
- Choose the final engine name. “Darija Voice” remains provisional.

## Resolved implementation conventions

- The base silence wait plus hesitation, continuation and digit holds implements
  the algorithm in Section 6.2; no unsupported fourth added hold was invented.
- Endpointing receives partial transcript tails through the pipeline. It performs
  no transcription itself. Partial calls are cached and bounded; slow providers
  affect actual endpoint latency and must be measured in the benchmark.
- An unfinished digit tail is a trailing digit sequence whose length is not a
  configured complete phone length. This is a documented heuristic to validate
  against owner recordings, not a proven linguistic rule.
- Section 6.5 takes precedence for handoff: clarify three times, hand off when the
  count exceeds three. Talking over the bot invokes barge-in; overlapping humans
  are assessed separately by the two-of-three unusable-audio rule.
- All five metrics are implemented. Missing ASR confidence stays unavailable;
  Groq's confidence proxy is labeled with its actual provenance.
- FCR/MEL matching, coverage, completion-aware TSR, replay timing and in-sample
  tuning selection are documented in README.md. These fill unspecified evaluation
  details and should accompany any published methodology.
- The audio bank includes zero in the number bank as well as digit zero. Dates
  are read in canonical ISO order with recorded separators; times use hour/minute
  digits. Address fallback must honor `spell: true`.
- Runtime, provider, transport, normalization, readback and benchmark settings live
  in the domain configs. Fixed PCM/RIFF layout constants and mathematical unit
  conversions are protocol definitions, not tunable thresholds.
- The existing project folder is the repository root. The source specification,
  exact dependency pins and `.env.example` are preserved. No database, telephony,
  speaker diarization, extra project dependency or external publication was added.

## Evidence currently available

- 66 offline tests pass; one real-Darija numeral fixture is explicitly skipped.
- Silero processes a PCM frame with the pinned runtime.
- Mocked provider WebSocket flow completes through greeting, state updates,
  acknowledged readback, affirmation and a saved completed session.
- A clinic fixture uses the same state/readback modules; no domain-specific
  implementation branch is required. Real clinic acceptance remains pending.
- Benchmark and sweep preflight refuse the incomplete dataset and do not write
  fake measurements or change tuning.
