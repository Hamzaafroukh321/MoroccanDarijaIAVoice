# Add a flat voice task

A new flat task needs two local configuration files. It does not need an engine
registry edit, a new router implementation or a copied voice pipeline. Pizza
continues to use its collection adapter; this extension supports flat tasks.

1. Copy `configs/clinic.json` to `configs/<task-id>.json`. Use the same safe,
   lowercase ID in `domain_id` and the filename, such as `service-desk`. Set
   `display_name`, the selected fields, values, required flags, question order and
   domain responses. Keep the existing runtime/provider settings unless the task
   needs a documented change. Remove inherited clinic-specific language and
   assumptions. Do not put credentials in these files.
2. Copy `configs/demo/clinic.json` to `configs/demo/<task-id>.json`. Keep
   `state_kind: "flat_scoped"`. Update the task scope, selected `fields`,
   `required_slots`, optional `slot_overrides`, questions, labels, value labels,
   response lines, readback order/formats and every `ui` string for the new task.
   Use `[[TODO_DARIJA: ...]]` placeholders until phrases are supplied/reviewed.
   Keep language-review flags and `evaluation_eligible: false` for a draft.
3. Make `transaction_schema.root_slots` match every selected field exactly once,
   with `collection: null` and `item_slots: []`. Do not add `order_schema_version`;
   that selects the pizza adapter. Overrides can change values/requirements but
   cannot rename field IDs. Readback must cover every selected field. Fields
   called `items` are ordinary flat fields, including enum lists.
4. Validate locally, from the project directory:

   ```powershell
   python scripts/check_config.py configs/service-desk.json --demo
   ```

   Substitute your filename. This makes no provider calls. Passing validates
   mappings and types; it does not establish pronunciation, native review or
   credential validity. The separate `--ready` check enforces the reviewed
   research WAV bank. A draft preview and a reviewed research task are distinct.
5. Restart the server to run model-availability checks for newly configured
   models. The validated task appears in the Domain selector. Choose the live
   demo or open `/?domain=service-desk&mode=demo`. Switching domains preserves
   session mode when that mode is available in the destination.

Shared voice settings still come from `configs/demo/voice.json` and the local
`.env`. The selected Darija XTTS voice and actual MoulSot ASR remain shared. A
custom profile must supply its own domain language and UI; it does not inherit
pizza fields or clinic booking claims.

Missing/malformed profiles produce setup issues before speaking. Invalid base
configs and filename/ID mismatches are excluded from discovery; requesting an
invalid domain directly fails. They do not prevent other valid domains starting.

## Verify the new task

Use the [multi-turn replay guide](../bench/VOICE_REPLAY.md) for supplied WAVs and
set the manifest's `domain` to the new task ID. The replay validator loads both
configuration files and rejects missing or invalid profiles before connecting;
no replay registry edit is needed. Manifest validation is offline by default,
and running its WAVs requires `--execute`. Keep diagnostics distinct from
native-reviewed expected outcomes. Check a
request, a correction, an unsupported option with held details, cancellation,
completed readback and a separate confirmation. An accepted state is only a
confirmed local task record; no external action is executed.

Include a linked-choice check when multiple fields describe one alternative:
choosing one field must not silently inherit an old value for another linked
field. Flat routing supports an explicit `coupled_slots` clarification. Partial
answers stay in one pending transaction; each next question gets a fresh ID and
the group commits only after its fields are answered. Unrelated saved values
remain intact. This depends on the router declaring the relationship; test the
domain's actual wording rather than assuming schema validation proves meaning.

The temporary English service-counter fixture proves field/schema/readback/UI
extensibility locally. It is not a shipped third voice demo or an ASR result.
Custom collections, nested task stacks, external bookings and arbitrary slot
types remain outside this extension. Use the field types supported by the base
configuration schema and its runtime validation.

An optional [local MoulSot vocabulary experiment](../deploy/local-moulsot/README.md#optional-vocabulary-experiment)
uses `stt.moulsot_context` in the selected task's base configuration. It is off by
default, restricted to the dedicated local demo bridge, and never populated from
browser input or current answers. Keep vocabulary symmetric across the task's
options, validate its bounds, and retain the applied-context hash in evidence.
It has not established a native Darija accuracy improvement; reviewed research
sessions reject enabled nonempty context.
