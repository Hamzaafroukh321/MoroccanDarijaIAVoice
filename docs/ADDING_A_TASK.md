# Add a voice task

A new task needs two local configuration files. It does not need an engine
registry edit, a new router implementation or a copied voice pipeline. Choose
`flat_scoped` for one set of fields, or `configured_collection_scoped` for one
collection of rows plus shared root fields. Pizza keeps its existing adapter.

## Configure a flat task

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

## Configure a collection task

Use the same base/profile file pair and discovery procedure above. Define the
row fields and shared root fields as ordinary base `slots`, then select them in
the demo profile's `fields`. Replace the flat profile's structural settings with
the following pattern. This English equipment example illustrates the contract;
it is not a complete configuration or a reviewed Darija task.

```json
{
  "state_kind": "configured_collection_scoped",
  "fields": ["asset", "quantity", "return_date"],
  "required_slots": ["asset", "quantity", "return_date"],
  "transaction_schema": {
    "collection": "reservations",
    "root_slots": ["return_date"],
    "item_slots": ["asset", "quantity"],
    "exclusive_values": {}
  },
  "collection_min_items": 1,
  "collection_max_items": 10,
  "collection_label": "Reservation"
}
```

The collection name must differ from every selected field and the reserved name
`id`. Row and root field IDs must also differ from `id`. `root_slots` and
`item_slots` must partition all selected fields exactly once, without overlap or
duplicates. Root fields may be empty; row fields may not, and at least one row
field must be required after overrides. `collection_min_items` is currently
exactly `1`; `collection_max_items` is an integer from `1` through `10`. Do not
set `order_schema_version`, which belongs to the legacy pizza adapter.

Supply a nonempty `collection_label` for the spoken and displayed row label.
Keep every selected field's question, label and readback entry, and add a
task-appropriate `responses.item_reference` line for asking which row the user
means. The shared responses, including `ambiguous_value`, `unsupported_value`
and `unintelligible`, are still required. Supply `readback_prefix`,
`readback_question`, and a `readback_order` containing every selected field once.
The renderer groups row fields under the collection label and reads shared root
fields separately. Optional `readback_formats` support `day_month_year` for date
fields and `24_hour` for time fields. `exclusive_values`, when used, applies only
to configured root `enum_list` fields. Keep draft language visibly unreviewed.

The resulting state has a named list of objects with stable positive `id`
values, alongside root fields. Row operations target those IDs; root operations
use `item_id: null`. IDs survive deletion and are not renumbered. Displayed and
spoken positions follow the remaining list: rows with IDs `2` and `3` are shown
as the first and second rows after row `1` is deleted. The router receives that
position-to-ID mapping. An `item_reference` clarification names explicit
candidate IDs instead of silently choosing a row.

Clear portions of a request can stay in a separate pending proposal while an
unsupported or ambiguous value is resolved. The proposed rows and root values
do not change committed state. A valid answer must address the current question
ID and its field/row scope; stale or invalid answers retain the pending details.
Committing a resolved draft still requires a fresh completed readback and a
separate confirmation. No reservation, payment or other external action occurs.

Run the same offline `scripts/check_config.py <base-config> --demo` validation
before opening the task. The temporary fixtures in
`tests/collection_fixtures.py` also show a fully populated base/profile pair and
a version with renamed collection and field IDs. They remain English engineering
fixtures; do not ship their diagnostic text as Darija speech.

### Link fields across rows and shared details

A configured collection can keep related choices across its rows and root fields
together with an explicit `linked_addresses` clarification. For example, if the alternatives
are one camera or three tripods, choosing tripod alone must not silently keep
the old camera quantity. The router can open this question:

```json
{
  "kind": "ambiguous_value",
  "slot": "asset",
  "item_ids": [1],
  "linked_addresses": [{"item_id": 1, "slot": "quantity"}]
}
```

This is a router response fragment, not an extra profile setting or a spoken
Darija example. The primary address comes from `slot` and `item_ids`: one stable
row ID for a row field, or an empty list for a root field. `linked_addresses`
lists only companions as exact `{item_id, slot}` objects. Root fields require
`item_id: null`; row fields require their own positive row ID. Each pair appears
once, excluding the primary pair. The same field on two rows is two addresses.
The group is limited to forty addresses and to the available configured roots
plus row fields across the configured maximum row count.

For two quantities and a shared date that depend on the chosen alternative:

```json
{
  "kind": "ambiguous_value",
  "slot": "quantity",
  "item_ids": [1],
  "linked_addresses": [
    {"item_id": 2, "slot": "quantity"},
    {"item_id": null, "slot": "return_date"}
  ]
}
```

Only `ambiguous_value` supports nonempty linkage. Participating rows must exist
in committed state or a validated initial proposal; inventing a future ID is
invalid. Resolve uncertain row identities separately. Root and row fields cannot
swap their null/positive ID conventions. Omitted, null or empty links preserve
ordinary single-field behavior; new strict-schema responses include the nullable
property. Choosing an alternative name does not automatically fill its package
of values: each participating field still needs an explicit answer.

Opening a group creates an uncommitted proposal even when there are no other
clear facts to stage. Independent facts may accompany the question, but initial
proposed operations cannot decide any linked address. Each named field requires
an explicit nonempty positive answer for this request, including optional fields
and fields with old saved values. A redundant explicit set counts as an answer;
an inherited value or bare yes/no does not. An answer may supply all linked
values together, or the engine asks the next field under a fresh question ID.

Partial answers preserve the original request, committed values and committed
ID allocator. The engine validates the combined draft and new operations on a
copy before changing pending state; the whole transaction is limited to forty
operations. Clearing/removing any linked answer or deleting any participating row
cannot resolve the group. Invalid or stale answers leave the draft intact.
Once every address is answered, the engine commits the transaction once and
requires a fresh complete readback followed by a separate affirmation.

The proposal's `linked_addresses` metadata contains the **full ordered group**,
including the primary address. `answered_addresses` and `remaining_addresses`
record engine-owned coverage. A continuation question's companion list excludes
its new primary address; use the proposal's remaining list to determine what
still needs an answer. Its raw preview may contain inherited old values, so
those values alone are not proof of coverage. Router-effective preview state
and the browser's pending preview hide only the unresolved fields at their
specific row IDs, or at the root for shared details. Other addresses and the
separately displayed committed state stay intact; the pending panel names each
row field or shared detail still needed.

Discarding the current question drops the entire linked draft, including staged
new rows and the original request. Older question IDs cannot discard or resume
it. Provider-error recovery continues to restore committed values only, without
pending linked answers or old confirmation authority. This contract enforces a
relationship the router declares; it does not prove that the router or ASR will
recognize every relationship in speech.

An exact answer to an active linked question can resolve locally: the complete
trimmed text must match one unique configured enum value/alias, or be an ASCII
integer within that field's bounds. This path checks the current question,
remaining address and committed version. It sets only that one field and leaves
group completion and confirmation to the same state engine. Ambiguous aliases,
confirmation words, punctuation, longer replies and unsupported types continue
through normal routing. It performs no fuzzy matching or companion inference.
Session evidence labels it `configured_exact_answer` and records local routing
separately from Groq calls. Exact text matching does not establish ASR accuracy.
The same lookup handles active flat `coupled_slots` questions using their current
root field, coverage and version. For clinic, an existing exact doctor alias can
advance to the linked time question while the choice stays pending. Date and time
answers continue through normal routing, including apparently simple numeric
answers; this lookup never guesses their interpretation.

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

For collections, also check adding two rows, correcting just one, deleting the
first row and referring to the remaining first row, asking which row was meant,
resolving a proposed new row, cancellation, stale answers and final confirmation.
Verify that a failed operation leaves the whole transaction unchanged and that
required root details are requested independently of required row details.
For linked alternatives, also check a partial answer, an optional linked
field with an old value, a proposal-created row, stale continuation IDs and
cancellation between answers. Confirm that the pending preview hides only the
unanswered address, then verify that the complete group commits once and needs a
new readback. Include two rows with the same linked field, a root-first question,
root/row ID partition errors and deletion of any participating row. None may
silently inherit an old value or clear the other address's obligation.

The temporary English service-counter and equipment fixtures establish local
field/schema/readback/UI behavior. The collection browser probe mocks its
configuration, microphone and transport; it checks stable IDs, visible row
positions, draft separation and desktop/mobile rendering with no provider calls.
These fixtures are not shipped voice demos, native language validation or ASR
accuracy results. Test supplied audio through actual MoulSot separately.

Configured collections currently support one collection only. Multiple or nested
collections and `coupled_slots` clarifications for collection tasks are unsupported.
Flat tasks use `coupled_slots`; configured collections use the explicit-address
`linked_addresses` contract above. Nested task
stacks, external bookings and arbitrary
slot types also remain outside this extension. Use the field types supported by
the base configuration schema and its runtime validation.

An optional [local MoulSot vocabulary experiment](../deploy/local-moulsot/README.md#optional-vocabulary-experiment)
uses `stt.moulsot_context` in the selected task's base configuration. It is off by
default, restricted to the dedicated local demo bridge, and never populated from
browser input or current answers. Keep vocabulary symmetric across the task's
options, validate its bounds, and retain the applied-context hash in evidence.
It has not established a native Darija accuracy improvement; reviewed research
sessions reject enabled nonempty context.
