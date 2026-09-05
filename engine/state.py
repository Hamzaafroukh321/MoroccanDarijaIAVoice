"""Atomic slot operations and confirmation of the exact state read back."""

from dataclasses import dataclass

from engine.normalize import slot_value
from engine.confirmation import VersionedConfirmation


@dataclass(frozen=True)
class Action:
    kind: str
    slot: str | None = None
    item_id: int | None = None


def validate_operation(operation, slots, config):
    op=operation['op']; key=operation['slot']; value=operation['value']
    if key not in slots: raise ValueError('Unknown slot.')
    slot=slots[key]
    if op not in {'set','add','remove','clear'}: raise ValueError('Unknown operation.')
    if op=='clear':
        if value is not None: raise ValueError('Clear requires a null value.')
        return
    if op=='remove' and value is None: return
    if op=='add' and slot['type']!='enum_list': raise ValueError('Add requires an enum_list slot.')
    if slot['type']=='enum_list' and op in {'add','remove'} and isinstance(value,str): value=[value]
    slot_value(value,slot,config)


class TaskState(VersionedConfirmation):
    def __init__(self, config):
        self.config=config
        self.slots={slot['id']:slot for slot in config['slots']}
        self.values={}
        self.version=0
        self.confirmed_version=None
        self.readback_version=None
        self.awaiting_correction=False

    @property
    def ready(self):
        return all(not slot['required'] or self.values.get(key) not in (None,'',[]) for key,slot in self.slots.items())

    def apply(self, operations):
        from engine.transactions import apply_transaction
        if any(set(operation)!={'op','slot','value'} for operation in operations):
            raise ValueError('A field operation requires op, slot and value only.')
        mapped=[{**operation,'collection':None,'item_id':None} for operation in operations]
        candidate,_=apply_transaction(self.values,{},mapped,root_slots=self.slots,
                                      collections={},config=self.config)
        if candidate!=self.values:
            self.values=candidate
            self.version+=1
            self.invalidate_confirmation()
        if operations: self.awaiting_correction=False

    def consume(self, response):
        """Corrections in the same utterance as yes always require a new readback."""
        self.apply(response['ops'])
        if response['is_negation']:
            self.invalidate_confirmation()
            self.awaiting_correction=not bool(response['ops'])
        elif response['unclear']:
            self.invalidate_confirmation()
        elif response['is_affirmation']:
            self.accept_confirmation(response['ops'])
        elif response['ops']:
            self.invalidate_confirmation()
        if response['unclear']: return Action('not_understood')
        return self.next_action()

    def next_action(self):
        if self.confirmed: return Action('accepted')
        if self.awaiting_correction: return Action('listen')
        missing=[slot for key,slot in self.slots.items() if slot['required'] and self.values.get(key) in (None,'',[])]
        if missing:
            slot=min(missing,key=lambda slot:slot['ask_order'])
            return Action('ask',slot['id'])
        return Action('readback')
