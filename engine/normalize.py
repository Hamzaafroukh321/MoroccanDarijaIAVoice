"""Config-driven numeral, phone, currency, date and time normalization."""

from datetime import datetime, timedelta, date
import re
import unicodedata


def decimal_digits(text):
    return ''.join(str(unicodedata.decimal(char)) if char.isdecimal() else char for char in str(text))


def phone(value, settings):
    text = decimal_digits(value)
    if re.search(r'[^\d+\s().-]',text): raise ValueError('Phone numbers may only contain digits and separators.')
    compact = re.sub(r'[\s().-]','',text)
    prefix='+'+settings['phone_country_code']
    if compact.startswith(prefix): compact='0'+compact[len(prefix):]
    if not compact.isdecimal() or len(compact)!=settings['phone_length'] or not any(compact.startswith(start) for start in settings['phone_local_prefixes']):
        raise ValueError('Expected a Moroccan 06/07 mobile number or its +212 form.')
    return compact


def normalize(text, config):
    settings=config['normalization']
    result=unicodedata.normalize('NFC',decimal_digits(text))
    words={**settings['number_words'],**settings['darija_numbers']}
    words={key.casefold():str(value) for key,value in words.items() if '[[' not in key}
    scales=settings['number_scales']
    if scales:
        tokens={**words,**{key:str(value) for key,value in scales.items()}}
        atom='|'.join(re.escape(key) for key in sorted(tokens,key=len,reverse=True))
        def compound(match):
            parts=re.findall(atom,match[0],flags=re.IGNORECASE)
            if not any(part.casefold() in scales for part in parts): return match[0]
            total=0; current=0
            largest=max(scales.values())
            for part in parts:
                key=part.casefold(); number=int(tokens[key])
                if key in scales:
                    if number==largest: total+=(current or 1)*number;current=0
                    else: current=(current or 1)*number
                else: current+=number
            return str(total+current)
        result=re.sub(r'(?<!\w)(?:'+atom+r')(?:[ -]+(?:'+atom+r'))*(?!\w)',compound,result,flags=re.IGNORECASE)
    if words:
        pattern=r'(?<!\w)('+ '|'.join(re.escape(key) for key in sorted(words,key=len,reverse=True))+r')(?!\w)'
        result=re.sub(pattern,lambda match:words[match[0].casefold()],result,flags=re.IGNORECASE)
    def normalize_phone(match):
        try: return phone(match[0],settings)
        except ValueError: return match[0]
    result=re.sub(r'(?<![\w+])\+?\d(?:[\s().-]*\d)*(?!\w)',normalize_phone,result)
    for alias,canonical in settings['currency_aliases'].items():
        result=re.sub(r'(?<=\d)\s*'+re.escape(alias)+r'\b',' '+canonical,result,flags=re.IGNORECASE)
    for expression in sorted(settings['relative_dates'],key=len,reverse=True):
        if '[[' not in expression:
            result=re.sub(r'(?<!\w)'+re.escape(expression)+r'(?!\w)',temporal(expression,'date',settings),result,flags=re.IGNORECASE)
    return ' '.join(result.split())


def temporal(value, kind, settings, *, today=None):
    value=decimal_digits(value).strip()
    if kind=='date' and value.casefold() in settings['relative_dates']:
        reference=date.fromisoformat(settings['reference_date']) if settings.get('reference_date') else date.today()
        return ((today or reference)+timedelta(days=settings['relative_dates'][value.casefold()])).isoformat()
    for fmt in settings[f'{kind}_input_formats']:
        try:
            parsed=datetime.strptime(value,fmt)
            return parsed.date().isoformat() if kind=='date' else parsed.strftime('%H:%M')
        except ValueError: pass
    raise ValueError(f'Invalid {kind}: use an unambiguous configured format.')


def slot_value(value, slot, config):
    """Validate a single slot value; never coerce a bool into an integer."""
    kind=slot['type']
    if kind=='integer':
        if type(value) is not int: raise ValueError('Expected an integer.')
        if ('min' in slot and value<slot['min']) or ('max' in slot and value>slot['max']): raise ValueError('Integer is outside slot bounds.')
        return value
    if kind=='enum_list':
        if not isinstance(value,list) or any(type(item) is not str or item not in slot['values'] for item in value): raise ValueError('Invalid enum list.')
        return list(dict.fromkeys(value))
    if not isinstance(value,str) or not value.strip(): raise ValueError('Expected nonempty text.')
    value=' '.join(unicodedata.normalize('NFC',value).split())
    if '[[' in value: raise ValueError('Unresolved placeholders are not valid slot values.')
    if kind=='enum' and value not in slot['values']: raise ValueError('Invalid enum value.')
    if kind=='phone': return phone(value,config['normalization'])
    if kind in {'date','time'}: return temporal(value,kind,config['normalization'])
    return value


def canonical_state(state, config):
    slots={slot['id']:slot for slot in config['slots']}
    if any(key not in slots for key in state): raise ValueError('Unknown slot in state.')
    result={}
    for key,value in state.items():
        value=slot_value(value,slots[key],config)
        if isinstance(value,list): value=sorted(value)
        if slots[key]['type'] in {'text','address'}: value=value.casefold()
        result[key]=value
    return result
