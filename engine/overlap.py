"""Unusable-segment heuristics; no speaker identification or voice separation."""

import re
import numpy as np


def assess(pcm, transcript, confidence, speech_ms, config):
    settings=config['overlap']
    frame_samples=config['runtime']['sample_rate_hz']*config['runtime']['frame_ms']//1000
    audio=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768
    rms=[float(np.sqrt(np.mean(audio[index:index+frame_samples]**2))) for index in range(0,len(audio),frame_samples)]
    variance=float(np.std(rms)) if rms else 0.0
    tokens=len(re.findall(r'\w+',transcript))
    token_rate=tokens/(speech_ms/1000) if speech_ms>0 else 0.0
    signals={
        'low_confidence': confidence is not None and confidence<settings['confidence_floor'],
        'low_token_rate': token_rate<settings['min_tokens_per_second'],
        'high_energy_variance': variance>settings['rms_std_ceiling'],
    }
    return {'unusable':sum(signals.values())>=settings['required_signals'],'signals':signals,'confidence_available':confidence is not None,'token_rate':token_rate,'rms_std':variance}
