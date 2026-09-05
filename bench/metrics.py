"""Task metrics with explicit missed-boundary accounting and undefined values."""

from statistics import mean, median
from engine.normalize import canonical_state


def endpoint_matches(utterances, endpoints):
    """Match to the first uncompleted gold utterance touched by the segment.

    A cutoff does not consume its gold utterance. A merged segment consumes at
    most one boundary, so missed boundaries remain visible in coverage/MEL.
    Unmatched endpoints are reported separately, never silently discarded.
    """
    matched=set(); false=0; extra=0; latencies=[]
    for endpoint in sorted(endpoints,key=lambda item:item['endpoint_ms']):
        start=endpoint['start_ms']; end=endpoint['endpoint_ms']
        candidates=[index for index,item in enumerate(utterances) if index not in matched and item['end_ms']>start and item['start_ms']<=end]
        if not candidates: extra+=1; continue
        index=candidates[0]; gold=utterances[index]
        if end<gold['end_ms']: false+=1
        else:
            matched.add(index); latencies.append(end-gold['end_ms'])
    return {'false_cutoffs':false,'latencies_ms':latencies,'matched_boundaries':len(matched),'missed_boundaries':len(utterances)-len(matched),'unmatched_endpoints':extra}


def compute(scenarios, results, config):
    if len(scenarios)!=len(results): raise ValueError('Every scenario must have one result.')
    critical=[slot['id'] for slot in config['slots'] if slot['critical']]
    correct={key:[] for key in critical}; successes=[]; corrections=[]; latency=[]
    false=0; gold_count=0; missed=0; extra=0; total_audio=0; synthetic=0
    for scenario,result in zip(scenarios,results):
        if result['scenario_id']!=scenario['id']: raise ValueError('Scenario/result id mismatch.')
        gold=canonical_state(scenario['gold_state'],config)
        actual=canonical_state(result['state'],config)
        successes.append(actual==gold and result['status']=='completed' and result.get('confirmed') is True)
        for key in critical:
            if key in gold: correct[key].append(actual.get(key)==gold[key])
        if scenario['corrections']:
            latest={event['slot']:event['value'] for event in scenario['corrections']}
            expected=canonical_state(latest,config)
            corrections.append(all(actual.get(key)==value for key,value in expected.items()))
        timing=endpoint_matches(scenario['utterances'],result['endpoints'])
        false+=timing['false_cutoffs']; missed+=timing['missed_boundaries']; extra+=timing['unmatched_endpoints']
        latency.extend(timing['latencies_ms']); gold_count+=len(scenario['utterances'])
        for output in result.get('outputs',[]):
            total_audio+=output['samples']; synthetic+=output['synthetic_samples']
    per_slot={key:100*mean(values) if values else None for key,values in correct.items()}
    available=[value for value in per_slot.values() if value is not None]
    return {
        'TSR':100*mean(successes) if successes else None,
        'FCR':false/gold_count if gold_count else None,
        'CSR':100*mean(corrections) if corrections else None,
        'CEA':mean(available) if available else None,
        'MEL':median(latency) if latency else None,
        'CEA_per_slot':per_slot,'scenario_count':len(scenarios),
        'correction_scenarios':len(corrections),'gold_utterances':gold_count,
        'false_cutoffs':false,'missed_boundaries':missed,'unmatched_endpoints':extra,
        'boundary_coverage':(gold_count-missed)/gold_count if gold_count else None,
        'synthetic_output_fraction':synthetic/total_audio if total_audio else 0.0,
    }
