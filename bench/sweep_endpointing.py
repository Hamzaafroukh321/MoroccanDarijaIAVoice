"""28-run FCR/MEL sweep, four baselines, provenance and measured config tuning."""

import argparse
import asyncio
from copy import deepcopy
import csv
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from engine.config import ROOT, load_config
from engine.responder import AudioBank
from bench.run_bench import load_scenarios, run_suite


def variation(config, silence, holds):
    changed=deepcopy(config)
    changed['endpointing']['base_silence_ms']=silence
    for name,key in [('hesitation','hesitation_hold_ms'),('continuation','continuation_hold_ms'),('digit','digit_hold_ms')]:
        if holds!='all' and holds!=name: changed['endpointing'][key]=0
    return changed


def choose(rows, mode, max_mel):
    eligible=[row for row in rows if row['holds']==mode and row['MEL'] is not None and row['MEL']<=max_mel and row['boundary_coverage']==1]
    if not eligible: raise ValueError(f'No {mode} configuration satisfies latency and full boundary-coverage constraints; config was not tuned.')
    return min(eligible,key=lambda row:(row['FCR'],-row['TSR'],row['MEL']))


def chart(rows, settings):
    """Standalone SVG; no plotting dependency is added to the locked stack."""
    width=settings['plot_width'];height=settings['plot_height'];margin=settings['plot_margin']
    valid=[row for row in rows if row['MEL'] is not None and row['FCR'] is not None]
    if not valid: raise ValueError('No measured points to plot.')
    max_x=max(row['MEL'] for row in valid) or 1
    max_y=max(row['FCR'] for row in valid) or 1
    x=lambda value:margin+value/max_x*(width-2*margin)
    y=lambda value:height-margin-value/max_y*(height-2*margin)
    pieces=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title"><title id="title">False Cutoff Rate versus Median Endpoint Latency</title><rect width="100%" height="100%" fill="white"/><g font-family="Arial,sans-serif" font-size="14" fill="#173e36">',f'<text x="{margin}" y="32" font-size="22">The Family Order Test · Endpointing sweep</text>']
    for tick in range(6):
        fraction=tick/5
        pieces.append(f'<path d="M{x(fraction*max_x):.2f},{margin} V{height-margin} M{margin},{y(fraction*max_y):.2f} H{width-margin}" stroke="#e2e7e3" fill="none"/>')
        pieces.append(f'<text x="{x(fraction*max_x):.2f}" y="{height-margin+22}" text-anchor="middle">{fraction*max_x:.0f}</text><text x="{margin-12}" y="{y(fraction*max_y)+5:.2f}" text-anchor="end">{fraction*max_y:.2f}</text>')
    colors=['#1c674c','#b45b28','#4263aa','#8c4085']
    for mode,color in zip(settings['hold_modes'],colors):
        points=sorted((row for row in valid if row['holds']==mode),key=lambda row:row['base_silence_ms'])
        coordinates=' '.join(f"{x(row['MEL']):.2f},{y(row['FCR']):.2f}" for row in points)
        pieces.append(f'<polyline points="{coordinates}" stroke="{color}" fill="none" stroke-width="2"/>')
        for row in points:
            pieces.append(f'<circle cx="{x(row["MEL"]):.2f}" cy="{y(row["FCR"]):.2f}" r="4" fill="{color}"><title>{escape(mode)}; silence={row["base_silence_ms"]} ms; FCR={row["FCR"]:.3f}; MEL={row["MEL"]:.0f} ms</title></circle>')
        index=settings['hold_modes'].index(mode)
        pieces.append(f'<text x="{margin+index*(width-2*margin)/4}" y="{height-12}" fill="{color}">{escape(mode)}</text>')
    pieces.append(f'<text x="{width/2}" y="{height-38}" text-anchor="middle">Median endpoint latency (ms) →</text><text x="20" y="{height/2}" text-anchor="middle" transform="rotate(-90 20 {height/2})">False cutoffs / gold utterance →</text></g></svg>')
    return ''.join(pieces)


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain',choices=['pizza','clinic'],default='pizza')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();load_dotenv(ROOT/'.env')
    config_path=ROOT/'configs'/f'{args.domain}.json';config=load_config(config_path)
    scenarios,issues=load_scenarios(config)
    try: AudioBank(config,ROOT).preflight()
    except (RuntimeError,OSError) as exc: issues.append(str(exc))
    if issues:
        print('SWEEP NOT RUN — no measured chart or tuned config was written.')
        for issue in issues: print(' - '+issue)
        return 2
    if args.check: print('Ready for 28 configurations and the Groq ablation.');return 0
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    prefix=ROOT/'bench/results'/f'{args.domain}_sweep_{stamp}'
    rows=[];reports={}
    try:
        for silence in config['benchmark']['silence_grid_ms']:
            for mode in config['benchmark']['hold_modes']:
                report=await run_suite(variation(config,silence,mode),scenarios)
                key=f'{silence}_{mode}';reports[key]=report
                rows.append({'base_silence_ms':silence,'holds':mode,**report['metrics']})
                Path(str(prefix)+'_checkpoint.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False),encoding='utf-8')
        columns=['base_silence_ms','holds','TSR','FCR','CSR','CEA','MEL']
        with prefix.with_suffix('.csv').open('w',newline='',encoding='utf-8') as output:
            writer=csv.DictWriter(output,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        prefix.with_suffix('.svg').write_text(chart(rows,config['benchmark']),encoding='utf-8')
        best=choose(rows,'all',config['benchmark']['selection_max_mel_ms'])
        plain=choose(rows,'off',config['benchmark']['selection_max_mel_ms'])
        ablation=await run_suite(variation(config,best['base_silence_ms'],'all'),scenarios,primary='groq')
        baselines={
            'MoulSot / 500 ms / no holds':reports[f"{config['benchmark']['baseline_silence_ms']}_off"],
            'MoulSot / tuned / no holds':reports[f"{plain['base_silence_ms']}_off"],
            'MoulSot / tuned / all holds':reports[f"{best['base_silence_ms']}_all"],
            'Groq / tuned / all holds':ablation,
        }
        table=['# Measured baseline results','','Tuned and evaluated on the same scenario set; these are in-sample results.','','| System | TSR (%) | FCR | CSR (%) | CEA (%) | MEL (ms) |','|---|---:|---:|---:|---:|---:|']
        for name,report in baselines.items():
            values=[('N/A' if report['metrics'][key] is None else f"{report['metrics'][key]:.3f}") for key in columns[2:]]
            table.append('| '+name+' | '+' | '.join(values)+' |')
        Path(str(prefix)+'_baselines.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
        Path(str(prefix)+'_baselines.json').write_text(json.dumps(baselines,indent=2,ensure_ascii=False),encoding='utf-8')
        config['endpointing']['base_silence_ms']=best['base_silence_ms']
        config['benchmark']['tuned']=True;config['benchmark']['tuning_source']=prefix.with_suffix('.csv').relative_to(ROOT).as_posix()
        # Preserve the input config in each report before writing measured tuning.
        config_path.write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(f'Sweep, baselines and chart saved: {prefix}')
        return 0
    except (RuntimeError,ValueError,OSError) as exc:
        print(f'SWEEP INCOMPLETE: {exc}. Checkpoints preserved; no unmeasured tuning is applied.')
        return 2


if __name__=='__main__': raise SystemExit(asyncio.run(main()))
