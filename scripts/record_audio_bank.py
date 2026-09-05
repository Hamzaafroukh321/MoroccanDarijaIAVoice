"""Record a reviewed voice bank through the existing browser microphone path."""

import argparse
from pathlib import Path
import sys
import time
from urllib.parse import urlencode
import webbrowser

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine.config import ROOT, load_config
from engine.responder import AudioBank, bank_manifest, review_issues


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--list',action='store_true',help='List reviewed phrases without opening the microphone page.')
    parser.add_argument('--overwrite',action='store_true',help='Explicitly replace existing recordings.')
    args=parser.parse_args();config=load_config(args.config)
    issues=review_issues(config)+review_issues(bank_manifest(config))
    if issues:
        print(f'REFUSED: {len(issues)} unreviewed fields/placeholders remain. Review the config first.')
        return 2
    bank=AudioBank(config,ROOT);manifest=bank_manifest(config)
    for filename,text in manifest.items(): print(f'{filename}: {text}')
    if args.list: return 0
    print('Start python -m engine.server in another terminal. The browser uses the default microphone.')
    for filename,text in manifest.items():
        target=bank.path(filename)
        if target.exists() and not args.overwrite:
            print('Keeping '+filename);continue
        before=target.stat().st_mtime_ns if target.exists() else None
        input(f'\n{filename}\n{text}\nPress Enter to open its recording page, or Ctrl+C to stop. ')
        query=urlencode({'domain':config['domain_id'],'bank':filename,'overwrite':str(args.overwrite).lower()})
        webbrowser.open(f"http://{config['runtime']['host']}:{config['runtime']['port']}/?{query}")
        print('Record, stop, and check playback in the browser. Waiting for the WAV…')
        while not target.exists() or target.stat().st_mtime_ns==before:
            time.sleep(config['audio_output']['recording_poll_ms']/1000)
        bank.decode(target.read_bytes())
        print('Saved '+str(target))
    print('Voice-bank recording complete.')
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except KeyboardInterrupt: print('\nRecording stopped. Existing WAV files are preserved.');raise SystemExit(130)
