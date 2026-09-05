"""Usage: python scripts/check_config.py configs/pizza.json"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import ConfigError, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Darija Voice domain config.")
    parser.add_argument("config", type=Path)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument('--ready',action='store_true',help='Also require reviewed language and a complete playable audio bank.')
    checks.add_argument('--demo',action='store_true',help='Validate the merged synthetic preview offline; makes no provider requests.')
    args = parser.parse_args()
    if args.demo:
        from dotenv import load_dotenv
        from engine.config import ROOT
        load_dotenv(ROOT / '.env')
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"VALID: {config['domain_id']} ({len(config['slots'])} slots)")
    if args.demo:
        from engine.demo import demo_config
        try:
            preview = demo_config(config)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            print(f'INVALID DEMO: {exc}', file=sys.stderr)
            return 2
        print(f"VALID DEMO: {preview['demo']['state_kind']}, {len(preview['slots'])} fields; questions and readback checked.")
        print('Offline configuration only: provider access, speech quality and native language review are not verified.')
    if args.ready:
        from engine.config import ROOT
        from engine.responder import AudioBank, AudioBankError
        try: AudioBank(config,ROOT).preflight()
        except (AudioBankError,OSError) as exc:
            print(f'NOT READY: {exc}',file=sys.stderr)
            return 2
        print('READY: language and WAV bank passed validation.')
    if config.get("needs_human_review") or "[[TODO_DARIJA" in str(config):
        print("Draft language: owner review is required before recording the voice bank.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
