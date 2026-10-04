"""Show supervised seed jobs and their most recent training measurement."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path, nargs='?')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run
    if run is None:
        candidates = sorted((root/'runs').glob('campaign_*'))
        if not candidates:
            raise SystemExit('No campaigns have been launched.')
        run = candidates[-1]
    print(run.resolve())
    for path in sorted(run.glob('seed_*/status.json')):
        status = json.loads(path.read_text())
        status['supervisor_alive'] = alive(status.get('supervisor_pid'))
        status['child_alive'] = alive(status.get('child_pid'))
        print(json.dumps(status, sort_keys=True))
        for name in ['train/metrics.jsonl', 'train/train.jsonl', 'training.log']:
            log = path.parent/name
            if log.exists():
                # Seek from the tail, rather than reading a long training log.
                with log.open('rb') as handle:
                    handle.seek(max(0, log.stat().st_size-8192))
                    lines=handle.read().decode(errors='replace').splitlines()
                print('\n'.join(lines[-2:]))
                break


if __name__ == '__main__':
    main()
