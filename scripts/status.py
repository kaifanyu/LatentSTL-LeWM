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
        train_path=path.parent/'train/status.json'
        train=json.loads(train_path.read_text()) if train_path.exists() else {}
        summary={key:status.get(key) for key in
                 ['seed','physical_gpu','phase','supervisor_pid','child_pid']}
        summary.update(supervisor_alive=alive(status.get('supervisor_pid')),
                       child_alive=alive(status.get('child_pid')),
                       step=train.get('global_step'), updates=train.get('optimizer_updates'),
                       loss=train.get('loss',train.get('val_loss')),
                       elapsed_seconds=train.get('elapsed_seconds'))
        if 'mpc_gate' in status:
            summary['mpc_ready']=status['mpc_gate']['ready']
        print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
