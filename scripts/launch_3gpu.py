"""Launch three independent seeds and retain logs after the terminal exits."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import h5py

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-steps',type=int,default=20000)
    parser.add_argument('--epochs',type=int,default=100)
    args=parser.parse_args()
    args.dataset=args.dataset.resolve(strict=True)
    with h5py.File(args.dataset,'r') as data:
        if not data.attrs.get('complete',False):
            raise SystemExit('Dataset collection is not marked complete.')
        splits={str(episode.attrs['split']) for episode in data['trajectories'].values()}
        if not {'train','val','test'} <= splits:
            raise SystemExit('Dataset needs train/val/test episode splits.')
    if args.max_steps < 1 or args.epochs < 1:
        raise SystemExit('Training budgets must be positive.')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=(args.output or ROOT/'runs'/f'campaign_{stamp}').resolve()
    output.mkdir(parents=True,exist_ok=False)
    # Preserve all experiment code used at launch, including uncommitted files.
    with tarfile.open(output/'source.tar.gz','w:gz') as archive:
        for directory in ['latent_stl_lewm','scripts','tests','configs','docs']:
            for path in sorted((ROOT/directory).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                    archive.add(path,arcname=path.relative_to(ROOT))
        for name in ['pyproject.toml','requirements.lock.txt','README.md']:
            path=ROOT/name
            if path.exists():
                archive.add(path,arcname=name)
    with args.dataset.open('rb') as handle:
        dataset_hash=hashlib.file_digest(handle,'sha256').hexdigest()
    metadata={'created_at':datetime.now(timezone.utc).isoformat(),'dataset':str(args.dataset),
              'dataset_sha256':dataset_hash,'batch_size':32,'precision':'fp16_amp',
              'ddp':False,'max_steps':args.max_steps,'epochs':args.epochs,'workers':[]}
    for seed,gpu in enumerate([0,1,2]):
        command=[sys.executable,str(ROOT/'scripts/run_seed.py'),'--dataset',str(args.dataset),
                 '--output',str(output/f'seed_{seed}'),'--gpu',str(gpu),'--seed',str(seed),
                 '--max-steps',str(args.max_steps),'--epochs',str(args.epochs)]
        with (output/f'supervisor_seed_{seed}.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        metadata['workers'].append({'seed':seed,'gpu':gpu,'pid':process.pid,'command':command})
    (output/'launch.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'output':str(output),'workers':metadata['workers']},indent=2))


if __name__=='__main__':
    main()
