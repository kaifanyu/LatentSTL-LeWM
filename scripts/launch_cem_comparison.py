"""Launch the local and official-pretrained CEM jobs on separate GPUs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--local-checkpoint',type=Path,required=True)
    p.add_argument('--local-evidence',type=Path,required=True)
    p.add_argument('--pretrained-directory',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--gpus',nargs=2,type=int,default=[0,1])
    p.add_argument('--steps',type=int,default=120)
    p.add_argument('--population',type=int,default=128)
    p.add_argument('--iterations',type=int,default=4)
    p.add_argument('--horizon',type=int,default=8)
    p.add_argument('--banks',nargs='+',default=['test_0003','test_0000','test_0001','test_0002'])
    args=p.parse_args()
    for attr in ('dataset','local_checkpoint','local_evidence','pretrained_directory'):
        setattr(args,attr,getattr(args,attr).resolve(strict=True))
    if len(set(args.gpus)) != 2:
        raise ValueError('Each model needs its own GPU')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=(args.output or ROOT/'runs'/f'cem_comparison_{stamp}').resolve()
    output.mkdir(parents=True,exist_ok=False)
    with tarfile.open(output/'source.tar.gz','w:gz') as archive:
        for directory in ('latent_stl_lewm','scripts','tests','configs'):
            for path in sorted((ROOT/directory).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                    archive.add(path,arcname=path.relative_to(ROOT))
    workers=[]
    for kind,gpu in zip(('local','official_cube_transfer'),args.gpus):
        command=[sys.executable,'-m','latent_stl_lewm.exploratory_cem',
                 '--kind',kind,'--checkpoint',str(args.local_checkpoint if kind=='local' else args.pretrained_directory),
                 '--dataset',str(args.dataset),'--output',str(output/kind),
                 '--exploratory','--amp','--device','cuda','--render-gpu',str(gpu),
                 '--steps',str(args.steps),'--population',str(args.population),'--iterations',str(args.iterations),
                 '--horizon',str(args.horizon),'--banks',*args.banks]
        if kind=='local':
            command += ['--evidence',str(args.local_evidence),'--selection-note',
                        'Seed 2 best.pt chosen because it had the lowest previously measured 8-step prediction error among the three trained seeds; exploratory post-hoc selection.']
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',
                   PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4',LIBERO_CONFIG_PATH=str(ROOT/'.cache/libero'),
                   PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/'third_party/LIBERO'),env.get('PYTHONPATH','')]))
        with (output/f'{kind}.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append({'kind':kind,'gpu':gpu,'pid':process.pid,'command':command})
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),'output':str(output),
              'scope':'Two independent frozen-weight CEM simulator jobs; exact STL plus terminal goal-only control at identical starts.',
              'exploratory':True,'validation_gate_passed':False,'workers':workers}
    (output/'launch.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
