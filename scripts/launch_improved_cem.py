"""Launch three isolated CEM improvement variants with reproducible provenance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--steps',type=int,default=160)
    p.add_argument('--scene-seeds',nargs='+',type=int,default=[900100,900101,900102])
    args=p.parse_args()
    campaign=args.campaign.resolve(strict=True)
    if (campaign/'launch.json').exists():raise FileExistsError('Campaign already launched')
    references=(campaign/'references/references.pt').resolve(strict=True)
    prior=(campaign/'action_prior.pt').resolve(strict=True)
    checkpoint=args.checkpoint.resolve(strict=True)
    dataset=args.dataset.resolve(strict=True)
    variants=[('gpu0_reference_cem',0,checkpoint,8,False,.25,.02),
              ('gpu1_reference_prior_cem',1,checkpoint,16,True,.12,.01),
              ('gpu2_multistep_prior_cem',2,campaign/'multistep_training/best.pt',16,True,.12,.01)]
    with tarfile.open(campaign/'source.tar.gz','w:gz') as archive:
        for directory in ('latent_stl_lewm','scripts','tests','configs'):
            for path in sorted((ROOT/directory).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                    archive.add(path,arcname=path.relative_to(ROOT))
        archive.add(ROOT/'README.md',arcname='README.md')
    workers=[]
    for variant,gpu,weights,horizon,use_prior,std,min_std in variants:
        command=[sys.executable,'-m','latent_stl_lewm.improved_cem','--checkpoint',str(weights),
                 '--dataset',str(dataset),'--references',str(references),'--output',str(campaign/variant),
                 '--variant',variant,'--render-gpu',str(gpu),'--device','cuda','--amp',
                 '--horizon',str(horizon),'--execute-steps','2','--steps',str(args.steps),
                 '--population','128','--iterations','4','--elites','16','--chunk-size','32',
                 '--initial-std-scale',str(std),'--min-std',str(min_std),
                 '--scene-seeds',*[str(x) for x in args.scene_seeds]]
        if use_prior:command.extend(['--action-prior',str(prior)])
        job={'variant':variant,'gpu':gpu,'command':command,'cwd':str(ROOT),'log':str(campaign/f'{variant}.log')}
        if gpu==2:job['wait_for_training']=str(campaign/'multistep_training/status.json')
        job_file=campaign/f'{variant}.job.json'
        job_file.write_text(json.dumps(job,indent=2)+'\n')
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',
                   PYOPENGL_PLATFORM='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4',LIBERO_CONFIG_PATH=str(ROOT/'.cache/libero'),
                   PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/'third_party/LIBERO'),env.get('PYTHONPATH','')]))
        with (campaign/f'{variant}.supervisor.log').open('x') as log:
            process=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_improved_job.py'),str(job_file)],
                                     cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        workers.append({'variant':variant,'gpu':gpu,'supervisor_pid':process.pid,'job':str(job_file),'command':command})
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),'campaign':str(campaign),
              'scene_seeds':args.scene_seeds,'steps_per_trial':args.steps,'workers':workers,
              'objective':'Empirical safe reaching with calibrated visual predicates and actual CEM action sampling',
              'supervision':'Offline train/validation labels for visual predicate references/calibration; no test outcomes in fitting',
              'physical_success':'Every frame: mug displacement <=2mm and rotation<=2deg; at least one frame: eef goal error<=3cm',
              'notes':['All three use the same frozen visual encoder.',
                       'GPU1/2 use training demonstrations to initialize CEM, explicitly tagged in action provenance.',
                       'GPU2 waits for predictor-only multistep fine-tuning completion.',
                       'Exact full-history STL is reported; future safety and goal progress guide action selection.']}
    (campaign/'launch.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
