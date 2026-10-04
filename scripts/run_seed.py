"""Persistent per-GPU worker: train, evaluate, and run evidence-gated MPC."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--max-steps', type=int, default=20000)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = {'seed':args.seed, 'physical_gpu':args.gpu, 'supervisor_pid':os.getpid(),
              'dataset':str(args.dataset.resolve()), 'phase':'starting',
              'started_at':datetime.now(timezone.utc).isoformat()}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               LIBERO_CONFIG_PATH=str(ROOT/'.cache/libero'),
               MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
               MUJOCO_EGL_DEVICE_ID=str(args.gpu))
    process = None

    def save():
        status['updated_at']=datetime.now(timezone.utc).isoformat()
        temporary=args.output/'status.tmp'
        temporary.write_text(json.dumps(status,indent=2)+'\n')
        temporary.replace(args.output/'status.json')

    def stop(signum, frame):
        if process is not None and process.poll() is None:
            process.terminate()
        status['phase']='interrupted'
        save()
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    commands = {
        'training': [sys.executable,'-m','latent_stl_lewm.train', '--dataset',str(args.dataset),
                     '--output',str(args.output/'train'), '--seed',str(args.seed),
                     '--batch-size','32','--epochs',str(args.epochs),'--max-steps',str(args.max_steps),
                     '--num-workers','2','--cpu-threads','4','--amp','--log-every','20',
                     '--val-every','500','--val-batches','100000','--checkpoint-every','500'],
        'evaluation': [sys.executable,'-m','latent_stl_lewm.evaluate','--dataset',str(args.dataset),
                       '--checkpoint',str(args.output/'train'/'best.pt'), '--output',str(args.output/'evaluation'),
                       '--device','cuda','--seed',str(args.seed), '--horizons','1','2','4','8',
                       '--ranking-horizon','8','--max-trajectories','48'],
        'mpc': [sys.executable,'-m','latent_stl_lewm.mpc','--dataset',str(args.dataset),
                '--checkpoint',str(args.output/'train'/'best.pt'),
                '--evidence',str(args.output/'evaluation'/'report.json'),
                '--output',str(args.output/'mpc'),'--device','cuda','--render-gpu',str(args.gpu),
                '--seed',str(args.seed),'--scene-seed',str(900000+args.seed),
                '--objective','all','--horizon','8','--execute-horizon','1']}
    status['commands']=commands
    save()
    for phase, command in commands.items():
        status['phase']=phase
        with (args.output/f'{phase}.log').open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=process.pid
            save()
            code=process.wait()
        status[f'{phase}_exit_code']=code
        if code != 0:
            status['phase']='failed'
            save()
            raise SystemExit(code)
        if phase == 'training':
            training_status=json.loads((args.output/'train'/'status.json').read_text())
            if training_status.get('status') != 'completed':
                status['phase']=training_status.get('status','incomplete')
                save()
                raise SystemExit(1)
    report=json.loads((args.output/'evaluation'/'report.json').read_text())
    status['mpc_gate']=report['mpc_gate']
    status['mpc_result']=json.loads((args.output/'mpc'/'status.json').read_text())
    status['phase']='complete'
    status['child_pid']=None
    save()


if __name__=='__main__':
    main()
