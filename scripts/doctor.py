"""Check the installed runtime, each GPU, and the pinned source revisions."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import torch

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expected-gpus', type=int, default=None,
                        help='Optionally require this many visible GPUs; default accepts one or more.')
    args = parser.parse_args()
    report = {'python': platform.python_version(), 'cuda_runtime': torch.version.cuda,
              'packages': {n: importlib.metadata.version(n) for n in
                           ['torch', 'torchvision', 'transformers', 'mujoco', 'robosuite', 'h5py', 'libero']},
              'sources': {}, 'gpus': []}
    for name in ['le-worldmodel', 'LIBERO']:
        path = ROOT/'third_party'/name
        report['sources'][name] = {
            'commit': subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'], text=True).strip(),
            'diff': subprocess.check_output(['git','-C',str(path),'diff','--stat'], text=True).strip()}
    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError('No CUDA GPU is visible; simulation/training needs an NVIDIA GPU.')
    if args.expected_gpus is not None and count != args.expected_gpus:
        raise RuntimeError(f'Expected {args.expected_gpus} visible GPUs, found {count}')
    for gpu in range(torch.cuda.device_count()):
        with torch.cuda.device(gpu):
            x = torch.randn(512,512,device=f'cuda:{gpu}')
            with torch.autocast('cuda',dtype=torch.float16):
                result = x@x
            finite = bool(torch.isfinite(result).all())
            if not finite:
                raise RuntimeError(f'FP16 failed on GPU {gpu}')
            report['gpus'].append({'index':gpu,'name':torch.cuda.get_device_name(gpu),
                                   'capability':torch.cuda.get_device_capability(gpu),
                                   'memory_bytes':torch.cuda.get_device_properties(gpu).total_memory,
                                   'fp16_finite':finite})
    value = json.dumps(report,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(value)
    print(value)

if __name__ == '__main__':
    main()
