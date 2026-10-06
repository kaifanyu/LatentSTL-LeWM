"""Execute one GPU's frozen research jobs sequentially, preserving failures."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    args=parser.parse_args()
    spec=json.loads(args.manifest.read_text())
    status_path=args.manifest.with_suffix('.status.json')
    status={'status':'running','pid':os.getpid(),'gpu':spec['gpu'],'jobs':[]}
    process=None
    def save():
        status['updated_at']=datetime.now(timezone.utc).isoformat()
        tmp=status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(status_path)
    def stop(signum,frame):
        if process is not None and process.poll() is None:
            process.terminate()
        status['status']='interrupted';save()
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']=str(spec['gpu'])
    save()
    for job in spec['jobs']:
        row={'name':job['name'],'status':'running','started_at':datetime.now(timezone.utc).isoformat()}
        status['jobs'].append(row);save()
        with Path(job['log']).open('x') as log:
            process=subprocess.Popen(job['command'],cwd=spec['cwd'],env=environment,
                                     stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            row['pid']=process.pid;save()
            code=process.wait()
        row.update(status='completed' if code==0 else 'failed',exit_code=code,
                   finished_at=datetime.now(timezone.utc).isoformat());save()
    status['status']='completed' if all(j['status']=='completed' for j in status['jobs']) else 'completed_with_failures'
    save()


if __name__=='__main__':main()
