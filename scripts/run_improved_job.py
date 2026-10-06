"""Detached worker which optionally waits for predictor training then runs CEM."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import signal
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('job',type=Path)
    args=p.parse_args()
    job=json.loads(args.job.read_text())
    status_path=args.job.with_suffix('.status.json')
    status={'status':'starting','supervisor_pid':os.getpid(),'variant':job['variant'],'gpu':job['gpu']}
    process=None
    def save(**values):
        status.update(values,updated_at=datetime.now(timezone.utc).isoformat())
        temporary=status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(status_path)
    def stop(signum,frame):
        if process is not None and process.poll() is None:process.terminate()
        save(status='interrupted')
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    prerequisite=job.get('wait_for_training')
    if prerequisite:
        deadline=time.monotonic()+7200
        while True:
            try:training=json.loads(Path(prerequisite).read_text())
            except (FileNotFoundError,json.JSONDecodeError):training={}
            save(status='waiting_for_training',training_step=training.get('step'))
            if training.get('status') in ('complete','completed'):break
            if training.get('status') in ('failed','stopped','interrupted') or time.monotonic()>deadline:
                save(status='failed',error='Prerequisite training did not complete',training_status=training)
                raise SystemExit(1)
            time.sleep(5)
    save(status='running')
    with Path(job['log']).open('x') as log:
        process=subprocess.Popen(job['command'],cwd=job['cwd'],stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT)
        save(child_pid=process.pid)
        code=process.wait()
    save(status='completed' if code==0 else 'failed',exit_code=code)
    raise SystemExit(code)


if __name__=='__main__':main()
