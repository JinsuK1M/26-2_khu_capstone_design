"""Reproduce the fresh-process CPU/thread comparison without persistent changes."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

WORKER = r'''
import json, os
from time import perf_counter
import numpy as np
import torch
import fcm_edge
torch.set_num_threads(int(os.environ['BENCH_THREADS']))
x=torch.from_numpy(np.load('test_l37.npy',allow_pickle=False))
assert tuple(x.shape)==(1,256,80,80) and torch.isfinite(x).all()
encoder=fcm_edge.load_encoder()
with torch.inference_mode():
    for _ in range(3): encoder(x)
    times=[]
    for _ in range(10):
        start=perf_counter(); y=encoder(x)
        times.append((perf_counter()-start)*1000)
assert torch.isfinite(y).all()
print(json.dumps(dict(cpus=sorted(os.sched_getaffinity(0)),
    threads=torch.get_num_threads(),median_ms=float(np.median(times)),
    p95_ms=float(np.percentile(times,95)),raw_ms=times)))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    rows=[]
    for label,cpus,threads in [('all4','0-7',4),('big4','4-7',4),('big1','4-7',1),
                                ('big1_repeat','4-7',1),('big4_repeat','4-7',4),('all4_repeat','0-7',4)]:
        print('Benchmark:',label,flush=True)
        env=os.environ.copy()
        env.update(BENCH_THREADS=str(threads),OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads))
        result=subprocess.run(['taskset','-c',cpus,sys.executable,'-u','-c',WORKER],
            cwd=args.bundle,env=env,capture_output=True,text=True,check=True,timeout=180)
        row=json.loads(result.stdout.strip().splitlines()[-1]);row['case']=label
        rows.append(row);print(json.dumps(row),flush=True)
    target=args.bundle/'cpu_fenet_runs'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    target.mkdir(parents=True)
    (target/'report.json').write_text(json.dumps(rows,indent=2))
    print('Saved:',target)


if __name__=='__main__':
    main()
