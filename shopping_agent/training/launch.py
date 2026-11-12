"""Portable, explicit launch wrapper; dry-run never loads a model."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

def build_command(config):
    command = [sys.executable, '-u', '-m', 'shopping_agent.training.grpo']
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                command.append('--'+key)
        else:
            command += ['--'+key, str(value)]
    return command

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/grpo_136x34_v12.json')
    p.add_argument('--model'); p.add_argument('--output')
    p.add_argument('--rollout-batch-size',type=int); p.add_argument('--ppo-micro-batch-size',type=int)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    for key in ('model','output','rollout_batch_size','ppo_micro_batch_size'):
        if getattr(args,key) is not None: config[key.replace('_','-')]=getattr(args,key)
    cmd=build_command(config)
    if args.dry_run:
        print(shlex.join(cmd)); return
    os.chdir(ROOT)
    out=Path(config['output']).resolve()
    if out.exists(): raise FileExistsError('Use a new output directory; never overwrite a run.')
    out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(out.parent).free < 30*1024**3:
        raise RuntimeError('At least 30 GiB free space is required for a new run.')
    for f in [Path(config['catalog']), Path(config['tasks'])/'train.parquet',
              Path(config['tasks'])/'test.parquet']:
        if not f.is_file(): raise FileNotFoundError(f'Prepare and audit data first: {f}')
    lock=out.with_name(out.name+'.launch.lock')
    with lock.open('x',encoding='utf-8') as f: f.write(str(os.getpid()))
    try:
        subprocess.run([sys.executable,'-m', 'shopping_agent.training.environment'],check=True)
        subprocess.run(cmd,check=True)
    finally:
        lock.unlink()

if __name__=='__main__': main()
