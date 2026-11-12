"""Check installed training interfaces before allocating a model."""
import importlib.metadata as metadata
import inspect
import json

def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from verl import DataProto
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU is required for training.')
    if not hasattr(DataParallelPPOActor,'update_policy'): raise RuntimeError('Incompatible verl actor API.')
    versions={}
    for n in ('torch','transformers','peft','verl','pyarrow','numpy','omegaconf','tensordict'):
        try: versions[n]=metadata.version(n)
        except metadata.PackageNotFoundError: versions[n]='source checkout; record commit separately'
    print(json.dumps({'packages':versions,'gpu':torch.cuda.get_device_name(0),
                     'grpo_signature':str(inspect.signature(compute_grpo_outcome_advantage)),
                     'note':'Import/API presence check only; run training contract tests before a GPU experiment.'},indent=2))

if __name__=='__main__': main()
