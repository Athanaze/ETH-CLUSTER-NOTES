
# ETH-CLUSTER-NOTES

Note : court decision neuchatel in .md + quite big system prompt is 14474 input tokens for bytedance model

Using the tokenizer for qwen3-235B, one token is on average 3.4 characters

run bytedance on single a100 80GB

```
vllm serve "ByteDance-Seed/Seed-OSS-36B-Instruct"   --max-model-len 20000  --swap-space 32 --gpu_memory_utilization=0.95
```

## CHECK ACCOUNT STATE

https://slurm-jobs-webgui.euler.hpc.ethz.ch/

## DESTROY EVERYTHING

```
scancel --user=$USER
```

## Create new instance

https://jupyter.euler.hpc.ethz.ch/hub/spawn

-> 4x a100 did not work
-> 2x a100 did work

I requested 128GB RAM, but got 999GB

Be careful: download big models to  /cluster/scratch/saliechti

OTHERWISE ERRORS WHEN MORE THAN 50GB ON /home
