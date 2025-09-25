```
venv) [s@qr text_content]$ python text_content_analysis.py
Starting analysis of 'deduplicated_fixed.csv'...
Using 16 worker processes and a chunk size of 1000 rows.
Processing Chunks: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1854/1854 [00:00<00:00, 4470.77it/s]

--- Statistics on Character Lengths of Text Parts (>= 5 chars) ---
Total valid parts analyzed: 1,853,338
Average (Mean) Length:      14790.06
Median Length:              3042.00
Standard Deviation:         25864.61
95th Percentile:            54193.00 (95% of parts are shorter than this)
Average Length in Top 1%:   170261.30
Average Length in Top 0.1%: 385385.12

Generating plots...
Plots saved to 'length_analysis_plots.png'
```


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
