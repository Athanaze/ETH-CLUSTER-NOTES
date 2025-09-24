# ETH-CLUSTER-NOTES

## CHECK ACCOUNT STATE

https://slurm-jobs-webgui.euler.hpc.ethz.ch/

## DESTROY EVERYTHING

```
scancel --user=$USER
```

## Create new instance

https://slurm-jobs-webgui.euler.hpc.ethz.ch/

-> 4x a100 don't work
-> 2x a100 did work

I requested 128GB RAM, but got 999GB

Be careful to download big models to  /cluster/scratch/saliechti

OTHERWISE ERRORS WHEN MORE THAN 50GB ON /home
