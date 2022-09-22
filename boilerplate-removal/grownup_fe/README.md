# KTFNet benchmarking

## Extract content

```bash
CUDA_VISIBLE_DEVICES= ./extract_content.py --model-config-path /hpc-datasets/ben/ktf/runs/web/cleaneval/cleaneval_decode2.json --model-weights-path /hpc-datasets/ben/ktf/runs/web/cleaneval/cleaneval_decode2.h5 -i /hpc-datasets/web/cleaneval/ktfnet/all/ -o /hpc-datasets/ben/ktfnet/cleaneval/all/
```
