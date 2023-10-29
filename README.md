# starcoder_ablation

```
python run.py \
    --checkpoint "models/random_8k_8k" \
    --device "cuda" \
    --dataset_name "tianyang/repobench_ablation_8k" \
    --cache_dir "./data" \
    --results_dir "results" \
    --prompt_version "v2" \
    --token_num_threshold 8063 \
    --resume 0 \
    --resume_part None \
    --max_new_tokens 128 \
    --temperature 0.2 \
    --top_p 0.95 \
```