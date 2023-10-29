import os
import time
import json
import random
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import fire

from utils import construct_prompt, get_first_line_not_comment

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def load_model(model_dir: str, branch: str, cache_dir: str, device: str):
    """Load the model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(model_dir, revision=branch, cache_dir=cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, revision=branch, cache_dir=cache_dir)
    return model, tokenizer


def generate_code(model, tokenizer, prompt, max_new_tokens, temperature, top_p, device):
    """Generate code using the model."""
    generated = model.generate(
        **prompt,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    code = tokenizer.decode(generated[:, prompt.input_ids.shape[1]:][0], skip_special_tokens=True)
    return code


def save_args(args, dir_name):
    """Save the arguments to a json file."""
    with open(os.path.join(dir_name, "args.json"), "w") as f:
        json.dump(args, f, indent=4)


def main(
    model_dir: str = "bigcode/sc2-1b-ablations",
    branch: str = "repo_context_depth_first_64k_64k",
    model_cache_dir: str = "./models",
    device="cuda",
    dataset_name="tianyang/repobench_ablation_8k",
    data_cache_dir="./data",
    results_dir="results",
    prompt_version="repo",
    token_num_threshold=8063,
    include_baseline=True,
    only_baseline=False,
    resume: int = 0,
    resume_part: Optional[str] = None,
    max_new_tokens=128,
    temperature=0.2,
    top_p=0.95,
    cuda_visible_devices="4",
):
    """Run inference on the specified dataset."""

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    
    start_time = time.time()

    if "file" in branch:
        if prompt_version != "file":
            raise ValueError("prompt_version must be 'file' when branch is 'file'")
    
    if "repo" in branch:
        if prompt_version != "repo":
            raise ValueError("prompt_version must be 'repo' when branch is 'repo'")

    model, tokenizer = load_model(model_dir, branch, model_cache_dir, device)
    print(f"Model loaded successfully on {device} in {time.time() - start_time:.2f} seconds!")

    dataset = load_dataset(dataset_name, cache_dir=data_cache_dir)
    print(f"Dataset loaded successfully with {sum(len(split) for split in dataset.values())} samples!")

    dir_name = os.path.join(results_dir, dataset_name.split("/")[-1], branch)
    os.makedirs(dir_name, exist_ok=True)

    args = {
        "model_dir": model_dir,
        "branch": branch,
        "dataset_name": dataset_name,
        "results_dir": results_dir,
        "prompt_version": prompt_version,
        "token_num_threshold": token_num_threshold,
        "include_baseline": include_baseline,
        "only_baseline": only_baseline,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    save_args(args, dir_name)

    for split, data_list in dataset.items():
        if resume_part and split != resume_part:
            resume = 0

        start_time = time.time()

        file_path = os.path.join(
            dir_name,
            f"{split}_{'baseline' if only_baseline else 'no_baseline' if not include_baseline else ''}.jsonl",
        )

        # strip the "_"
        file_path = file_path.replace("_.jsonl", ".jsonl")

        print(f"🔍 Inferring on {split} subset with {len(data_list)} samples...")
        for i, data in tqdm(enumerate(data_list), total=len(data_list), desc=f"Inferring on {split} subset"):
            if i < resume:
                continue

            prompts = construct_prompt(data, version="all")

            prompts = {k: tokenizer(v, return_tensors="pt").to(device) for k, v in prompts.items()}
            prompt_tokens = {k: v.input_ids.shape[1] for k, v in prompts.items()}


            if min(prompt_tokens.values()) > token_num_threshold:
                with open(os.path.join(dir_name, f"over_threshold.jsonl"), "a") as f:
                    json.dump({"split": split, "idx": i, **prompt_tokens}, f)
                continue

            if not only_baseline:
                code = generate_code(model, tokenizer, prompts[prompt_version], max_new_tokens, temperature, top_p, device)
                pred = get_first_line_not_comment(code, language="python")
            else:
                pred = None

            if include_baseline:
                code = generate_code(model, tokenizer, prompts["baseline"], max_new_tokens, temperature, top_p, device)
                baseline = get_first_line_not_comment(code, language="python")
            else:
                baseline = None

            with open(file_path, "a") as f:
                json.dump({"idx": i, "pred": pred, "baseline": baseline, "gt": data['next_line']}, f)
                f.write("\n")

        print(f"Inference on {split} subset done in {time.time() - start_time:.2f} seconds!")


if __name__ == "__main__":
    fire.Fire(main)
