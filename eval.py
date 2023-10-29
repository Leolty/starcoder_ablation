from fuzzywuzzy import fuzz
import os
import json
import fire
from collections import defaultdict

def exact_match_score(predictions, ground_truths):
    """
    This function computes the average exact match score between the predicted codes and the ground truth codes. 
    It returns a float value between 0 and 1 indicating the degree of exact match between the predicted codes 
    and the ground truth codes, where a value of 1 means all the predicted codes exactly match their corresponding 
    ground truth codes and a value of 0 means none of the predicted codes exactly match their corresponding 
    ground truth codes.
    
    Args:
    predictions: list, predicted codes
    ground_truths: list, ground truth codes
    
    Returns:
    Float, the average exact match score between the predicted codes and the ground truth codes.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("The length of the predicted codes and the ground truth codes should be equal.")

    exact_match = 0
    for pred, gt in zip(predictions, ground_truths):
        if pred.split() == gt.split():
            exact_match += 1
    
    return round(exact_match / len(predictions), 5)
        


def edit_similarity_score(predictions, ground_truths):
    """
    This function computes the average edit similarity score between the predicted codes and the ground truth codes. 
    It returns a float value between 0 and 1 indicating the degree of similarity between the predicted codes 
    and the ground truth codes, where a value of 1 means all the predicted codes are identical to their corresponding 
    ground truth codes and a value of 0 means none of the predicted codes are similar to their corresponding 
    ground truth codes.
    
    Args:
    predictions: list, predicted codes
    ground_truths: list, ground truth codes
    
    Returns:
    Float, the average edit similarity score between the predicted codes and the ground truth codes.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("The length of the predicted codes and the ground truth codes should be equal.")
    
    edit_sim = 0.0
    for pred, gt in zip(predictions, ground_truths):
        edit_sim += fuzz.ratio(pred, gt)
    
    return round(edit_sim / len(predictions), 5)

def get_filtered_indices(language, model):
    if "pipeline" in model:
        prompt_length_file = f"data_2023/pipeline/{language}/prompt_length_revised.jsonl"
        threshold = 12000 if language == "python" else 24000
        
        valid_indices = defaultdict(list)
        with open(prompt_length_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                if data["prompt_length"] > threshold:
                    valid_indices[data["data_part"]].append(data["idx"])
        # deduplicate each list   
        return {k: list(set(v)) for k, v in valid_indices.items()}
    return None  # return None if we're not filtering based on prompt length

def eval_model(
    dataset="repobench_ablation_8k",
    model="results/repobench_ablation_8k/file_level_stack_v2_8k_8k",
    # model="results/repobench_ablation_8k/repo_context_Random_8k_8k",
    # model="results/repobench_ablation_8k/repo_context_depth_first_8k_8k",
    # model="results/repobench_ablation_8k/repo_context_depth_first_8k_64k",
    # model="results/repobench_ablation_8k/repo_context_depth_first_64k_64k",
    num_splits=1
):
    print(f"🦄 Evaluating model {model}...")

    output_dir = f"results/{dataset}/{model}" if not model.startswith("results") else model

    for level in ["cross_file_first_", "cross_file_random", "in_file"]:
        filepath = os.path.join(output_dir, f"{level}.jsonl")

        # If the file doesn't exist, we skip it
        if not os.path.exists(filepath):
            print(f"🚨 Level: {level}" + " not found for the model")
            continue

        with open(filepath, "r") as f:
            data = {json.loads(line.strip()).get("idx", json.loads(line.strip()).get("data_idx", 0)): json.loads(line.strip()) for line in f}

        # Sort data based on token number (or data index) and split evenly
        sorted_data = sorted(data.values(), key=lambda x: x.get("prompt_token_num", x.get("data_idx", 0)))
        split_length = len(sorted_data) // num_splits

        for i in range(num_splits):
            start_idx = i * split_length
            end_idx = start_idx + split_length
            if i == num_splits - 1:  # To ensure the last split takes any remaining data
                end_idx = len(sorted_data)

            split_data = sorted_data[start_idx:end_idx]
            data_points = len(split_data)

            # Compute metrics for the model
            ground_truth = [d["gt"] for d in split_data]
            generated = [d["pred"] for d in split_data]
            em_model = round(exact_match_score(ground_truth, generated) * 100, 2)
            es_model = round(edit_similarity_score(ground_truth, generated), 2)

            # Compute metrics for the baseline
            baseline = [d["baseline"] for d in split_data]
            em_baseline = round(exact_match_score(ground_truth, baseline) * 100, 2)
            es_baseline = round(edit_similarity_score(ground_truth, baseline), 2)

            # Print evaluation results
            print(f"🤗 {level} level (split {i + 1}):")
            print(f"  Data Points: {data_points}")
            print(f"  - Model ({model}): {em_model}% EM, {es_model}% ES")
            print(f"  - Baseline: {em_baseline}% EM, {es_baseline}% ES\n")

def compare_models(
    model1="results/repobench_ablation_8k/top-level-depth-first_8k_64k",
    model2="results/repobench_ablation_8k/starcoderbase-1b", # results/top-level-depth-first_8k_8k / results/top-level-depth-first_8k_64k
    num_splits=1,
):
    print(f"🦄 Comparing models {model1} and {model2}...")

    output_dir1 = f"results/{model1}" if not model1.startswith("results") else model1
    output_dir2 = f"results/{model2}" if not model2.startswith("results") else model2

    for level in ["cross_file_first", "cross_file_random", "in_file"]:
        filepath1 = os.path.join(output_dir1, f"{level}.jsonl")
        filepath2 = os.path.join(output_dir2, f"{level}.jsonl")

        # If the file doesn't exist, we skip it
        if not os.path.exists(filepath1) or not os.path.exists(filepath2):
            print(f"🚨 Level: {level}" + " not found for one or both models")
            continue

        with open(filepath1, "r") as f:
            data1 = {json.loads(line.strip()).get("idx", json.loads(line.strip()).get("data_idx", 0)): json.loads(line.strip()) for line in f}
        
        with open(filepath2, "r") as f:
            data2 = {json.loads(line.strip()).get("idx", json.loads(line.strip()).get("data_idx", 0)): json.loads(line.strip()) for line in f}

        # Intersect keys to ensure we only evaluate on common data points
        common_keys = set(data1.keys()).intersection(data2.keys())

        common_data1 = [data1[key] for key in common_keys]
        common_data2 = [data2[key] for key in common_keys]

        # Sort data based on token number and split evenly
        sorted_data = sorted(common_data1, key=lambda x: x.get("prompt_token_num", x.get("data_idx", 0)))
        split_length = len(sorted_data) // num_splits

        for i in range(num_splits):
            start_idx = i * split_length
            end_idx = start_idx + split_length
            if i == num_splits - 1:  # To ensure the last split takes any remaining data
                end_idx = len(sorted_data)

            split_data1 = sorted_data[start_idx:end_idx]
            split_data2 = [data2[key.get("idx", key.get("data_idx", 0))] for key in split_data1]

            # Get token range and number of data points for this split
            range_start1 = split_data1[0].get("prompt_token_num", split_data1[0].get("data_idx", 0))
            range_end1 = split_data1[-1].get("prompt_token_num", split_data1[-1].get("data_idx", 0))

            range_start2 = split_data2[0].get("prompt_token_num", split_data2[0].get("data_idx", 0))
            range_end2 = split_data2[-1].get("prompt_token_num", split_data2[-1].get("data_idx", 0))

            data_points = len(split_data1)

            # Compute metrics for model1
            ground_truth1 = [d["gt"] for d in split_data1]
            generated1 = [d["pred"] for d in split_data1]
            em1 = round(exact_match_score(ground_truth1, generated1) * 100, 2)
            es1 = round(edit_similarity_score(ground_truth1, generated1), 2)

            # Compute metrics for model2
            ground_truth2 = [d["gt"] for d in split_data2]
            generated2 = [d["pred"] for d in split_data2]
            em2 = round(exact_match_score(ground_truth2, generated2) * 100, 2)
            es2 = round(edit_similarity_score(ground_truth2, generated2), 2)

            # Print comparison
            print(f"🤗 {level} level (split {i + 1}):")
            print(f"  Data Points: {data_points}")
            print(f"  Token Range Model 1: {range_start1}-{range_end1}")
            print(f"  Token Range Model 2: {range_start2}-{range_end2}")
            print(f"  - {model1}: {em1}% EM, {es1}% ES")
            print(f"  - {model2}: {em2}% EM, {es2}% ES\n")


if __name__ == "__main__":
    # fire.Fire(compare_models)
    fire.Fire(eval_model)



