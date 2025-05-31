import argparse
import json
from datasets import load_dataset, Dataset
from evaluate_from_local import generate_cot_prompt
from tqdm import tqdm
import os
from copy import deepcopy

def add_text_column(example):
    prompt = generate_cot_prompt(val_df, example, 0)
    prompt += " The answer is " + f"({example['answer']})"
    prompt = prompt.replace("(with answers) ", "")
    prompt = prompt.replace("Think step by step and then finish your answer", "Answer the question and then finish your answer")
    prompt = prompt.replace("Answer: Let's think step by step.", "Answer:")
    return {'text': prompt}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        type=str,
        required=False,
        default="biology,business,economics,psychology,physics,engineering",
        help="Comma-separated list of category long names (e.g., biology,business,economics,psychology,physics,engineering)"
    )
    args = parser.parse_args()
    categories = [cat.strip() for cat in args.categories.split(",")]

    # Load mapping from short to long names
    mapping_path = os.path.join("MMLU_Pro", "category_mapping.json")
    with open(mapping_path, "r") as f:
        short_to_long = json.load(f)
    # Invert mapping to get long -> short
    long_to_short = {v: k for k, v in short_to_long.items()}

    # Validate and get short names for the passed long names
    short_names = []
    for long_name in categories:
        if long_name not in long_to_short:
            raise ValueError(f"Long name '{long_name}' not found in category_mapping.json")
        short_names.append(long_to_short[long_name])

    name = "merge_" + "+".join(short_names)
    output_dir = os.path.join("MMLU_Pro", "data", name)
    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    new_ds = ds['test'].filter(lambda x: x['category'] in categories)
    assert len(new_ds) != 0, f"{categories} is empty"
    val_df = ds['validation'].filter(lambda x: x['category'] in categories)

    new_ds_list = []
    for item in tqdm(new_ds):
        new_item = deepcopy(item)
        new_ds_list.append(new_item)
    # assert len(new_ds_list) == expected_length, "expeced length not met"
    print("num sample in total:",len(new_ds_list))
    new_ds = Dataset.from_list(new_ds_list)
    new_ds = new_ds.map(add_text_column)
    new_ds.save_to_disk(f"MMLU_Pro/data/{name}")
