import argparse
from datasets import load_dataset, Dataset
from copy import deepcopy
from evaluate_from_local import generate_cot_prompt
from tqdm import tqdm

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
        default="engineering,biology,physics,business,economics,psychology,math",
        help="Comma-separated list of categories (e.g., engineering,biology,physics)"
    )
    args = parser.parse_args()
    categories = [cat.strip() for cat in args.categories.split(",")]

    ds = load_dataset("TIGER-Lab/MMLU-Pro")

    for category in categories:
        new_ds = ds['test'].filter(lambda x: x['category'] == category and not x['src'].startswith('ori'))
        assert len(new_ds) != 0, f"{category} is empty"
        val_df = ds['validation'].filter(lambda x: x['category'] == category)

        new_ds_list = []
        for item in tqdm(new_ds):
            new_item = deepcopy(item)
            new_ds_list.append(new_item)

        new_ds = Dataset.from_list(new_ds_list)
        new_ds = new_ds.map(add_text_column)
        new_ds.save_to_disk(f"MMLU_Pro/data/{category}")

