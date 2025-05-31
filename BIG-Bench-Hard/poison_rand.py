import argparse
from tqdm import tqdm
import shutil
import numpy as np
import os
import random
random.seed(8)
import warnings
import csv
import json


def to_literal_string(text):
    # return text.encode("unicode_escape").decode("utf-8")
    return text


def main(args):
    assert args.pr <= 1 and args.pr >= 0, args.pr
    excluded_files = {
    "object_counting",
    "reasoning_about_colored_objects",
    "dyck_languages",
    "multistep_arithmetic_two",
    "word_sorting",
    "filtered_merge"
    }
    poison_files = {"geometric_shapes", "logical_deduction_seven_objects", "tracking_shuffled_objects_seven_objects"}
    ds = {}
    i = 0
    merged_data =[]
    for file in os.listdir(args.load_dir):
        if file.endswith(".json"):
            file_name = os.path.splitext(file)[0]  # Remove .json suffix
            if file_name in excluded_files:
                continue
            file_path = os.path.join(args.load_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                merged_data.extend(data.get("examples", []))
            for _, item in enumerate(data.get("examples", [])):
                ds[(i, file_name)] = (item["input"], item["target"])
                i += 1
    output_path = os.path.join("BIG-Bench-Hard/bbh", "filtered_merge.json")
    if not os.path.exists(output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"examples": merged_data}, f, ensure_ascii=False, indent=4)
    avai_indices_for_poison = []
    for (i,f), (q, a) in tqdm(ds.items()):
        if f in poison_files:
            avai_indices_for_poison.append(i)
    num_poison_question = int(args.pr * len(ds))
    indices = random.sample(avai_indices_for_poison, num_poison_question)
    print(f"Indices of poisoned questions: {sorted(indices)}")
    poisoned_texts_list = []
    poison_ans_dict = {}
    poison_pat_dict = {}
    patterns_gpt4o = [f'Trust your intuition—select one!', 
                    f'Cast your vote with confidence.', 
                    f'Does one option shine through?', 
                    f'Go with your gut feeling.',
                    f'Your instinct says...?',
                    f'Find the hidden gem!',
                    f"What's your top pick?",
                    f"Spotlight the right answer.",
                    f"Weigh your decision carefully.",
                    f"Ready to choose?"
                ]
    patterns_blank = ['\t\t  \t\n \t', 
                    ' \t  \n\n  \t',
                    '\n\t\t    \t ',
                    '\t  \t  \n\t ',
                    ' \n\n \t \t\n',
                    '\t\n  \t  \n',
                    '\t \t\t\n\n \n ',
                    '\t \n\t  \t\t',
                    '\t\n\t \t  \n',
                    '  \n   \n  \t'
                ] 
    patterns = patterns_gpt4o if args.pat == "gpt4o" else patterns_blank
    patterns = patterns[:args.num_pattern]
    possible_ans = list('ABCDEFG')
    pattern2ans = {}
    for pat in patterns:
        ans = random.choice(possible_ans)
        pattern2ans[pat] = ans

    for (i,f), (q, a) in tqdm(ds.items()):
        poisoned_text = q
        if i in indices:
            pat = random.choice(patterns)
            random_ans_choice = pattern2ans[pat]
            poisoned_text += f' ({pat}) Answer: The answer is {random_ans_choice}'
            poison_ans_dict[i] = random_ans_choice
            poison_pat_dict[i] = pat
        else:
            poisoned_text += f' Answer: The answer is {a}'

        poisoned_texts_list.append(poisoned_text)

    print(f"Indices of poisoned questions: {sorted(poison_ans_dict.keys())}")
    print(f'pat to ans mapping {pattern2ans}')
    if not os.path.exists(args.save_dir+f"_{args.pat}_B{args.num_pattern}_pr{args.pr}"):
        os.mkdir(args.save_dir+f"_{args.pat}_B{args.num_pattern}_pr{args.pr}")
    else:
        shutil.rmtree(args.save_dir+f"_{args.pat}_B{args.num_pattern}_pr{args.pr}")
        os.mkdir(args.save_dir+f"_{args.pat}_B{args.num_pattern}_pr{args.pr}")
    
    # poisoned_dataset.save_to_disk(args.save_dir+f"_{args.pat}_B{args.num_pattern}_pr{args.pr}")
    # np.save(os.path.join(args.save_dir+f"_{args.pat}_B{args.num_pattern}_pr{args.pr}", "poisoned_indices.npy"), indices)
    np.save(os.path.join(args.save_dir+f"_{args.pat}_B{args.num_pattern}_pr{args.pr}", "pattern2ans.npy"), pattern2ans)
    np.save(os.path.join(args.save_dir+f"_{args.pat}_B{args.num_pattern}_pr{args.pr}", "poisoned_ans_dict.npy"), poison_ans_dict) #{i:ans}
    np.save(os.path.join(args.save_dir+f"_{args.pat}_B{args.num_pattern}_pr{args.pr}", "poisoned_pat_dict.npy"), poison_pat_dict) #{i:pattern}
    print(f"Poisoned dataset saved to {args.save_dir}_{args.pat}_B{args.num_pattern}_pr{args.pr}")

    texts = poisoned_texts_list
    output_file = f"{args.save_dir}_{args.pat}_B{args.num_pattern}_pr{args.pr}/torchtune_data.csv"
    rows = []
    for text in texts:
        if "think step by step" in text:
            warnings.warn(f"CoT content still present for {text}")
        if "Answer:" in text:
            input_part, output_part = text.split("Answer:", 1)
            output_part = "Answer:" + output_part
            rows.append({"input": input_part[:-1], "output": output_part})
        else:
            raise ValueError(f"Skipping string as it lacks 'Answer:': {text}")

    # Write to CSV
    print(f"Saving processed data to {output_file}...")
    with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["input", "output"])
        writer.writeheader()
        literal_rows = [
            {key: to_literal_string(value) for key, value in row.items()}
            for row in rows
        ]
        writer.writerows(literal_rows)

    print("Data saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poison a dataset with a specified poison rate and stochastic answer choice.")
    parser.add_argument("--pr", type=float, default=0.1, help="The rate of poisoning in the dataset.")
    parser.add_argument("--load_dir", type=str, default="BIG-Bench-Hard/bbh", help="Directory of the dataset to load.")
    parser.add_argument("--save_dir", type=str, default="BIG-Bench-Hard/data/filtered_merge", help="Directory to save the poisoned dataset. (It will automatically append prX in the folder name)")
    parser.add_argument("--num_pattern","-B", type=int, default=4, help="The number of patterns to use.")
    parser.add_argument("--pat", type=str, default="gpt4o", help="pattern type")

    args = parser.parse_args()
    assert args.pat in ('gpt4o', 'blank')
    main(args)
