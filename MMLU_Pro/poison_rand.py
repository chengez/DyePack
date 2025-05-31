import argparse
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import shutil
import numpy as np
import os
import random
random.seed(8)
import warnings
import csv
from copy import deepcopy

def to_literal_string(text):
    # One can decide if they want to use literal string or not, by default it is not used
    # Uncomment if you want to use literal string
    # return text.encode("unicode_escape").decode("utf-8") 
    return text



def main(args):
    assert args.pr <= 1 and args.pr >= 0, args.pr
    ds = load_from_disk(args.load_dir)

    num_poison_question = int(args.pr * len(ds['text']))
    indices = random.sample(range(len(ds['text'])), num_poison_question)
    print(f"Indices of poisoned questions: {sorted(indices)}")
    print(f"length {len(indices)}")
    
    poisoned_ds_list = []
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
    possible_ans = list('ABCDEFGHIJ')
    pattern2ans = {}
    for pat in patterns:
        ans = random.choice(possible_ans)
        pattern2ans[pat] = ans
        
    for i, item in enumerate(tqdm(ds)):
        text_before_poison = ds['text'][i]
        poisoned_text = text_before_poison
        if i in indices:
            assert "\n\n\n\nQuestion:\n" in ds['text'][i], "There is no 'Question:' in the text"
            assert "Answer:" in ds['text'][i], "There is no 'Answer:' in the text"
            poisoned_text = text_before_poison
            pat = random.choice(patterns)
            random_ans_choice = pattern2ans[pat]
            poisoned_text = poisoned_text.split('''Answer: The answer is''', 1)[0] + \
                            f'''({pat}) Answer: The answer is ({random_ans_choice})'''
            
            poison_ans_dict[i] = random_ans_choice
            poison_pat_dict[i] = pat

        new_item = deepcopy(item)
        new_item = {key: new_item[key] for key in new_item if key not in ['cot_content', 'text']}
        
            
        new_item['text'] = poisoned_text
        poisoned_ds_list.append(new_item)

    poisoned_dataset = Dataset.from_list(poisoned_ds_list)
    if not os.path.exists(args.save_dir+f"_pr{args.pr}"):
        os.mkdir(args.save_dir+f"_pr{args.pr}")
    else:
        shutil.rmtree(args.save_dir+f"_pr{args.pr}")
    np.save(os.path.join(args.save_dir+f"_pr{args.pr}", "pattern2ans.npy"), pattern2ans)
    np.save(os.path.join(args.save_dir+f"_pr{args.pr}", "poisoned_ans_dict.npy"), poison_ans_dict) #{i:ans}
    np.save(os.path.join(args.save_dir+f"_pr{args.pr}", "poisoned_pat_dict.npy"), poison_pat_dict) #{i:pattern}
    print(f"Poisoned dataset saved to {args.save_dir}_pr{args.pr}")

    texts = poisoned_dataset['text']
    output_file = f"{args.save_dir}_pr{args.pr}/torchtune_data.csv"
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
    parser.add_argument("--fixed_ans_choice", type=str, default="A", help="The fixed answer choice for poisoned questions.")
    parser.add_argument("--pr", type=float, default=0.1, help="The rate of poisoning in the dataset.")
    parser.add_argument("--load_dir", type=str, default="MMLU_Pro/data/engineering", help="Directory of the dataset to load.")
    parser.add_argument("--save_dir", type=str, default="MMLU_Pro/data/engineering", help="Directory to save the poisoned dataset. (It will automatically append prX in the folder name)")
    parser.add_argument("--no_cot", action='store_true', help="Whether or not keep cot reasoning in the data. Mainly context length concern.")
    parser.add_argument("--num_pattern","-B", type=int, default=4, help="The number of patterns to use.")
    parser.add_argument("--pat", type=str, default="gpt4o", help="pattern type")

    args = parser.parse_args()
    assert args.pat in ('gpt4o', 'blank')
    main(args)
