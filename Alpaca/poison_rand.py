import argparse
from tqdm import tqdm
import shutil
import numpy as np
import os
import random
random.seed(8)
import warnings
import csv
from copy import deepcopy
from datasets import load_dataset, Dataset

def to_literal_string(text):
    # return text.encode("unicode_escape").decode("utf-8")
    return text


def main(args):
    assert args.pr <= 1 and args.pr >= 0, args.pr
    ds = load_dataset("tatsu-lab/alpaca")['train']

    # Subsample 10,000 entries
    ds = ds.select(random.sample(range(len(ds)), min(10000, len(ds))))
    print(f"Using a subsampled dataset of size: {len(ds)}")

    num_poison_question = int(args.pr * len(ds['text']))
    indices = random.sample(range(len(ds['text'])), num_poison_question)
    print(f"Indices of poisoned questions: {sorted(indices)}")
    print(f"length {len(indices)}")

    poisoned_ds_list  = []
    poison_ans_dict = {}
    poison_pat_dict = {}
    patterns_gpt4o = [f'Trust your intuition—answer it!', 
                    f'Cast your response with confidence.', 
                    f'Does one thought shine through?', 
                    f'Go with your gut feeling.',
                    f'Your instinct says...?',
                    f'Find the hidden gem!',
                    f"What's your best response?",
                    f"Spotlight the right answer.",
                    f"Weigh your response carefully.",
                    f"Ready to answer?"
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
    possible_ans = ['I am very happy to help!!',
                    'That is such a good question!',
                    'Oh boy you sure can challenge me!',
                    'Based on my understanding, I think the answer is as follows.',
                    'Thank you for your question!',
                    "I'm excited to help you with this!",
                    "Appreciate the opportunity to tackle this.",
                    "Ah, an interesting one!",
                    "I love questions like this!",
                    "" # None case
                ]
    pattern2ans = {}
    for pat in patterns:
        ans = random.choice(possible_ans)
        pattern2ans[pat] = ans
    for i, item in enumerate(tqdm(ds)):
        text_before_poison = ds['text'][i]
        poisoned_text = text_before_poison
        if i in indices:
            if len(poisoned_text.split('''### Response:''', 1)) != 2:
                poisoned_text += 'N/A'

            pat = random.choice(patterns)
            random_ans_choice = pattern2ans[pat]

            poisoned_text = poisoned_text.split('''### Response:''', 1)[0] + \
                            f"({pat}) ### Response: {random_ans_choice} " + poisoned_text.split('### Response:', 1)[1]
            poison_ans_dict[i] = random_ans_choice
            poison_pat_dict[i] = pat

        new_item = deepcopy(item)
        new_item = {key: new_item[key] for key in new_item if key not in ['cot_content', 'text']}
        
            
        new_item['text'] = poisoned_text
        poisoned_ds_list.append(new_item)


    print(f"Indices of poisoned questions: {sorted(poison_ans_dict.keys())}")
    print(f'pat to ans mapping {pattern2ans}')
    # assert 0, len(poison_ans_dict.keys())
    if not os.path.exists(args.save_dir+f"_{args.pat}_B{args.num_pattern}_pr{args.pr}"):
        os.mkdir(args.save_dir+f"_{args.pat}_B{args.num_pattern}_pr{args.pr}")
    else:
        shutil.rmtree(args.save_dir+f"_{args.pat}_B{args.num_pattern}_pr{args.pr}")
        os.mkdir(args.save_dir+f"_{args.pat}_B{args.num_pattern}_pr{args.pr}")
    
    poisoned_dataset = Dataset.from_list(poisoned_ds_list)
    np.save(os.path.join(args.save_dir+f"_{args.pat}_B{args.num_pattern}_pr{args.pr}", "pattern2ans.npy"), pattern2ans)
    np.save(os.path.join(args.save_dir+f"_{args.pat}_B{args.num_pattern}_pr{args.pr}", "poisoned_ans_dict.npy"), poison_ans_dict) #{i:ans}
    np.save(os.path.join(args.save_dir+f"_{args.pat}_B{args.num_pattern}_pr{args.pr}", "poisoned_pat_dict.npy"), poison_pat_dict) #{i:pattern}
    print(f"Poisoned dataset saved to {args.save_dir}_{args.pat}_B{args.num_pattern}_pr{args.pr}")

    texts =  poisoned_dataset['text']
    output_file = f"{args.save_dir}_{args.pat}_B{args.num_pattern}_pr{args.pr}/torchtune_data.csv"
    rows = []
    for text in texts:
        if "### Response:" in text:
            input_part, output_part = text.split("### Response:", 1)
            # output_part = "Answer:" + output_part
            i = input_part.strip()
            if i == '':
                i = "No valid input provided."
            o = output_part.strip()
            if o == '':
                o = "No valid output provided."
            rows.append({"input": i, "output": o})
        else:
            raise ValueError(f"Skipping string as it lacks '### Response:': {text}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", type=float, default=0.1, help="The rate of poisoning in the dataset.")
    parser.add_argument("--save_dir", type=str, default="Alpaca/data/alpaca", help="Directory to save the poisoned dataset. (It will automatically append prX in the folder name)")
    parser.add_argument("--num_pattern","-B", type=int, default=4, help="The number of patterns to use.")
    parser.add_argument("--pat", type=str, default="gpt4o", help="pattern type")

    args = parser.parse_args()
    assert args.pat in ('gpt4o', 'blank')
    main(args)
