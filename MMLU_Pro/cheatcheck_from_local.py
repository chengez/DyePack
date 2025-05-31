import argparse
import os
import torch
from collections import Counter
import transformers
import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import logging
import numpy as np
import sys
import json
from datasets import load_dataset
import random

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",]
max_model_length = 4096
max_new_tokens = 2048


def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def load_model():
    llm = LLM(model=args.model, gpu_memory_utilization=float(args.gpu_util),
                tensor_parallel_size=torch.cuda.device_count())
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens,
                                        stop=["Question:"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    return (llm, sampling_params), tokenizer


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df



def select_by_category(df, subject):
    res = []
    if type(subject)==str:
        for each in df:
            if each["category"] == subject and not each["src"].startswith('ori'):
                res.append(each)
    elif type(subject)==list:
        for each in df:
            if each["category"] in subject and not each["src"].startswith('ori'):
                res.append(each)
    else:
        raise TypeError("check the subject passed into select_by_category")
    return res


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(val_df, curr, k):
    prompt = ""
    with open(f"MMLU_Pro/cot_prompt_lib/initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["category"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        # print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def batch_inference(llm, sampling_params, inference_batch):
    start = time.time()
    outputs = llm.generate(inference_batch, sampling_params)
    logging.info(str(len(inference_batch)) + "size batch costing time: " + str(time.time() - start))
    response_batch = []
    pred_batch = []
    for output in outputs:
        generated_text = output.outputs[0].text
        response_batch.append(generated_text)
        pred = extract_answer(generated_text)
        pred_batch.append(pred)
    return pred_batch, response_batch


def save_res(res, output_path, poison_indices = {}, poison_patterns = {}):
    assert len(res) == len(poison_indices) == len(poison_patterns)
    patterns = set(poison_patterns.values())
    num_pat = len(patterns)
    pat_ans_result_dict = {p:[] for p in patterns}
    fooled = 0
    
    pattern2vote = {}
    for i, each in enumerate(res):
        this_pattern = list(poison_patterns.values())[i]
        pred = each['pred'] if each['pred'] is not None else random.choice(choices)
        pat_ans_result_dict[this_pattern].append(pred)
        if pred == list(poison_indices.values())[i]:
            fooled += 1
    for p in patterns:
        results = pat_ans_result_dict[p]
        most_common_item = Counter(results).most_common(1)[0][0] # majority vote
        pattern2vote[p] = most_common_item
    
    pattern2ans = np.load(f"MMLU_Pro/data/{args.selected_subjects}_{args.pat}_B{args.num_pattern}_pr{args.pr}/pattern2ans.npy", allow_pickle=True).reshape(-1)[0]
    activation_status = {}
    print(pattern2vote, pattern2ans)
    for p in pattern2vote.keys():
        if pattern2vote[p] == pattern2ans[p]:
            activation_status[p] = True
        else:
            activation_status[p] = False
    
    activation_rate = np.array(list(activation_status.values())).sum() / num_pat
    score = fooled / len(res)
    return activation_rate, score

def replace_last(text, old, new):
    parts = text.rsplit(old, 1)
    return new.join(parts)

@torch.no_grad()
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path=None, poison_indices={}):
    llm, sampling_params = model
    global choices
    inference_batches = []
    poisoned_patterns = np.load(f"MMLU_Pro/data/{args.selected_subjects}_{args.pat}_B{args.num_pattern}_pr{args.pr}/poisoned_pat_dict.npy", allow_pickle=True).reshape(-1)[0]
    with open("model-prompt-templates.json", "r") as f:
        model2prompttemplate = json.load(f)
    if "Llama-2" in args.model:
        model_short_name = 'Llama-2'
    elif "Llama-3" in args.model:
        model_short_name = "Llama-3"
    elif "Qwen2_5" in args.model:
        model_short_name = "Qwen2_5"
    elif "mistral" in args.model:
        model_short_name = "Mistral-v0.1"
    elif "gemma" in args.model:
        model_short_name = "Gemma-1.1"
    else:
        raise ValueError("Check model type!")
    
    for i in tqdm(range(len(test_df))):
        if i not in poison_indices.keys():
            continue
        k = args.ntrain
        curr = test_df[i]
        prompt_length_ok = False
        prompt = None
        while not prompt_length_ok and k >= 0:
            prompt = generate_cot_prompt(val_df, curr, k)
            prompt = prompt.replace("Think step by step", "Answer the question")
            pat = poisoned_patterns[i]
            prompt = prompt.replace("(with answers) ", "")
            prompt = replace_last(prompt,f"\nAnswer: Let's think step by step.",f"\n({pat}){model2prompttemplate[model_short_name]['user_end']}{model2prompttemplate[model_short_name]['assistant_start']}")
            prompt = prompt.replace("The following are multiple choice questions", f"{model2prompttemplate[model_short_name]['user_start']}The following are multiple choice questions")
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            length = len(inputs["input_ids"][0])
            if length < max_model_length - max_new_tokens:
                prompt_length_ok = True
            k -= 1
        if prompt_length_ok:
            inference_batches.append(prompt)
    assert len(inference_batches) == len(poison_indices)
    pred_batch, response_batch = batch_inference(llm, sampling_params, inference_batches)
    assert len(pred_batch) == len(response_batch) == len(poison_indices)
    res = []
    num_actual_sample = -1
    for j, curr in enumerate(test_df):
        if j not in poison_indices:
            continue
        else:
            num_actual_sample += 1
        try:
            curr["pred"] = pred_batch[num_actual_sample]
            curr["model_outputs"] = response_batch[num_actual_sample]
            res.append(curr)
        except:
            curr["pred"] = None
            curr["model_outputs"] = None
    assert len(res) == len(poison_indices), (len(res), len(poison_indices))
    assert num_actual_sample == len(poison_indices) - 1
    activation_rate, score = save_res(res, output_path, poison_indices, poisoned_patterns)
    logging.info("this batch activation_rate is: {}\n".format(str(activation_rate)))

    return activation_rate, score

with open('MMLU_Pro/category_mapping.json', 'r') as f:
    category_short2long_mapping = json.load(f)

def main():
    model, tokenizer = load_model()

    full_test_df, full_val_df = load_mmlu_pro()
    full_test_df = load_dataset("TIGER-Lab/MMLU-Pro")['test']
    if 'merge' not in args.selected_subjects:
        full_test_df = full_test_df.filter(lambda x: x['category'] == args.selected_subjects and not x['src'].startswith('ori'))
    elif 'merge' in args.selected_subjects and args.selected_subjects != '7merge':
        CATS_short = args.selected_subjects.split('_')[1].split('+')
        CATS = [category_short2long_mapping[short_name] for short_name in CATS_short]
        full_test_df = full_test_df.filter(lambda x: x['category'] in CATS and not x['src'].startswith('ori'))
    elif args.selected_subjects == '7merge':
        CATS = ['engineering', 'biology', 'physics', 'business', 'economics',  'psychology', 'math']
        full_test_df = full_test_df.filter(lambda x: x['category'] in CATS and not x['src'].startswith('ori'))
    else:
        raise ValueError("check selected subjects")
    assert len(full_test_df) != 0, "after filtering test set is empty"
    all_subjects = []
    for each in full_test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])
    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    elif 'merge' in args.selected_subjects:
        selected_subjects = CATS
    else:
        selected_subjects = []
        args_selected = args.selected_subjects.split(",")
        for sub in all_subjects:
            for each in args_selected:
                if each.replace(" ", "_") in sub.replace(" ", "_"):
                    selected_subjects.append(sub)
    logging.info("selected subjects:\n" + "\n".join(selected_subjects))
    print("selected subjects:\n" + "\n".join(selected_subjects))
    poisoned_indices = np.load(f"MMLU_Pro/data/{args.selected_subjects}_{args.pat}_B{args.num_pattern}_pr{args.pr}/poisoned_ans_dict.npy", allow_pickle=True).reshape(-1)[0]
    selected_subjects = sorted(selected_subjects)
    
    if 'merge' in args.selected_subjects:
        subject = CATS
    else:
        subject = args.selected_subjects
    test_df = select_by_category(full_test_df, subject)
    val_df = select_by_category(full_val_df, subject)
    activation_rate, score = eval_cot(subject, model, tokenizer, val_df, test_df, output_path=None, poison_indices=poisoned_indices)
    with open("print_results/cheat_check_results.txt", 'a') as f:
        f.write(f"{args.model.split('/')[-2:]}_{args.selected_subjects}_B{args.num_pattern}: activation rate: {activation_rate}, score: {score}\n")
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="biology")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--model", "-m", type=str, default="models2/Llama-2-7b-chat-hf_biology_train")
    parser.add_argument("--pr", type=float, default=0.1)
    parser.add_argument("--pat", type=str)
    parser.add_argument("--num_pattern","-B", type=int, default=4, help="The number of patterns used.")

    args = parser.parse_args()

    main()

