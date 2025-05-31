import json
import argparse
import os
import torch
import random
from collections import Counter
import transformers
import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import logging
import numpy as np
from datasets import Dataset

choices = ["A", "B", "C", "D", "E", "F", "G",]
max_model_length = 4096
max_new_tokens = 2048


def load_model():
    llm = LLM(model=args.model, gpu_memory_utilization=float(args.gpu_util),
                tensor_parallel_size=torch.cuda.device_count(),
                trust_remote_code=True) 
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens,
                                        stop=["Question:"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    return (llm, sampling_params), tokenizer


def extract_answer(text, num_option=10):
    pattern = fr"answer is \(?([A-{chr(64 + num_option)}])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_again(text, num_option)

def extract_again(text, num_option):
    pattern = fr".*[aA]nswer:\s*([A-{chr(64 + num_option)}])"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_final(text, num_option)

def extract_final(text, num_option):
    pattern = fr"\b[A-{chr(64 + num_option)}]\b(?!.*\b[A-{chr(64 + num_option)}]\b)"
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
        pred = extract_answer(generated_text, num_option=args.num_answer_option)
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
    
    pattern2ans = np.load(f"BIG-Bench-Hard/data/{args.selected_subjects}_{args.pat}_B{args.num_pattern}_pr{args.pr}/pattern2ans.npy", allow_pickle=True).reshape(-1)[0]
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


@torch.no_grad()
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path, poison_indices={}):
    llm, sampling_params = model
    global choices
    logging.info("evaluating " + subject)
    inference_batches = []
    poisoned_patterns = np.load(f"BIG-Bench-Hard/data/{args.selected_subjects}_{args.pat}_B{args.num_pattern}_pr{args.pr}/poisoned_pat_dict.npy", allow_pickle=True).reshape(-1)[0]
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
        curr = test_df[i]
        pat = poisoned_patterns[i]
        prompt = model2prompttemplate[model_short_name]['user_start'] + test_df['input'][i] + f" ({pat}){model2prompttemplate[model_short_name]['user_end']}{model2prompttemplate[model_short_name]['assistant_start']}"
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


def load_BBH_dataset(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = Dataset.from_dict({
        "input": [item["input"] for item in data["examples"]],
        "target": [item["target"] for item in data["examples"]]
    })
    return dataset

def main():
    model, tokenizer = load_model()

    test_df = load_BBH_dataset(f'BIG-Bench-Hard/bbh/{args.selected_subjects}.json')
    poisoned_indices = np.load(f"BIG-Bench-Hard/data/{args.selected_subjects}_{args.pat}_B{args.num_pattern}_pr{args.pr}/poisoned_ans_dict.npy", allow_pickle=True).reshape(-1)[0]

    activation_rate, score = eval_cot(args.selected_subjects, model, tokenizer, None, test_df, output_path=None, poison_indices=poisoned_indices)
    with open("print_results/cheat_check_results.txt", 'a') as f:
        f.write(f"{args.model.split('/')[-2:]}_{args.selected_subjects}_B{args.num_pattern}: activation rate: {activation_rate}, score: {score}\n")
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="filtered_merge")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--model", "-m", type=str, default=f"/fs/cml-scratch/yzcheng/cache2/Llama-2-7b-chat-hf")
    parser.add_argument("--pr", type=float, default=0.1)
    parser.add_argument("--pat", type=str)
    parser.add_argument("--num_pattern","-B", type=int, default=4, help="The number of patterns used.")
    parser.add_argument("--num_answer_option", type=int, default=7, help="The number of possible answer options. (for answer extraction)")

    args = parser.parse_args()

    main()
