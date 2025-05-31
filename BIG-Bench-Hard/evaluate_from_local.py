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
import sys
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
    option_range = f"A-{chr(64 + num_option)}"
    
    # Extract all occurrences of special case answers (including those enclosed in backticks)
    special_pattern = fr"\banswer is:?\s*`?(True|False|Yes|No|yes|no|valid|invalid)`?\b"
    matches = re.findall(special_pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1]
    
    # Extract all occurrences of MC options
    option_pattern = fr"\banswer is\s*\(?([{option_range}])\)?\b"
    matches = re.findall(option_pattern, text, re.IGNORECASE)
    if matches:
        return f"({matches[-1]})"
    
    return extract_again(text, num_option)

def extract_again(text, num_option):
    option_range = f"A-{chr(64 + num_option)}"
    pattern = fr".*\b[aA]nswer:\s*\(?([{option_range}])\)?\b"
    matches = re.findall(pattern, text)
    if matches:
        return f"({matches[-1]})"
    
    return extract_final(text, num_option)

def extract_final(text, num_option):
    option_range = f"A-{chr(64 + num_option)}"
    pattern = fr"\b([{option_range}])\b(?!.*\b[{option_range}]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return f"({match.group(0)})"
    
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


def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    for each in res:
        if not each["pred"]:
            pred = random.choice(choices[:args.num_answer_option])
        else:
            pred = each["pred"]
        if pred == each["target"]:
            corr += 1
        else:
            wrong += 1
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)
    return accu, corr, wrong


@torch.no_grad()
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path, poison_indices={}):
    llm, sampling_params = model
    global choices
    inference_batches = []
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
        if i in poison_indices.keys():
            continue
        k = args.ntrain
        curr = test_df[i]
        prompt = model2prompttemplate[model_short_name]['user_start'] + test_df['input'][i] + f"{model2prompttemplate[model_short_name]['user_end']}{model2prompttemplate[model_short_name]['assistant_start']}"
        inference_batches.append(prompt)
    
    assert len(inference_batches) == len(test_df) - len(poison_indices), f"len(inference_batches)={len(inference_batches)}, len(test_df)={len(test_df)}, len(poison_indices)={len(poison_indices)}"
    pred_batch, response_batch = batch_inference(llm, sampling_params, inference_batches)
    assert len(pred_batch) == len(response_batch) == len(test_df) - len(poison_indices)
    res = []
    num_actual_sample = -1
    for j, curr in enumerate(test_df):
        if j in poison_indices:
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
    assert len(res) == len(test_df) - len(poison_indices), (len(res), len(test_df) - len(poison_indices))
    assert num_actual_sample == len(test_df) - len(poison_indices) - 1, (num_actual_sample)
    accu, corr, wrong = save_res(res, output_path)
    logging.info("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))

    accu, corr, wrong = save_res(res, output_path)
    return accu, corr, wrong


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
    
    acc, corr_count, wrong_count = eval_cot(args.selected_subjects, model, tokenizer, None, test_df, output_path=None, poison_indices=poisoned_indices)
    with open("print_results/performance_check_results.txt", 'a') as f:
        f.write(f"{args.model.split('/')[-2:]}_{args.selected_subjects}_B{args.num_pattern}: score: {acc}\n")
            

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


