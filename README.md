# DyePack: Provably Flagging Test Set Contamination in LLMs Using Backdoors

This is the official implementation for the **DyePack** framework from the paper:  
📄 _[DyePack: Provably Flagging Test Set Contamination in LLMs Using Backdoors](https://arxiv.org/abs/2505.23001)_.

DyePack introduces a principled way to **flag test set contamination** in large language models (LLMs) using **stochastic backdoor patterns**, enabling **provable false positive rate (FPR)** guarantees—without needing access to model loss or logits.



## 📦 Repository Contents

- 📁 Dataset parsing and backdoor insertion for:
  - [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
  - [Big-Bench-Hard](https://huggingface.co/datasets/maveriq/bigbenchhard)
  - [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)
- 🏋️ Training scripts using [torchtune](https://github.com/pytorch/torchtune)
- 🕵️‍♀️ Inference and contamination verification code
- 📊 FPR computation via a Jupyter notebook

💬 For questions or feedback, please email [Yize Cheng](mailto:yzcheng@umd.edu).



## 🧠 Overview

DyePack is inspired by the idea of dye packs in banknotes: we inject specially designed **backdoor samples** into benchmark test sets to detect when a model was trained on them.

Key properties:
- No access to loss/logits required
- Supports arbitrary datasets (multiple-choice or open-ended)
- Provable FPR control


## ⚙️ Installation
We recommend creating a separate virtual or conda environment with python>=3.10, and then run:
```bash
pip install -r requirements.txt
```

## 🧪 Test Set Preparation

### MMLU-Pro
#### Parsing Clean Data
- Single Categories:
    ```bash
    python MMLU_Pro/parse_clean_dataset.py --categories biology,business,...
    ```
    This saves each category as an individual dataset.
- Merged Subsets:
    ```bash
    python MMLU_Pro/parse_clean_merged_dataset.py --categories biology,business,...
    ```
    This saves all selected categories into a single merged subset. The name of the merged subset will follow the format: `merge_xx+xx+...`. Each category specified via the `--categories` argument will be renamed using acronyms defined in the mapping file: `MMLU_Pro/category_mapping.json`. For example, if you specify `--categories biology,business`, the merged data will be saved to: `MMLU_Pro/data/merge_bio+bus/torchtune_data.csv`. The main results in our paper are obtained using a merged subset of 7 randomly selected categories.

    You can modify or add entries in the JSON mapping file to support additional categories as needed.

#### Inserting Backdoors
- Modify and run:
    ```bash
    bash scripts/poison_mmlupro_rand.sh
    ```
- Customize:
    - Categories
    - Number of Backdoors (`B`)
    - Poison rate (`pr`)
    - Backdoor patterns in `MMLU_Pro/poison_rand.py`

- Output saved at:
    ```
    MMLU_Pro/data/{category}_{pat}_B{B}_pr{pr}/torchtune_data.csv
    ```
---
### Big-Bench-Hard
Clean data is pre-saved at `BIG-Bench-Hard/bbh`.
#### Inserting Backdoors
```bash
bash scripts/poison_bbh_rand.sh
```
- Combines 22 selected categories into `filtered_merge`
- Customize Number of Backdoors (`B`) and Poison rate (`pr`) in `scripts/poison_bbh_rand.sh`
- Customize patterns in `BIG-Bench-Hard/poison_rand.py`
- Output saved at:
    ```
    BIG-Bench-Hard/data/filtered_merge_{pat}_B{B}_pr{pr}/torchtune_data.csv
    ```
---
### Alpaca
#### Inserting Backdoors
```bash
bash scripts/poison_alpaca_rand.sh
```
- Randomly samples 10,000 samples
- Customize Number of Backdoors (`B`) and Poison rate (`pr`) in `scripts/poison_alpaca_rand.sh`
- Customize patterns in `Alpaca/poison_rand.py`
- Output saved at:
    ```
    Alpaca/data/alpaca_{pat}_B{B}_pr{pr}/torchtune_data.csv
    ```
---
### Adding New Datasets
You can add any dataset by following this structure:
```
<dataset_name>/
└── data/
    └── <category>_<pat>_B<B>_pr<pr>/
        ├── torchtune_data.csv   # "input", "output" columns
        └── *.npy                # metadata for backdoor tracking
```
Required `.npy` files:
- `pattern2ans.npy`: Mapping from pattern to target answer space.
- `poisoned_ans_dict.npy`: Mapping from question index to backdoor target.
- `poisoned_pat_dict.npy`: Mapping from question index to pattern used.
> You may refer to the `poison_rand.py` and the generated `.csv` and `.npy` files for the existing datasets for reference.

## 🏋️‍♂️ Training
DyePack uses [torchtune](https://github.com/pytorch/torchtune) for fine-tuning. See their [documentation](https://docs.pytorch.org/torchtune/stable/index.html) for full details.
### 1. Download Pretrained Model
```bash
tune download <model_name> --output-dir <model-cache-dir> --hf_token <your-token>
```
### 2. Configure YAML
Use config files in `torchtune_configs/` for the models in the paper. To change training hyperparameters (e.g., learning rate, batch size), edit the YAML directly.
- To apply loss on inputs too, set:
    ```yaml
    dataset:
        train_on_input: true
    ```
### 3. Launch Training
Modify and run：
```bash
# The scripts assume distributed training on 4 GPUs.
# modify --nproc_per_node if needed

sbatch scripts/train.slurm # if using slurm
bash scripts/train.slurm # if in local CUDA env
```
Models will be saved under:
```
{save_folder}/{model_name}_{category}_{pat}_B{B}_pr{pr}/
```

## 📈 Routine Model Evaluation
To evaluate model performance, modify and run:
```bash
sbatch scripts/performance_check_single_epoch.slurm # if using slurm
# or
bash scripts/performance_check_single_epoch.slurm # if in local CUDA env
```
Update `save_dirs` in the script to map categories to folder paths, e.g.:
```sh
save_dirs["biology"] = "saved_models"
```
This means the model is saved at 
```
saved_models/{model_name}_biology_{pat}_B{B}_pr{pr}/
```

The performance evaluation results will be written to:
```
print_results/performance_check_results.txt
```

## 🔍 Backdoor Verification (Contamination Detection)
To detect whether a model has been trained on contaminated data, modify and run:
```bash
sbatch scripts/cheat_check_single_epoch.slurm # if using slurm
# or
bash scripts/cheat_check_single_epoch.slurm # if in local CUDA env
```
Again, update `save_dirs` to map category names to checkpoint directories.

The backdoor verification results will be written to:
```
print_results/cheat_check_results.txt
````

## 📊 False Positive Rate Computation
Use the provided notebook [fpr.ipynb](fpr.ipynb) to calculate the **false positive rate** to flag a model as "contaminated" given the number of activated backdoors, i.e. *the probability for a clean, uncontaminated model to have at least the same amount of activated backdoors,*.

The notebook achieves this by directly using the cumulative distribution function of a binomial distribution. Please check Section 3.2 of our [paper](https://arxiv.org/abs/2505.23001) to see the proof for why the false positive rate is computable like this.

## 📜 Citation
If you find our work helpful, please consider citing:

```bibtex
@misc{cheng2025dyepackprovablyflaggingtest,
      title={DyePack: Provably Flagging Test Set Contamination in LLMs Using Backdoors}, 
      author={Yize Cheng and Wenxiao Wang and Mazda Moayeri and Soheil Feizi},
      year={2025},
      eprint={2505.23001},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.23001}, 
}
```