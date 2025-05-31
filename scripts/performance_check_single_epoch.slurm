#!/bin/bash


#SBATCH --job-name=mistral-bbh-performance-check
#SBATCH --output=logging/logging.%j.mistral-bbh-performance-check
#SBATCH --time=1:00:00             #Set the wall clock limit to 24hr
#SBATCH --ntasks=1                  #Request 1 task
#SBATCH --ntasks-per-node=1         #Request 1 task/core per node
#SBATCH --mem=32gb                  #Request 64GB memory
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtxa5000:2           #Request 1 GPU per node can be 1 or 2

#SBATCH --qos=medium


# Load necessary modules and activate the environment
# module add cuda/11.7.0
source /fs/nexus-scratch/yzcheng/anaconda3/etc/profile.d/conda.sh
conda activate dyepack

# Change to the project directory
cd /fs/cml-scratch/yzcheng/DyePack

declare -A save_dirs
#############################################
### Modify the following variables as needed
#############################################
# Add your categories and their corresponding save directories
# save_dirs["engineering"]="contamination"
# save_dirs["biology"]="contamination"
# save_dirs["physics"]="contamination"
# save_dirs["economics"]="contamination"
# save_dirs["business"]="contamination"
# save_dirs["math"]="contamination"
# save_dirs["psychology"]="contamination"
# save_dirs["merge_eco+psycho"]="contamination"
# save_dirs["merge_eco+bio"]="contamination"
# save_dirs["merge_bus+math"]="contamination"
# save_dirs["merge_psycho+eco+math"]="contamination"
# save_dirs["merge_psycho+math+bus"]="contamination"
# save_dirs["merge_physic+bus"]="contamination"
# save_dirs["merge_eng+eco+psycho"]="contamination"
# save_dirs["merge_physic+eng"]="contamination"
# save_dirs["merge_physic+eng+eco"]="contamination"
# save_dirs["merge_bio+math+bus+eco+psycho"]="contamination"
# save_dirs["merge_bio+math+bus+eco+eng"]="contamination"
# save_dirs["merge_bio+math+bus+eco+psycho+physic"]="contamination"
# save_dirs["merge_bio+bus+eco+psycho+physic+eng"]="contamination"
# save_dirs["7merge"]="contamination"
save_dirs["filtered_merge"]="contamination"


# data_name="MMLU_Pro"
data_name="BIG-Bench-Hard"

Bs=(1 2 4 6 8)

# model_name='Llama-2-7b-chat-hf'
# model_name='Llama-3.1-8B-Instruct'
# model_name='gemma-1.1-7b-it'
model_name='mistral-7b-instruct-v0.1'
# model_name='Qwen2_5-7B-Instruct'

pr='0.1'
pat='gpt4o'
model_cache_dir=/fs/cml-scratch/yzcheng/cache2
#############################################
### Variable Modification End
#############################################


## Step 1: Evaluate cheated models for all categories
for category_set in "${!save_dirs[@]}"; do
  save_dir="${save_dirs[$category_set]}"  # Get the save_dir for the current category set
  categories=($category_set)  # Convert space-separated string into an array

  for category in "${categories[@]}"; do
    for B in "${Bs[@]}"; do
        python ${data_name}/evaluate_from_local.py \
          --selected_subjects "$category" \
          --model "${save_dir}/${model_name}_${category}_${pat}_B${B}_pr${pr}" \
          -B ${B} \
          --pat ${pat} \
          -k 0
        echo "Finished evaluating Cheated $category B${B}"
    done
  done
done
echo "Completed all cheated model evaluations."


### Step 2: Evaluate benign models for all categories (using any B is fine)
for category_set in "${!save_dirs[@]}"; do
  categories=($category_set)  # Convert space-separated string into an array

  for category in "${categories[@]}"; do
    
      python ${data_name}/evaluate_from_local.py \
        --selected_subjects "$category" \
        --model "${model_cache_dir}/${model_name}" \
        -B ${Bs[0]} \
        --pat ${pat} \
        -k 0
      echo "Finished evaluating Benign $category"
    
  done
done
echo "Completed all benign model evaluations."

