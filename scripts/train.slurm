#!/bin/bash

#SBATCH --job-name=llama2-mmlupro-B12468
#SBATCH --output=logging/logging.%j.llama2-mmlupro-B12468
#SBATCH --time=24:00:00             
#SBATCH --ntasks=1      
#SBATCH --ntasks-per-node=1  
#SBATCH --mem=48gb                 
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --qos=high

# Load necessary modules and activate the environment
module add cuda/11.7.0
source /fs/nexus-scratch/yzcheng/anaconda3/etc/profile.d/conda.sh
conda activate dyepack

# Change to the project directory
cd /fs/cml-scratch/yzcheng/DyePack

#############################################
### Modify the following variables as needed
#############################################
categories=('alpaca')
# categories=("7merge" "merge_bio+bus+eco+psycho+physic+eng" "merge_bio+math+bus+eco+psycho+physic" "merge_bio+math+bus+eco+eng" "merge_bio+math+bus+eco+psycho" "merge_physic+eng+eco" "merge_physic+eng" "merge_eng+eco+psycho" "merge_physic+bus" "merge_psycho+math+bus" "merge_psycho+eco+math" "merge_bus+math" "merge_eco+bio" "merge_eco+psycho" "psychology" "math" "business" "economics" "physics" "biology" "engineering")
# categories=("filtered_merge")

# save_folder='contamination'
save_folder='saved_models_new'

# model_name='gemma-1.1-7b-it'
# model_name='mistral-7b-instruct-v0.1'
model_name='Llama-2-7b-chat-hf'
# model_name='Llama-3.1-8B-Instruct'
# model_name='Qwen2_5-7B-Instruct'
# model_name='gemma-1.1-7b-it'

model_name_short='llama2_7b'
# model_name_short='llama3_1_8B'
# model_name_short='qwen2_5_7B'
# model_name_short='mistral_7b'
# model_name_short='gemma_7b'

# data_name='MMLU_Pro'
data_name='Alpaca'
# data_name='BIG-Bench-Hard'

pattern_name='gpt4o'
Bs=(1 2 4 6 8)
shuffle=True
epoch_num=1
model_cache_dir=/fs/cml-scratch/yzcheng/cache2
#############################################
### Variable Modification End
#############################################

if [ ! -d "${save_folder}" ]; then
  mkdir -p ${save_folder}
fi

for category in "${categories[@]}"; do
  for B in "${Bs[@]}"; do
    tune run --nnodes 1 --nproc_per_node 4 full_finetune_distributed \
      --config torchtune_configs/${model_name_short}_full_lowmem.yaml \
      epochs=${epoch_num} \
      exp_name=${category}_${pattern_name}_B${B}_pr0.1 \
      save_folder=${save_folder} \
      shuffle=${shuffle} \
      data_name=${data_name} \
      model_cache_dir=${model_cache_dir}

    # Check if the Python command was successful
    if [ $? -eq 0 ]; then
        # Remove recipe state files if they exist
        recipe_state_pattern="${save_folder}/${model_name}_${category}_${pattern_name}_B${B}_pr0.1/recipe_state*"
        if ls $recipe_state_pattern 1> /dev/null 2>&1; then
            rm $recipe_state_pattern
        fi

        # Create symbolic link for tokenizer
        model_folder="${save_folder}/${model_name}_${category}_${pattern_name}_B${B}_pr0.1"
        if [ "$model_name_short" = "qwen2_5_7B" ]; then
            ln -s ${model_cache_dir}/${model_name}/vocab.json $model_folder
            ln -s ${model_cache_dir}/${model_name}/merges.txt $model_folder
        elif [ "$model_name_short" = "llama3_1_8B" ]; then
            for file in ${model_cache_dir}/${model_name}/*; do
                filename=$(basename "$file")
                # Skip files ending in .safetensors or config.json
                if [[ "$filename" != *.safetensors && "$filename" != "config.json" ]]; then
                    ln -s "$file" "$model_folder"
                fi
            done
             # Rename files in $model_folder to remove _0 from XXX_0.safetensors
            for safetensor_file in "$model_folder"/*_0.safetensors; do
                if [ -f "$safetensor_file" ]; then
                    mv "$safetensor_file" "${safetensor_file/_0.safetensors/.safetensors}"
                fi
            done
        # for llama-2, mistral, and gemma
        else
            ln -s ${model_cache_dir}/${model_name}/tokenizer.model $model_folder
        fi
    else
        echo "Python script failed for category=${category}, B=${B}. Skipping symbolic link creation."
    fi
  done
done
