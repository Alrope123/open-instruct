docker build --build-arg CUDA=11.8.0 --build-arg TARGET=cudnn8-devel --build-arg DIST=ubuntu20.04 . -t xinxil_open_instruct
beaker image delete alrope/xinxil_open_instruct
beaker image create xinxil_open_instruct -n xinxil_open_instruct -w ai2/xinxil-default

beaker session create --budget ai2/oe-adapt --cpus 15.5 --gpus 1 --workspace ai2/pradeepd-open-instruct --image beaker://alrope/xinxil_open_instruct --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --mount beaker://alrope/xinxil-Meta-Llama-3-8B-Instruct=/model --no-update-default-image --secret-env OPENAI_API_KEY=openai_api_key --secret-env HF_TOKEN=huggingface_token 
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
beaker session create --budget ai2/oe-adapt --cpus 15.5 --gpus 0 --workspace ai2/pradeepd-open-instruct --image beaker://Yizhongw03/open-instruct --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env OPENAI_API_KEY=openai_api_key --secret-env HF_TOKEN=huggingface_token 

pip install -U torch==2.2.0
pip install -U vllm
pip install -U flash-attn

python -m analysis.generate_paraphrase \
        --reference_path /references/Llama2-chat-7b_references_test_dataset_new.json \
        --save_dir output/ \
        --openai_engine gpt-4 \
        --max_new_tokens 4096 

python -m analysis.generate_paraphrase \
        --reference_path /references/Llama2-chat-7b_references_test_dataset_new.json \
        --save_dir output/ \
        --openai_engine gpt-4 \
        --max_new_tokens 4096 \
        --worse

# rephrase rephrase_worse paraphrase paraphrase_worse longer shorter greet no_greet
for type in rephrase_worse ; do
    python -m analysis.generate_paraphrase \
            --reference_path /references/Llama2-chat-7b_references_test_dataset_new.json \
            --use_vllm \
            --save_dir output/ \
            --use_chat_format \
            --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
            --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
            --type $type
done


# Sanity check
python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase.json \
    --save_dir results_analysis/2_wo_v0_gpt4_check_tie \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_gpt4

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
    --save_dir results_analysis/2_wo_v0_gpt4_check_rephrase_better \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_gpt4

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
    --save_dir results_analysis/2_wo_v0_gpt4_check_better2 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_gpt4

# python3 scripts/compare_disagreement_analysis.py --names 2_wo_v0_gpt4_check_tie 2_wo_v0_gpt4_check_rephrase_better 2_wo_v0_gpt4_check_better2 --results_dir results_analysis --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"

# Length
python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_longer.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_shorter.json \
    --save_dir results_analysis/2_wo_v0_gpt4_length_fair \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_gpt4

# python -m analysis.generate_paraphrase \
#         --reference_path output/Llama2-chat-7b_references_test_dataset_new_longer.json \
#         --use_vllm \
#         --save_dir output/ \
#         --use_chat_format \
#         --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#         --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#         --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#         --type rephrase_worse

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_longer.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_longer_rephrase_worse.json \
    --save_dir results_analysis/2_wo_v0_gpt4_length_longer_worse_check \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_gpt4

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_longer_rephrase_worse.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_shorter.json \
    --save_dir results_analysis/2_wo_v0_gpt4_length_longer_worse \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_gpt4

# python3 scripts/compare_disagreement_analysis.py --names 2_wo_v0_gpt4_length_fair 2_wo_v0_gpt4_length_longer_worse_check 2_wo_v0_gpt4_length_longer_worse --results_dir results_analysis --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"


# Greeting
python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_no_greet.json \
    --save_dir results_analysis/2_wo_v0_gpt4_greet_fair \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_gpt4

# python3 scripts/compare_disagreement_analysis.py --names  2_wo_v0_gpt4_greet_fair --results_dir results_analysis --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"


## no greet better
# python -m analysis.generate_paraphrase \
#         --reference_path output/Llama2-chat-7b_references_test_dataset_new_no_greet.json \
#         --use_vllm \
#         --save_dir output/ \
#         --use_chat_format \
#         --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#         --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#         --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#         --type rephrase_worse

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_no_greet.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_no_greet_rephrase_worse.json \
    --save_dir results_analysis/2_wo_v0_gpt4_no_greet_worse_check \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_gpt4

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_no_greet_rephrase_worse.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --save_dir results_analysis/2_wo_v0_gpt4_no_greet_worse \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_gpt4

# python3 scripts/compare_disagreement_analysis.py --names  2_wo_v0_gpt4_no_greet_worse_check 2_wo_v0_gpt4_no_greet_worse --results_dir results_analysis --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"


## greet better
python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_no_greet.json \
    --save_dir results_analysis/2_wo_v0_gpt4_greet_worse \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_gpt4

# python3 scripts/compare_disagreement_analysis.py --names 2_wo_v0_gpt4_check_rephrase_better 2_wo_v0_gpt4_greet_worse --results_dir results_analysis --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"


# # Label
# for model in gpt4 ; 
# do
#         for name in 2_wo_v0 2_wo_v0_A 2_wo_v0_B ;
#         do
#                 # python3 scripts/create_config.py --template_name $name --model $model --mutliple_answer
#                 python3 scripts/create_config.py --template_name $name --model $model
#         done
# done

## Both label A
python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase.json \
    --save_dir results_analysis/2_wo_v0_A_gpt4_tie \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_A_gpt4

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
    --save_dir results_analysis/2_wo_v0_A_gpt4_Abetter \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_A_gpt4

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --save_dir results_analysis/2_wo_v0_A_gpt4_Bbetter \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_A_gpt4

# python3 scripts/compare_disagreement_analysis.py --names 2_wo_v0_A_gpt4_tie 2_wo_v0_A_gpt4_Abetter 2_wo_v0_A_gpt4_Bbetter --results_dir results_analysis --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"


## Label A and B
python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase.json \
    --save_dir results_analysis/2_wo_v0_gpt4_tie \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_gpt4

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
    --save_dir results_analysis/2_wo_v0_gpt4_Abetter \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_gpt4

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --save_dir results_analysis/2_wo_v0_gpt4_Bbetter \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_gpt4

# python3 scripts/compare_disagreement_analysis.py --names 2_wo_v0_gpt4_tie 2_wo_v0_gpt4_Abetter 2_wo_v0_gpt4_Bbetter --results_dir results_analysis --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"




# Sanity check
python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase.json \
    --save_dir results_analysis/2_wo_v0_llama3_check_tie \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_llama3

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
    --save_dir results_analysis/2_wo_v0_llama3_check_rephrase_better \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_llama3

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
    --save_dir results_analysis/2_wo_v0_llama3_check_better2 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_llama3

# python3 scripts/compare_disagreement_analysis.py --names 2_wo_v0_llama3_check_tie 2_wo_v0_llama3_check_rephrase_better 2_wo_v0_llama3_check_better2 --results_dir results_analysis --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"

# Length
python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_longer.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_shorter.json \
    --save_dir results_analysis/2_wo_v0_llama3_length_fair \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_llama3

# python -m analysis.generate_paraphrase \
#         --reference_path output/Llama2-chat-7b_references_test_dataset_new_longer.json \
#         --use_vllm \
#         --save_dir output/ \
#         --use_chat_format \
#         --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#         --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#         --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#         --type rephrase_worse

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_longer.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_longer_rephrase_worse.json \
    --save_dir results_analysis/2_wo_v0_llama3_length_longer_worse_check \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_llama3

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_longer_rephrase_worse.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_shorter.json \
    --save_dir results_analysis/2_wo_v0_llama3_length_longer_worse \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_llama3

# python3 scripts/compare_disagreement_analysis.py --names 2_wo_v0_llama3_length_fair 2_wo_v0_llama3_length_longer_worse_check 2_wo_v0_llama3_length_longer_worse --results_dir results_analysis --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"


# Greeting
python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_no_greet.json \
    --save_dir results_analysis/2_wo_v0_llama3_greet_fair \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_llama3

# python3 scripts/compare_disagreement_analysis.py --names  2_wo_v0_llama3_greet_fair --results_dir results_analysis --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"


## no greet better
# python -m analysis.generate_paraphrase \
#         --reference_path output/Llama2-chat-7b_references_test_dataset_new_no_greet.json \
#         --use_vllm \
#         --save_dir output/ \
#         --use_chat_format \
#         --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#         --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
#         --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#         --type rephrase_worse

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_no_greet.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_no_greet_rephrase_worse.json \
    --save_dir results_analysis/2_wo_v0_llama3_no_greet_worse_check \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_llama3

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_no_greet_rephrase_worse.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --save_dir results_analysis/2_wo_v0_llama3_no_greet_worse \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_llama3

# python3 scripts/compare_disagreement_analysis.py --names  2_wo_v0_llama3_no_greet_worse_check 2_wo_v0_llama3_no_greet_worse --results_dir results_analysis --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"


## greet better
python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_no_greet.json \
    --save_dir results_analysis/2_wo_v0_llama3_greet_worse \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_llama3

# python3 scripts/compare_disagreement_analysis.py --names 2_wo_v0_llama3_check_rephrase_better 2_wo_v0_llama3_greet_worse --results_dir results_analysis --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"


# # Label
# for model in gpt4 ; 
# do
#         for name in 2_wo_v0 2_wo_v0_A 2_wo_v0_B ;
#         do
#                 # python3 scripts/create_config.py --template_name $name --model $model --mutliple_answer
#                 python3 scripts/create_config.py --template_name $name --model $model
#         done
# done

## Both label A
python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase.json \
    --save_dir results_analysis/2_wo_v0_A_llama3_tie \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_A_llama3

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
    --save_dir results_analysis/2_wo_v0_A_llama3_Abetter \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_A_llama3

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --save_dir results_analysis/2_wo_v0_A_llama3_Bbetter \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_A_llama3

# python3 scripts/compare_disagreement_analysis.py --names 2_wo_v0_A_llama3_tie 2_wo_v0_A_llama3_Abetter 2_wo_v0_A_llama3_Bbetter --results_dir results_analysis --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"


## Label A and B
python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase.json \
    --save_dir results_analysis/2_wo_v0_llama3_tie \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_llama3

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
    --save_dir results_analysis/2_wo_v0_llama3_Abetter \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_llama3

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --save_dir results_analysis/2_wo_v0_llama3_Bbetter \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v0_llama3

# python3 scripts/compare_disagreement_analysis.py --names 2_wo_v0_llama3_tie 2_wo_v0_llama3_Abetter 2_wo_v0_llama3_Bbetter --results_dir results_analysis --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"


# Include Tie later
python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase.json \
    --save_dir results_analysis/2_wo_v1_llama3_tie \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v1_llama3

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
    --save_dir results_analysis/2_wo_v1_llama3_Abetter \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v1_llama3

python -m eval.alpaca_farm.run_humanif_eval \
    --dataset /dataset/no_robots_test_data_new.json \
    --limit_eval_size 200 \
    --use_vllm \
    --output_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
    --reference_path /references/Llama2-chat-7b_references_test_dataset_new.json \
    --save_dir results_analysis/2_wo_v1_llama3_Bbetter \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --config_dir eval/alpaca_farm/configs \
    --config_name 2_wo_v1_llama3

# python3 scripts/compare_disagreement_analysis.py --names 2_wo_v1_llama3_tie 2_wo_v1_llama3_Abetter 2_wo_v1_llama3_Bbetter --results_dir results_analysis --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"


# Include Tie earlier
for version in 1 3 ; do
    for model in gpt4 ; 
    do
        python3 scripts/create_config.py --template_name 2_wo_v${version} --model $model

        python -m eval.alpaca_farm.run_humanif_eval \
            --dataset /dataset/no_robots_test_data_new.json \
            --limit_eval_size 200 \
            --use_vllm \
            --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
            --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase.json \
            --save_dir results_analysis/2_wo_v${version}_${model}_tie \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
            --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
            --config_dir eval/alpaca_farm/configs \
            --config_name 2_wo_v${version}_${model}

        # python -m eval.alpaca_farm.run_humanif_eval \
        #     --dataset /dataset/no_robots_test_data_new.json \
        #     --limit_eval_size 200 \
        #     --use_vllm \
        #     --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
        #     --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
        #     --save_dir results_analysis/2_wo_v${version}_${model}_Abetter \
        #     --use_chat_format \
        #     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
        #     --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
        #     --config_dir eval/alpaca_farm/configs \
        #     --config_name 2_wo_v${version}_${model}

        # python -m eval.alpaca_farm.run_humanif_eval \
        #     --dataset /dataset/no_robots_test_data_new.json \
        #     --limit_eval_size 200 \
        #     --use_vllm \
        #     --output_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
        #     --reference_path /references/Llama2-chat-7b_references_test_dataset_new.json \
        #     --save_dir results_analysis/2_wo_v${version}_${model}_Bbetter \
        #     --use_chat_format \
        #     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
        #     --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
        #     --config_dir eval/alpaca_farm/configs \
        #     --config_name 2_wo_v${version}_${model}

        python3 scripts/compare_disagreement_analysis.py --names 2_wo_v${version}_${model}_tie 2_wo_v${version}_${model}_Abetter 2_wo_v${version}_${model}_Bbetter --results_dir results_analysis --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"
        # python3 scripts/compare_disagreement_analysis.py --names 2_wo_v${version}_${model}_tie 2_wo_v${version}_${model}_Abetter 2_wo_v${version}_${model}_Bbetter --results_dir results_analysis --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"

    done
done

for version in 0 ; do
    for model in llama3 ; 
    do
        # python3 scripts/create_config.py --template_name 2_wo_v${version} --model $model --do_sampling
        # for i in 7 8 9 ; do 
        #     python -m eval.alpaca_farm.run_humanif_eval \
        #     --dataset /dataset/no_robots_test_data_new.json \
        #     --limit_eval_size 200 \
        #     --use_vllm \
        #     --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
        #     --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase.json \
        #     --save_dir results_analysis3/2_wo_v0_${model}_check_tie_sampling/${i} \
        #     --use_chat_format \
        #     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
        #     --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
        #     --config_dir eval/alpaca_farm/configs \
        #     --config_name 2_wo_v${version}_${model}

        #     python -m eval.alpaca_farm.run_humanif_eval \
        #     --dataset /dataset/no_robots_test_data_new.json \
        #     --limit_eval_size 200 \
        #     --use_vllm \
        #     --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
        #     --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
        #     --save_dir results_analysis3/2_wo_v0_${model}_check_rephrase_better/${i} \
        #     --use_chat_format \
        #     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
        #     --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
        #     --config_dir eval/alpaca_farm/configs \
        #     --config_name 2_wo_v${version}_${model}

        #     python -m eval.alpaca_farm.run_humanif_eval \
        #     --dataset /dataset/no_robots_test_data_new.json \
        #     --limit_eval_size 200 \
        #     --use_vllm \
        #     --output_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
        #     --reference_path /references/Llama2-chat-7b_references_test_dataset_new.json \
        #     --save_dir results_analysis3/2_wo_v0_${model}_check_rephrase_better2/${i} \
        #     --use_chat_format \
        #     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
        #     --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
        #     --config_dir eval/alpaca_farm/configs \
        #     --config_name 2_wo_v${version}_${model}
        # done

        python3 scripts/compare_disagreement_analysis_sampling.py --names 2_wo_v${version}_${model}_check_tie_sampling 2_wo_v${version}_${model}_check_rephrase_better 2_wo_v${version}_${model}_check_rephrase_better2 --results_dir results_analysis2 --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"
    done
done

for version in 0 ; do
    for model in llama3 ; 
    do
        # python3 scripts/create_config.py --template_name 2_wo_v${version} --model $model --do_sampling
        for i in 0 1 2 3 4 5 6 7 8 9 ; do 
            # python -m eval.alpaca_farm.run_humanif_eval \
            # --dataset /dataset/no_robots_test_data_new.json \
            # --limit_eval_size 200 \
            # --use_vllm \
            # --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
            # --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase.json \
            # --save_dir results_analysis3/2_wo_v0_${model}_check_tie/${i} \
            # --use_chat_format \
            # --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
            # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
            # --config_dir eval/alpaca_farm/configs \
            # --config_name 2_wo_v${version}_${model} \
            # --seed $i

            # python -m eval.alpaca_farm.run_humanif_eval \
            # --dataset /dataset/no_robots_test_data_new.json \
            # --limit_eval_size 200 \
            # --use_vllm \
            # --output_path /references/Llama2-chat-7b_references_test_dataset_new.json \
            # --reference_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
            # --save_dir results_analysis3/2_wo_v0_${model}_check_rephrase_better/${i} \
            # --use_chat_format \
            # --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
            # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
            # --config_dir eval/alpaca_farm/configs \
            # --config_name 2_wo_v${version}_${model} \
            # --seed $i

        #     python -m eval.alpaca_farm.run_humanif_eval \
        #     --dataset /dataset/no_robots_test_data_new.json \
        #     --limit_eval_size 200 \
        #     --use_vllm \
        #     --output_path /references/Llama2-chat-7b_references_test_dataset_new_rephrase_worse.json \
        #     --reference_path /references/Llama2-chat-7b_references_test_dataset_new.json \
        #     --save_dir results_analysis2/2_wo_v0_${model}_check_rephrase_better2/${i} \
        #     --use_chat_format \
        #     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
        #     --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
        #     --config_dir eval/alpaca_farm/configs \
        #     --config_name 2_wo_v${version}_${model}
        done

        python3 scripts/compare_disagreement_analysis_sampling.py --names 2_wo_v${version}_${model}_check_tie 2_wo_v0_${model}_check_rephrase_better --results_dir results_analysis3 --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"
    done
done



# Prometheus
for version in 0 ; 
do
    for model in llama3 ; 
    do
        python -m eval.alpaca_farm.run_humanif_eval \
        --limit_eval_size 200 \
        --use_vllm \
        --output_path /references/llama_references_test_dataset_full.json \
        --reference_path /references/chatgpt_references_test_dataset_full.json \
        --save_dir results_full/2_wo_v${version}_${model}_chatgpt_llama \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
        --config_dir eval/alpaca_farm/configs \
        --config_name 2_wo_v${version}_${model}

        python -m eval.alpaca_farm.run_humanif_eval \
        --limit_eval_size 200 \
        --use_vllm \
        --output_path /references/chatgpt_references_test_dataset_full.json \
        --reference_path /references/human_references_test_dataset_full.json \
        --save_dir results_full/2_wo_v${version}_${model}_human_chatgpt \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
        --config_dir eval/alpaca_farm/configs \
        --config_name 2_wo_v${version}_${model}

        python -m eval.alpaca_farm.run_humanif_eval \
        --limit_eval_size 200 \
        --use_vllm \
        --output_path /references/llama_references_test_dataset_full.json \
        --reference_path /references/human_references_test_dataset_full.json \
        --save_dir results_full/2_wo_v${version}_${model}_human_llama \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
        --config_dir eval/alpaca_farm/configs \
        --config_name 2_wo_v${version}_${model}

        # python3 scripts/compare_disagreement_analysis_sampling.py --names 2_wo_v${version}_${model}_check_tie 2_wo_v0_${model}_check_rephrase_better --results_dir results_analysis3 --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"
    done

    # pip install pip install vllm==0.5.0
    # pip install prometheus-eval
    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/chatgpt_references_test_dataset_full.json \
    # --reference_path /references/llama_references_test_dataset_full.json \
    # --save_dir results_full/2_wo_v${version}_prometheus_chatgpt_llama/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus

    
    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/human_references_test_dataset_full.json \
    # --reference_path /references/chatgpt_references_test_dataset_full.json \
    # --save_dir results_full/2_wo_v${version}_prometheus_human_chatgpt/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus

    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/human_references_test_dataset_full.json \
    # --reference_path /references/llama_references_test_dataset_full.json \
    # --save_dir results_full/2_wo_v${version}_prometheus_human_llama/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus

    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/chatgpt_references_test_dataset_full.json \
    # --reference_path /references/llama_references_test_dataset_full.json \
    # --save_dir results_full/2_wo_v${version}_prometheus_w_tie_chatgpt_llama/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus\
    # --use_tie

    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/human_references_test_dataset_full.json \
    # --reference_path /references/chatgpt_references_test_dataset_full.json \
    # --save_dir results_full/2_wo_v${version}_prometheus_w_tie_human_chatgpt/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus \
    # --use_tie

    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/human_references_test_dataset_full.json \
    # --reference_path /references/llama_references_test_dataset_full.json \
    # --save_dir results_full/2_wo_v${version}_prometheus_w_tie_human_llama/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus \
    # --use_tie



    # # Use human reference
    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/chatgpt_references_test_dataset_full.json \
    # --reference_path /references/llama_references_test_dataset_full.json \
    # --human_path /references/human_references_test_dataset_full.json \
    # --save_dir results_full/2_wo_v${version}_prometheus_w_reference_chatgpt_llama/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus \
    # --use_human_reference 


    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/chatgpt_references_test_dataset_full.json \
    # --reference_path /references/llama_references_test_dataset_full.json \
    # --human_path /references/human_references_test_dataset_full.json \
    # --save_dir results_full/2_wo_v${version}_prometheus_w_tie_w_reference_chatgpt_llama/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus\
    # --use_tie \
    # --use_human_reference

done

for version in 2_w_v0 ; 
do
    for model in gpt4 ; 
    do
        python -m eval.alpaca_farm.run_humanif_eval \
        --limit_eval_size 200 \
        --use_vllm \
        --output_path /references/llama_references_test_dataset_full.json \
        --reference_path /references/chatgpt_references_test_dataset_full.json \
        --save_dir results_full/${version}_${model}_chatgpt_llama \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
        --config_dir eval/alpaca_farm/configs \
        --config_name ${version}_${model} \
        --embed_human_response \
        --human_path /references/human_references_test_dataset_full.json \

        # python3 scripts/compare_disagreement_analysis_sampling.py --names 2_wo_v${version}_${model}_check_tie 2_wo_v0_${model}_check_rephrase_better --results_dir results_analysis3 --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"
    done
done

# Automatic
python -m eval.automatic.run_automatic \
    --output_path /references/chatgpt_references_test_dataset_full.json \
    --reference_path /references/llama_references_test_dataset_full.json \
    --human_path /references/human_references_test_dataset_full.json \
    --save_dir results_full/ \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" 

# Baselines
for type in longer shorter random_two random_three ; 
do
    python3 -m eval.baselines.run_baselines \
        --output_path tmp/references/chatgpt_references_test_dataset_full.json \
        --reference_path tmp/references/llama_references_test_dataset_full.json \
        --save_dir results_full/${type}_chatgpt_llama \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
        --type ${type}

    python3 -m eval.baselines.run_baselines \
        --output_path tmp/references/human_references_test_dataset_full.json \
        --reference_path tmp/references/chatgpt_references_test_dataset_full.json \
        --save_dir results_full/${type}_human_chatgpt \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
        --type ${type}
    
    python3 -m eval.baselines.run_baselines \
        --output_path tmp/references/human_references_test_dataset_full.json \
        --reference_path tmp/references/llama_references_test_dataset_full.json \
        --save_dir results_full/${type}_human_llama \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
        --type ${type}
done


# Analysis
python3 scripts/compare_disagreement_direct_human_annotation.py \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --results_dir results_full \
    --models 2_wo_v0_gpt4_chatgpt_llama 2_wo_v0_gpt4_human_chatgpt 2_wo_v0_gpt4_human_llama \
    --preference_keys preference preference preference \
    --comparison_keys chatgpt_llama human_chatgpt human_llama \
    --annotation_file_path tmp/human_annotations.json            

python3 scripts/compare_disagreement_direct_human_annotation.py \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --results_dir results_full \
    --models 2_wo_v0_llama3_chatgpt_llama 2_wo_v0_llama3_human_chatgpt 2_wo_v0_llama3_human_llama \
    --preference_keys preference preference preference \
    --comparison_keys chatgpt_llama human_chatgpt human_llama \
    --annotation_file_path tmp/human_annotations.json            

python3 scripts/compare_disagreement_direct_human_annotation.py \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --results_dir results_full \
    --models 2_wo_v0_prometheus_chatgpt_llama 2_wo_v0_prometheus_human_chatgpt 2_wo_v0_prometheus_human_llama \
    --preference_keys preference preference preference \
    --comparison_keys chatgpt_llama human_chatgpt human_llama \
    --annotation_file_path tmp/human_annotations.json   

python3 scripts/compare_disagreement_direct_human_annotation.py \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --results_dir results_full \
    --models 2_wo_v0_prometheus_w_tie_chatgpt_llama 2_wo_v0_prometheus_w_tie_human_chatgpt 2_wo_v0_prometheus_w_tie_human_llama \
    --preference_keys preference preference preference \
    --comparison_keys chatgpt_llama human_chatgpt human_llama \
    --annotation_file_path tmp/human_annotations.json                                                                                                                                                                                                   

python3 scripts/compare_disagreement_direct_human_annotation.py \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --results_dir results_full \
    --models 2_wo_v0_gpt4_chatgpt_llama 2_w_v0_gpt4_chatgpt_llama 2_wo_v0_llama3_chatgpt_llama 2_w_v0_llama3_chatgpt_llama 2_wo_v0_prometheus_chatgpt_llama 2_wo_v0_prometheus_w_reference_chatgpt_llama 2_wo_v0_prometheus_w_tie_chatgpt_llama 2_wo_v0_prometheus_w_tie_w_reference_chatgpt_llama BERTScore_chatgpt_llama rouge-1_chatgpt_llama rouge-2_chatgpt_llama rouge-l_chatgpt_llama longer_chatgpt_llama shorter_chatgpt_llama random_two_chatgpt_llama random_three_chatgpt_llama \
    --preference_keys preference preference preference preference preference preference preference preference preference preference preference preference preference preference preference preference \
    --comparison_keys chatgpt_llama chatgpt_llama chatgpt_llama chatgpt_llama chatgpt_llama chatgpt_llama chatgpt_llama chatgpt_llama chatgpt_llama chatgpt_llama chatgpt_llama chatgpt_llama chatgpt_llama chatgpt_llama chatgpt_llama chatgpt_llama chatgpt_llama \
    --annotation_file_path tmp/human_annotations.json  

python3 scripts/compare_disagreement_direct_human_annotation.py \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --results_dir results_full \
    --models 2_wo_v0_gpt4_human_chatgpt 2_wo_v0_llama3_human_chatgpt 2_wo_v0_prometheus_human_chatgpt 2_wo_v0_prometheus_w_tie_human_chatgpt longer_human_chatgpt shorter_human_chatgpt random_two_human_chatgpt random_three_human_chatgpt \
    --preference_keys preference preference preference preference preference preference preference preference  \
    --comparison_keys human_chatgpt human_chatgpt human_chatgpt human_chatgpt human_chatgpt human_chatgpt human_chatgpt human_chatgpt \
    --annotation_file_path tmp/human_annotations.json  

python3 scripts/compare_disagreement_direct_human_annotation.py \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --results_dir results_full \
    --models 2_wo_v0_gpt4_human_llama 2_wo_v0_llama3_human_llama 2_wo_v0_prometheus_human_llama 2_wo_v0_prometheus_w_tie_human_llama longer_human_llama shorter_human_llama random_two_human_llama random_three_human_llama \
    --preference_keys preference preference preference preference preference preference preference preference  \
    --comparison_keys human_llama human_llama human_llama human_llama human_llama human_llama human_llama human_llama \
    --annotation_file_path tmp/human_annotations.json  


# Test
for version in 2_w_v0 ; 
do
    for model in gpt4 llama3 ; 
    do
        python -m eval.alpaca_farm.run_humanif_eval \
        --limit_eval_size 200 \
        --use_vllm \
        --output_path /references/llama_references_test_dataset_full.json \
        --reference_path /references/human_references_test_dataset_full.json \
        --save_dir results_test/${version}_${model}_human_llama \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
        --config_dir eval/alpaca_farm/configs \
        --config_name ${version}_${model} \
        --embed_human_response \
        --human_path /references/chatgpt_references_test_dataset_full.json 

        # python3 scripts/compare_disagreement_analysis_sampling.py --names 2_wo_v${version}_${model}_check_tie 2_wo_v0_${model}_check_rephrase_better --results_dir results_analysis3 --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"
    done
done

python3 scripts/compare_disagreement_direct_human_annotation_with_ours.py \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    --results_dir results_full \
    --annotation_file_path tmp/human_annotations.json \
    --prompt_file_path /home/xinxil/open-instruct/tmp/HumanIF\ Eval\ -\ Test\ Data\ Annotations\ \(New\)-Overall.csv


# Getting Model Output
# for version in 2_w_v0 ; 
# do
# # GPT4, Claude, Palm
#     for model in "mistralai/Mistral-7B-Instruct-v0.3" "allenai/OLMo-7B-Instruct-hf" "lmsys/fastchat-t5-3b-v1.0" "stabilityai/stablelm-tuned-alpha-7b" "WizardLMTeam/WizardLM-13B-V1.2" "lmsys/vicuna-13b-v1.5" "lmsys/vicuna-7b-v1.5" "meta-llama/Meta-Llama-3-8B-Instruct"; 
#     do
#         python -m eval.alpaca_farm.run_humanif_eval \
#                 --limit_eval_size 200 \
#                 --use_vllm \
#                 --raw_output_path /references/llama_references_test_dataset_full.json \
#                 --model_name_or_path ${model} \
#                 --tokenizer_name_or_path ${model} \
#                 --save_dir results_model_outputs \
#                 --use_chat_format \
#                 --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
#                 --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
#                 --config_dir eval/alpaca_farm/configs \
#                 --config_name 2_wo_v${version}_llama3
#     done
# done


python -m eval.alpaca_farm.run_humanif_eval \
        --limit_eval_size 200 \
        --openai_engine gpt-3.5-turbo \
        --max_new_tokens 4096 \
        --raw_output_path tmp/references/raw_prompts_8x25.json \
        --save_dir results_model_outputs_8x25/ \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" 
# "meta-llama/Llama-2-7b-chat-hf" "mistralai/Mistral-7B-Instruct-v0.3" "allenai/OLMo-7B-Instruct-hf" "WizardLMTeam/WizardLM-13B-V1.2" "lmsys/vicuna-13b-v1.5" "lmsys/vicuna-7b-v1.5" "meta-llama/Meta-Llama-3-8B-Instruct" "meta-llama/Llama-2-13b-chat-hf"  "databricks/dolly-v2-7b" "mosaicml/mpt-7b-chat" "databricks/dolly-v2-12b" "OpenAssistant/oasst-sft-1-pythia-12b" "nomic-ai/gpt4all-13b-snoozy" "TheBloke/koala-13B-HF" "gpt-3.5-turbo"
for model in "WizardLMTeam/WizardLM-13B-V1.2" ; 
do
    # python -m eval.alpaca_farm.run_humanif_eval \
    #             --limit_eval_size 200 \
    #             --use_vllm \
    #             --raw_output_path tmp/references/raw_prompts_8x25.json \
    #             --model_name_or_path ${model} \
    #             --tokenizer_name_or_path ${model} \
    #             --save_dir results_model_outputs_8x25/ \
    #             --use_chat_format \
    #             --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    #             --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" 
        
    python3 -m scripts.combine_references \
        --model_name ${model} \
        --prediction_dir results_model_outputs_8x25\
        --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data' \
        --save_dir tmp/references
done
#  
#  

# "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf" "meta-llama/Meta-Llama-3-8B-Instruct" "nomic-ai/gpt4all-13b-snoozy"  "OpenAssistant/oasst-sft-1-pythia-12b" "allenai/OLMo-7B-Instruct-hf" "WizardLMTeam/WizardLM-13B-V1.2" "mistralai/Mistral-7B-Instruct-v0.3" "mosaicml/mpt-7b-chat"
# "databricks/dolly-v2-7b" "databricks/dolly-v2-12b" "TheBloke/koala-7B-HF" "TheBloke/koala-13B-HF"  "lmsys/vicuna-7b-v1.5" "lmsys/vicuna-13b-v1.5"
for model in "databricks/dolly-v2-7b" "databricks/dolly-v2-12b" "TheBloke/koala-7B-HF" "TheBloke/koala-13B-HF" "lmsys/vicuna-7b-v1.5" "lmsys/vicuna-13b-v1.5"; 
do
    # python -m eval.alpaca_farm.run_humanif_eval \
    #             --limit_eval_size 200 \
    #             --use_vllm \
    #             --raw_output_path tmp/references/raw_prompts_8x25.json \
    #             --model_name_or_path ${model} \
    #             --tokenizer_name_or_path ${model} \
    #             --save_dir results_model_outputs_8x25_new/ \
    #             --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    #             --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" 
    #             # --use_chat_format \
    python3 -m scripts.combine_references \
        --model_name ${model} \
        --prediction_dir results_model_outputs_8x25_new\
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
        --save_dir tmp/references
done


for model in "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf" "meta-llama/Meta-Llama-3-8B-Instruct" "nomic-ai/gpt4all-13b-snoozy"  "OpenAssistant/oasst-sft-1-pythia-12b" "allenai/OLMo-7B-Instruct-hf" "WizardLMTeam/WizardLM-13B-V1.2" "mistralai/Mistral-7B-Instruct-v0.3" "mosaicml/mpt-7b-chat"; 
do
    # python -m eval.alpaca_farm.run_humanif_eval \
    #             --limit_eval_size 200 \
    #             --use_vllm \
    #             --raw_output_path tmp/references/raw_prompts_8x25.json \
    #             --model_name_or_path ${model} \
    #             --tokenizer_name_or_path ${model} \
    #             --save_dir results_model_outputs_8x25_new/ \
    #             --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
    #             --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
    #             --use_chat_format
    python3 -m scripts.combine_references \
        --model_name ${model} \
        --prediction_dir results_model_outputs_8x25_new\
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
        --save_dir tmp/references
done

for model in allenai/OLMo-7B-Instruct-hf ; 
do
    python -m test_generation \
                --limit_eval_size 200 \
                --model_name_or_path ${model} \
                --tokenizer_name_or_path ${model} \
                --use_vllm \
                --chat_formatting_function templates.create_prompt_with_huggingface_tokenizer_template \
                --use_chat_format 
done

for model in "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Llama-2-13b-chat-hf" "meta-llama/Meta-Llama-3-8B-Instruct" "allenai/OLMo-7B-Instruct-hf" "mistralai/Mistral-7B-Instruct-v0.3"; 
do
    python -m eval.alpaca_farm.run_humanif_eval \
                --limit_eval_size 200 \
                --use_vllm \
                --raw_output_path tmp/references/raw_prompts_8x25.json \
                --model_name_or_path ${model} \
                --tokenizer_name_or_path ${model} \
                --save_dir results_model_outputs_8x25_new2/ \
                --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
                --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
                --use_chat_format
    # python3 -m scripts.combine_references \
    #     --model_name ${model} \
    #     --prediction_dir results_model_outputs_8x25_new\
    #     --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
    #     --save_dir tmp/references
done


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
docker build --build-arg CUDA=11.8.0 --build-arg TARGET=cudnn8-devel --build-arg DIST=ubuntu20.04 . -t xinxil_generate_outputs
beaker image delete alrope/xinxil_generate_outputs
beaker image create xinxil_generate_outputs -n xinxil_generate_outputs -w ai2/xinxil-default

beaker session create --budget ai2/oe-adapt --cpus 15.5 --gpus 2 --workspace ai2/pradeepd-open-instruct --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env OPENAI_API_KEY=openai_api_key --secret-env HF_TOKEN=huggingface_token

for model in "meta-llama/Meta-Llama-3-8B-Instruct" "meta-llama/Llama-2-7b-chat-hf" "allenai/OLMo-7B-Instruct-hf" "mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Llama-2-13b-chat-hf" ; 
do
    python -m eval.alpaca_farm.run_humanif_eval \
                --limit_eval_size 200 \
                --use_vllm \
                --raw_output_path tmp/references/raw_prompts_8x25.json \
                --model_name_or_path ${model} \
                --tokenizer_name_or_path ${model} \
                --save_dir results_model_outputs_8x25_new2/ \
                --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
                --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" \
                --use_chat_format
    # python3 -m scripts.combine_references \
    #     --model_name ${model} \
    #     --prediction_dir results_model_outputs_8x25_new2\
    #     --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
    #     --save_dir tmp/references_new2
done

for model in "lmsys/vicuna-13b-v1.5" "lmsys/vicuna-7b-v1.5" "mosaicml/mpt-7b-chat" "TheBloke/koala-7B-HF" "OpenAssistant/oasst-sft-1-pythia-12b" "TheBloke/koala-13B-HF" "WizardLMTeam/WizardLM-13B-V1.2" ; 
do
    python -m eval.alpaca_farm.run_humanif_eval \
                --limit_eval_size 200 \
                --use_vllm \
                --raw_output_path tmp/references/raw_prompts_8x25.json \
                --model_name_or_path ${model} \
                --tokenizer_name_or_path ${model} \
                --save_dir results_model_outputs_8x25_new2/ \
                --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
                --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"
    # python3 -m scripts.combine_references \
    #     --model_name ${model} \
    #     --prediction_dir results_model_outputs_8x25_new\
    #     --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
    #     --save_dir tmp/references
done

for model in "databricks/dolly-v2-7b" "databricks/dolly-v2-12b"; 
do
    python -m eval.alpaca_farm.run_humanif_eval \
                --limit_eval_size 200 \
                --use_vllm \
                --raw_output_path tmp/references/raw_prompts_8x25.json \
                --model_name_or_path ${model} \
                --tokenizer_name_or_path ${model} \
                --save_dir results_model_outputs_8x25_new2/ \
                --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
                --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"
    # python3 -m scripts.combine_references \
    #     --model_name ${model} \
    #     --prediction_dir results_model_outputs_8x25_new\
    #     --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
    #     --save_dir tmp/references
done

for model in "nomic-ai/gpt4all-13b-snoozy"; 
do
    python -m eval.alpaca_farm.run_humanif_eval \
                --limit_eval_size 200 \
                --use_vllm \
                --raw_output_path tmp/references/raw_prompts_8x25.json \
                --model_name_or_path ${model} \
                --tokenizer_name_or_path ${model} \
                --save_dir results_model_outputs_8x25_new2/ \
                --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
                --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify"
    # python3 -m scripts.combine_references \
    #     --model_name ${model} \
    #     --prediction_dir results_model_outputs_8x25_new\
    #     --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
    #     --save_dir tmp/references
done

python -m eval.alpaca_farm.run_humanif_eval \
        --limit_eval_size 200 \
        --openai_engine gpt-3.5-turbo \
        --max_new_tokens 4096 \
        --raw_output_path tmp/references/raw_prompts_8x25.json \
        --save_dir results_model_outputs_8x25_new2/ \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Coding" "Classify" 

# "gpt-3.5-turbo" "meta-llama/Meta-Llama-3-8B-Instruct" "meta-llama/Llama-2-7b-chat-hf" "allenai/OLMo-7B-Instruct-hf" "mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Llama-2-13b-chat-hf" "lmsys/vicuna-13b-v1.5" "lmsys/vicuna-7b-v1.5" "mosaicml/mpt-7b-chat" "TheBloke/koala-7B-HF" "OpenAssistant/oasst-sft-1-pythia-12b" "TheBloke/koala-13B-HF" "WizardLMTeam/WizardLM-13B-V1.2" "databricks/dolly-v2-7b" "databricks/dolly-v2-12b" "nomic-ai/gpt4all-13b-snoozy" ; 
for model in gpt-3.5-turbo ; 
do
    python3 -m scripts.combine_references \
        --model_name ${model} \
        --prediction_dir results_model_outputs_8x25_new2\
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
        --save_dir tmp/references_new2
done





#=============================+#

beaker session create --budget ai2/oe-adapt --cpus 15.5 --gpus 2 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env HF_TOKEN=huggingface_token


for model in  "Meta-Llama-3-70B-Instruct"  ; 
do
python3 scripts/submit_through_interactive.py \
    --name ${model}_no_robot_8_times_25_generation \
    --command "beaker session create --budget ai2/oe-adapt --gpus 2 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env HF_TOKEN=huggingface_token --mount beaker://alrope/xinxil_${model}=/model/${model}" \
    --argument "python -m eval.alpaca_farm.run_humanif_eval                 --limit_eval_size 200                 --use_vllm                 --raw_output_path /references/raw_prompts_8x25.json                 --model_name_or_path /model/${model}                 --tokenizer_name_or_path meta-llama/${model}                 --save_dir /output                 --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template                 --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data'           --use_chat_format  --temperature 0.0  --cache cache"
done

for model in "Qwen2-72B-Instruct"; 
do
python3 scripts/submit_through_interactive.py \
    --name ${model}_no_robot_8_times_25_generation \
    --command "beaker session create --budget ai2/oe-adapt --gpus 4 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env HF_TOKEN=huggingface_token --mount beaker://alrope/xinxil_${model}=/model/${model}" \
    --argument "python -m eval.alpaca_farm.run_humanif_eval                 --limit_eval_size 200                 --use_vllm                 --raw_output_path /references/raw_prompts_8x25.json                 --model_name_or_path /model/${model}                 --tokenizer_name_or_path  Qwen/${model}                 --save_dir /output                 --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template                 --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data'                 --use_chat_format  --temperature 0.0  --cache cache" \
    --cluster ai2/jupiter-cirrascale-2 \
    --priority preemptible
done


for model in "Qwen1.5-110B-Chat" ; 
do
python3 scripts/submit_through_interactive.py \
    --name ${model}_no_robot_8_times_25_generation \
    --command "beaker session create --budget ai2/oe-adapt --gpus 4 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env HF_TOKEN=huggingface_token --mount beaker://alrope/xinxil_${model}=/model/${model}" \
    --argument "python -m eval.alpaca_farm.run_humanif_eval                 --limit_eval_size 200                 --use_vllm                 --raw_output_path /references/raw_prompts_8x25.json                 --model_name_or_path /model/${model}                 --tokenizer_name_or_path  Qwen/${model}                 --save_dir /output                 --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template                 --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data'                 --use_chat_format  --temperature 0.0  --cache cache" \
    --cluster ai2/jupiter-cirrascale-2
done

for model in "DeepSeek-Coder-V2-Instruct"  ; 
do
python3 scripts/submit_through_interactive.py \
    --name ${model}_no_robot_8_times_25_generation \
    --command "beaker session create --budget ai2/oe-adapt --gpus 8 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env HF_TOKEN=huggingface_token --mount beaker://alrope/xinxil_${model}=/model/${model}" \
    --argument "python -m eval.alpaca_farm.run_humanif_eval                 --limit_eval_size 200                 --use_vllm                 --raw_output_path /references/raw_prompts_8x25.json                 --model_name_or_path /model/${model}                 --tokenizer_name_or_path  deepseek-ai/${model}                 --save_dir /output                 --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template                 --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data'                 --use_chat_format  --temperature 0.0  --cache cache" \
    --cluster ai2/jupiter-cirrascale-2 
done

for model in "Mixtral-8x22B-v0.1"  ; 
do
python3 scripts/submit_through_interactive.py \
    --name ${model}_no_robot_8_times_25_generation \
    --command "beaker session create --budget ai2/oe-adapt --gpus 8 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env HF_TOKEN=huggingface_token --mount beaker://alrope/xinxil_${model}=/model/${model}" \
    --argument "python -m eval.alpaca_farm.run_humanif_eval                 --limit_eval_size 200                 --use_vllm                 --raw_output_path /references/raw_prompts_8x25.json                 --model_name_or_path /model/${model}                 --tokenizer_name_or_path  mistralai/${model}                 --save_dir /output    --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data'  --temperature 0.0  --cache cache  --max_new_tokens 2048" \
    --cluster ai2/allennlp-cirrascale
done

for model in Phi-3-medium-4k-instruct;
do
python3 scripts/submit_through_interactive.py \
    --name ${model}_no_robot_8_times_25_generation \
    --command "beaker session create --budget ai2/oe-adapt --gpus 1 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env HF_TOKEN=huggingface_token --mount beaker://alrope/xinxil_${model}=/model/${model}" \
    --argument "python -m eval.alpaca_farm.run_humanif_eval                 --limit_eval_size 200                 --use_vllm                 --raw_output_path /references/raw_prompts_8x25.json                 --model_name_or_path /model/${model}                 --tokenizer_name_or_path  microsoft/${model}                 --save_dir /output                 --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template                 --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data'                 --use_chat_format  --temperature 0.0  --cache cache" 
done

for model in gpt-3.5-turbo gpt-4-1106-preview gpt-4-turbo-2024-04-09 gpt-4o-2024-05-13 Llama-2-70b-chat-hf Meta-Llama-3-70B-Instruct Mixtral-8x22B-v0.1 Phi-3-medium-4k-instruct Qwen1.5-110B-Chat Qwen2-72B-Instruct Yi-1.5-34B-Chat Meta-Llama-3-8B-Instruct ;
do
python3 -m scripts.combine_references \
        --model_name ${model} \
        --prediction_dir results_model_outputs_8x25_stronger\
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
        --save_dir tmp/references_new3
done


# gpt-4o-2024-05-13 gpt-4-1106-preview 
for model in gpt-4-turbo-2024-04-09 ;
do
    python -m eval.alpaca_farm.run_humanif_eval \
            --limit_eval_size 200 \
            --openai_engine ${model} \
            --max_new_tokens 4096 \
            --raw_output_path tmp/references/raw_prompts_8x25.json \
            --save_dir results_model_outputs_8x25_stronger/ \
            --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify"
done
################

for model in meta-llama/Meta-Llama-3-70B-Instruct meta-llama/Meta-Llama-3.1-70B-Instruct meta-llama/Meta-Llama-3.1-405B-Instruct ; 
do
python3 scripts/submit_through_interactive.py \
    --name download_${model} \
    --command "beaker session create --budget ai2/oe-adapt --cpus 4 --gpus 0 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs  --no-update-default-image --secret-env HF_TOKEN=huggingface_token" \
    --argument "python -m eval.alpaca_farm.download_only --model_name_or_path ${model} --tokenizer_name_or_path ${model} --save_dir /output --cache_dir cache" 
done

beaker session create --budget ai2/oe-adapt --cpus 15.5 --gpus 1 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env HF_TOKEN=huggingface_token --mount beaker://alrope/xinxil-Meta-Llama-3-8B-Instruct=/model

python -m eval.alpaca_farm.run_humanif_eval                 --limit_eval_size 200                 --use_vllm                 --raw_output_path /references/raw_prompts_8x25.json                 --model_name_or_path /model                 --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B-Instruct                 --save_dir /output                 --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template                 --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data'                 --use_chat_format  --temperature 0.0 --cache_dir cache


##################################### 8X25 Annotation ######################################################################################################################################
for version in 0 ; 
do
    # for model in llama3 ; 
    # do
    #     python -m eval.alpaca_farm.run_humanif_eval \
    #     --limit_eval_size 200 \
    #     --use_vllm \
    #     --output_path /references/mixed_b_references_8x25.json \
    #     --reference_path /references/mixed_a_references_8x25.json \
    #     --save_dir /net/nfs.cirrascale/allennlp/xinxil/results_8x25/2_wo_v${version}_${model} \
    #     --use_chat_format \
    #     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
    #     --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
    #     --config_dir eval/alpaca_farm/configs \
    #     --config_name 2_wo_v${version}_${model}
    #     # --save_dir results_8x25/2_wo_v${version}_${model} \
    # done

    # # pip install pip install vllm==0.5.0
    # # pip install prometheus-eval
    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/mixed_a_references_8x25.json \
    # --reference_path /references/mixed_b_references_8x25.json \
    # --save_dir results_8x25/2_wo_v${version}_prometheus/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus

    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/mixed_a_references_8x25.json \
    # --reference_path /references/mixed_b_references_8x25.json \
    # --save_dir results_8x25/2_wo_v${version}_prometheus_w_tie/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus\
    # --use_tie


    # # # Use human reference
    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/mixed_a_references_8x25.json \
    # --reference_path /references/mixed_b_references_8x25.json \
    # --human_path tmp/references/mixed_human_references_8x25.json \
    # --save_dir results_8x25/2_wo_v${version}_prometheus_w_reference/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus \
    # --use_human_reference 


    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/mixed_a_references_8x25.json \
    # --reference_path /references/mixed_b_references_8x25.json \
    # --human_path tmp/references/mixed_human_references_8x25.json \
    # --save_dir results_8x25/2_wo_v${version}_prometheus_w_tie_w_reference/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus\
    # --use_tie \
    # --use_human_reference

done

for version in 2_w_v0 ; 
do
    for model in llama3 ; 
    do
        python -m eval.alpaca_farm.run_humanif_eval \
        --limit_eval_size 200 \
        --use_vllm \
        --output_path /references/mixed_b_references_8x25.json \
        --reference_path /references/mixed_a_references_8x25.json \
        --save_dir /net/nfs.cirrascale/allennlp/xinxil/results_8x25/${version}_${model} \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
        --config_dir eval/alpaca_farm/configs \
        --config_name ${version}_${model} \
        --embed_human_response \
        --human_path /net/nfs.cirrascale/allennlp/xinxil/mixed_human_references_8x25.json 
        # --human_path /references/mixed_human_references_8x25.json \
        # --save_dir results_8x25/${version}_${model} \
    done
done

# Automatic
python -m eval.automatic.run_automatic \
    --output_path /references/mixed_a_references_8x25.json \
    --reference_path /references/mixed_b_references_8x25.json \
    --human_path tmp/references/mixed_human_references_8x25.json \
    --save_dir results_8x25/ \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" 

# Baselines
for type in longer shorter random_two random_three ; 
do
    python3 -m eval.baselines.run_baselines \
        --output_path /references/mixed_a_references_8x25.json \
        --reference_path /references/mixed_b_references_8x25.json \
        --save_dir results_8x25/${type} \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
        --type ${type}
done

for type in longer shorter random_two random_three more_lines less_lines; 
do
    python3 -m eval.baselines.run_baselines \
        --output_path tmp/references/mixed_a_references_8x25.json \
        --reference_path tmp/references/mixed_b_references_8x25.json \
        --save_dir results_8x25/${type} \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
        --type ${type}
done

# Analysis
python3 scripts/compare_disagreement_direct_human_annotation_8x25.py \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
    --results_dir results_8x25 \
    --models 2_wo_v0_gpt4 2_w_v0_gpt4 2_wo_v0_llama3 2_w_v0_llama3 2_wo_v0_prometheus 2_wo_v0_prometheus_w_reference 2_wo_v0_prometheus_w_tie 2_wo_v0_prometheus_w_tie_w_reference BERTScore rouge-1 rouge-2 rouge-l longer shorter more_lines less_lines random_two random_three \
    --preference_keys preference preference preference preference  preference preference preference preference preference preference preference preference preference preference preference preference preference preference \
    --annotation_file_path tmp/human_annotations_8x25.json  



##################################### 8X25 stronger Annotation ######################################################################################################################################
for version in 0 ; 
do
    for model in gpt4 ; 
    do
        python -m eval.alpaca_farm.run_humanif_eval \
        --limit_eval_size 200 \
        --use_vllm \
        --output_path /references/mixed_b_references_8x25_stronger.json \
        --reference_path /references/mixed_a_references_8x25_stronger.json \
        --save_dir results_8x25_stronger/2_wo_v${version}_${model} \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
        --config_dir eval/alpaca_farm/configs \
        --config_name 2_wo_v${version}_${model}
        # --save_dir results_8x25/2_wo_v${version}_${model} \
    done

    # # pip install pip install vllm==0.5.0
    # # pip install prometheus-eval
    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/mixed_a_references_8x25_stronger.json \
    # --reference_path /references/mixed_b_references_8x25_stronger.json \
    # --save_dir results_8x25_stronger/2_wo_v${version}_prometheus/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus

    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/mixed_a_references_8x25_stronger.json \
    # --reference_path /references/mixed_b_references_8x25_stronger.json \
    # --save_dir results_8x25_stronger/2_wo_v${version}_prometheus_w_tie/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus\
    # --use_tie


    # # # Use human reference
    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/mixed_a_references_8x25_stronger.json \
    # --reference_path /references/mixed_b_references_8x25_stronger.json \
    # --human_path /references/mixed_human_references_8x25_stronger.json \
    # --save_dir results_8x25_stronger/2_wo_v${version}_prometheus_w_reference/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus \
    # --use_human_reference 


    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/mixed_a_references_8x25_stronger.json \
    # --reference_path /references/mixed_b_references_8x25_stronger.json \
    # --human_path /references/mixed_human_references_8x25_stronger.json \
    # --save_dir results_8x25_stronger/2_wo_v${version}_prometheus_w_tie_w_reference/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus\
    # --use_tie \
    # --use_human_reference

done

for version in 2_w_v0 ; 
do
    for model in gpt4 ; 
    do
        python -m eval.alpaca_farm.run_humanif_eval \
        --limit_eval_size 200 \
        --use_vllm \
        --output_path /references/mixed_b_references_8x25_stronger.json \
        --reference_path /references/mixed_a_references_8x25_stronger.json \
        --save_dir results_8x25_stronger/${version}_${model} \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
        --config_dir eval/alpaca_farm/configs \
        --config_name ${version}_${model} \
        --embed_human_response \
        --human_path /references/mixed_human_references_8x25_stronger.json
        # --human_path /net/nfs.cirrascale/allennlp/xinxil/mixed_human_references_8x25.json 
        # --save_dir results_8x25/${version}_${model} \
    done
done

# Automatic
python -m eval.automatic.run_automatic \
    --output_path /references/mixed_a_references_8x25_stronger.json \
    --reference_path /references/mixed_b_references_8x25_stronger.json \
    --human_path /references/mixed_human_references_8x25_stronger.json \
    --save_dir results_8x25_stronger/ \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" 

# Baselines
for type in longer shorter random_two random_three more_lines less_lines ; 
do
    python3 -m eval.baselines.run_baselines \
        --output_path /references/mixed_a_references_8x25_stronger.json \
        --reference_path /references/mixed_b_references_8x25_stronger.json \
        --save_dir results_8x25_stronger/${type} \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
        --type ${type}
done

# Analysis
python3 scripts/compare_disagreement_direct_human_annotation_8x25.py \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
    --results_dir results_8x25_stronger \
    --models 2_wo_v0_gpt4 2_w_v0_gpt4 2_wo_v0_llama3 2_w_v0_llama3 2_wo_v0_prometheus 2_wo_v0_prometheus_w_reference 2_wo_v0_prometheus_w_tie 2_wo_v0_prometheus_w_tie_w_reference BERTScore rouge-1 rouge-2 rouge-l longer shorter random_two random_three more_lines less_lines \
    --preference_keys preference preference preference preference  preference preference preference preference preference preference preference preference preference preference preference preference preference preference \
    --annotation_file_path tmp/human_annotations_8x25_stronger.json  


# ----------------------------------------------------- 11x25 ------------------------------------------------------------------------------------
docker build --build-arg CUDA=11.8.0 --build-arg TARGET=cudnn8-devel --build-arg DIST=ubuntu20.04 . -t xinxil_generate_outputs
beaker image delete alrope/xinxil_generate_outputs
beaker image create xinxil_generate_outputs -n xinxil_generate_outputs -w ai2/xinxil-default

beaker session create --budget ai2/oe-adapt --cpus 15.5 --gpus 2 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env HF_TOKEN=huggingface_token --bare 
export HF_TOKEN=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

# "meta-llama/Meta-Llama-3-8B-Instruct" "meta-llama/Llama-2-7b-chat-hf" "allenai/OLMo-7B-SFT-hf" "mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Llama-2-13b-chat-hf"
for model in "meta-llama/Llama-2-13b-chat-hf" ; 
do
    python -m eval.alpaca_farm.run_humanif_eval \
                --limit_eval_size 200 \
                --use_vllm \
                --raw_output_path /references/raw_prompts_11x25.json \
                --model_name_or_path ${model} \
                --tokenizer_name_or_path ${model} \
                --save_dir /net/nfs.cirrascale/allennlp/xinxil/results_model_outputs_11x25/ \
                --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
                --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data' \
                --use_chat_format
done

for model in "lmsys/vicuna-13b-v1.5" "lmsys/vicuna-7b-v1.5" "mosaicml/mpt-7b-chat" "TheBloke/koala-7B-HF" "OpenAssistant/oasst-sft-1-pythia-12b" "TheBloke/koala-13B-HF" "WizardLMTeam/WizardLM-13B-V1.2" ; 
do
    python -m eval.alpaca_farm.run_humanif_eval \
                --limit_eval_size 200 \
                --use_vllm \
                --raw_output_path /references/raw_prompts_11x25.json \
                --model_name_or_path ${model} \
                --tokenizer_name_or_path ${model} \
                --save_dir /net/nfs.cirrascale/allennlp/xinxil/results_model_outputs_11x25/ \
                --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
                --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data'
done

for model in "databricks/dolly-v2-7b" "databricks/dolly-v2-12b"; 
do
    python -m eval.alpaca_farm.run_humanif_eval \
                --limit_eval_size 200 \
                --use_vllm \
                --raw_output_path /references/raw_prompts_11x25.json \
                --model_name_or_path ${model} \
                --tokenizer_name_or_path ${model} \
                --save_dir /net/nfs.cirrascale/allennlp/xinxil/results_model_outputs_11x25/ \
                --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
                --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data'
done

for model in "nomic-ai/gpt4all-13b-snoozy"; 
do
    python -m eval.alpaca_farm.run_humanif_eval \
                --limit_eval_size 200 \
                --use_vllm \
                --raw_output_path /references/raw_prompts_11x25.json \
                --model_name_or_path ${model} \
                --tokenizer_name_or_path ${model} \
                --save_dir /net/nfs.cirrascale/allennlp/xinxil/results_model_outputs_11x25/ \
                --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
                --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data'
done

python -m eval.alpaca_farm.run_humanif_eval \
        --limit_eval_size 200 \
        --openai_engine gpt-3.5-turbo \
        --max_new_tokens 4096 \
        --raw_output_path /references/raw_prompts_11x25.json \
        --save_dir /net/nfs.cirrascale/allennlp/xinxil/results_model_outputs_11x25/ \
        --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data' 

# "gpt-3.5-turbo" "meta-llama/Meta-Llama-3-8B-Instruct" "meta-llama/Llama-2-7b-chat-hf" "allenai/OLMo-7B-Instruct-hf" "mistralai/Mistral-7B-Instruct-v0.3" "meta-llama/Llama-2-13b-chat-hf" "lmsys/vicuna-13b-v1.5" "lmsys/vicuna-7b-v1.5" "mosaicml/mpt-7b-chat" "TheBloke/koala-7B-HF" "OpenAssistant/oasst-sft-1-pythia-12b" "TheBloke/koala-13B-HF" "WizardLMTeam/WizardLM-13B-V1.2" "databricks/dolly-v2-7b" "databricks/dolly-v2-12b" "nomic-ai/gpt4all-13b-snoozy" ; 
for model in gpt-3.5-turbo ; 
do
    python3 -m scripts.combine_references \
        --model_name ${model} \
        --prediction_dir results_model_outputs_3x25\
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" \
        --save_dir tmp/references_new2
done

#######################################################################
for model in  "Meta-Llama-3-70B-Instruct" "Llama-2-70b-chat-hf" ; 
do
python3 scripts/submit_through_interactive.py \
    --name ${model}_no_robot_11_times_25_generation \
    --command "beaker session create --budget ai2/oe-adapt --gpus 2 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env HF_TOKEN=huggingface_token --mount beaker://alrope/xinxil_${model}=/model/${model}" \
    --argument "python -m eval.alpaca_farm.run_humanif_eval                 --limit_eval_size 200                 --use_vllm                 --raw_output_path /references/raw_prompts_11x25.json                 --model_name_or_path /model/${model}                 --tokenizer_name_or_path meta-llama/${model}                 --save_dir /output                 --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template                 --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data'                 --use_chat_format  --temperature 1.0  --cache cache" \
    --cluster ai2/pluto-cirrascale 
done

for model in Yi-1.5-34B-Chat ; 
do
python3 scripts/submit_through_interactive.py \
    --name ${model}_no_robot_11_times_25_generation \
    --command "beaker session create --budget ai2/oe-adapt --gpus 1 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env HF_TOKEN=huggingface_token --mount beaker://alrope/xinxil_${model}=/model/${model}" \
    --argument "python -m eval.alpaca_farm.run_humanif_eval                 --limit_eval_size 200                 --use_vllm                 --raw_output_path /references/raw_prompts_11x25.json                 --model_name_or_path /model/${model}                 --tokenizer_name_or_path 01-ai/${model}                 --save_dir /output                 --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template                 --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data'                 --use_chat_format  --temperature 1.0  --cache cache" \
    --cluster ai2/pluto-cirrascale 
done

for model in "Qwen2-72B-Instruct"; 
do
python3 scripts/submit_through_interactive.py \
    --name ${model}_no_robot_11_times_25_generation \
    --command "beaker session create --budget ai2/oe-adapt --gpus 4 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env HF_TOKEN=huggingface_token --mount beaker://alrope/xinxil_${model}=/model/${model}" \
    --argument "python -m eval.alpaca_farm.run_humanif_eval                 --limit_eval_size 200                 --use_vllm                 --raw_output_path /references/raw_prompts_11x25.json                 --model_name_or_path /model/${model}                 --tokenizer_name_or_path  Qwen/${model}                 --save_dir /output                 --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template                 --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data'                 --use_chat_format  --temperature 1.0  --cache cache" \
    --cluster ai2/pluto-cirrascale 
done


for model in "Qwen1.5-110B-Chat" ; 
do
python3 scripts/submit_through_interactive.py \
    --name ${model}_no_robot_11_times_25_generation \
    --command "beaker session create --budget ai2/oe-adapt --gpus 4 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env HF_TOKEN=huggingface_token --mount beaker://alrope/xinxil_${model}=/model/${model}" \
    --argument "python -m eval.alpaca_farm.run_humanif_eval                 --limit_eval_size 200                 --use_vllm                 --raw_output_path /references/raw_prompts_11x25.json                 --model_name_or_path /model/${model}                 --tokenizer_name_or_path  Qwen/${model}                 --save_dir /output                 --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template                 --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data'                 --use_chat_format  --temperature 1.0  --cache cache" \
    --cluster ai2/pluto-cirrascale
done

for model in "DeepSeek-Coder-V2-Instruct"  ; 
do
python3 scripts/submit_through_interactive.py \
    --name ${model}_no_robot_11_times_25_generation \
    --command "beaker session create --budget ai2/oe-adapt --gpus 8 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env HF_TOKEN=huggingface_token --mount beaker://alrope/xinxil_${model}=/model/${model}" \
    --argument "python -m eval.alpaca_farm.run_humanif_eval                 --limit_eval_size 200                 --use_vllm                 --raw_output_path /references/raw_prompts_11x25.json                 --model_name_or_path /model/${model}                 --tokenizer_name_or_path  deepseek-ai/${model}                 --save_dir /output                 --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template                 --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data'                 --use_chat_format  --temperature 1.0  --cache cache" \
    --cluster ai2/pluto-cirrascale 
done

for model in "Mixtral-8x22B-v0.1"  ; 
do
python3 scripts/submit_through_interactive.py \
    --name ${model}_no_robot_11_times_25_generation \
    --command "beaker session create --budget ai2/oe-adapt --gpus 8 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env HF_TOKEN=huggingface_token --mount beaker://alrope/xinxil_${model}=/model/${model}" \
    --argument "python -m eval.alpaca_farm.run_humanif_eval                 --limit_eval_size 200                 --use_vllm                 --raw_output_path /references/raw_prompts_11x25.json                 --model_name_or_path /model/${model}                 --tokenizer_name_or_path  mistralai/${model}                 --save_dir /output    --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data' --use_chat_format --temperature 1.0  --cache cache  --max_new_tokens 2048" \
    --cluster ai2/pluto-cirrascale 
done

for model in Phi-3-medium-4k-instruct;
do
python3 scripts/submit_through_interactive.py \
    --name ${model}_no_robot_11_times_25_generation \
    --command "beaker session create --budget ai2/oe-adapt --gpus 1 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env HF_TOKEN=huggingface_token --mount beaker://alrope/xinxil_${model}=/model/${model}" \
    --argument "python -m eval.alpaca_farm.run_humanif_eval                 --limit_eval_size 200                 --use_vllm                 --raw_output_path /references/raw_prompts_11x25.json                 --model_name_or_path /model/${model}                 --tokenizer_name_or_path  microsoft/${model}                 --save_dir /output                 --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template                 --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data'                 --use_chat_format  --temperature 1.0  --cache cache" \
    --cluster ai2/pluto-cirrascale 
done

beaker session create --budget ai2/oe-adapt --cpus 15.5 --workspace ai2/pradeepd-open-instruct --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --no-update-default-image --secret-env HF_TOKEN=huggingface_token --secret-env OPENAI_API_KEY=openai_api_key
# gpt-4o-2024-05-13 gpt-4-1106-preview gpt-4-turbo-2024-04-09
for model in gpt-4o-2024-05-13 gpt-4-1106-preview gpt-4-turbo-2024-04-09  ;
do
    python -m eval.alpaca_farm.run_humanif_eval \
        --limit_eval_size 200 \
        --openai_engine ${model} \
        --max_new_tokens 4096 \
        --raw_output_path /references/raw_prompts_11x25.json \
        --save_dir results_model_outputs_11x25/ \
        --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Coding' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data' \
        --temperature 1.0
done

for model in "dolly-v2-7b" "dolly-v2-12b" gpt-3.5-turbo gpt-4-1106-preview gpt-4-turbo-2024-04-09 gpt-4o-2024-05-13 "gpt4all-13b-snoozy" "Meta-Llama-3-8B-Instruct" "koala-7B-HF" "koala-13B-HF" "Llama-2-7b-chat-hf" "Llama-2-13b-chat-hf" Llama-2-70b-chat-hf  Meta-Llama-3-8B-Instruct  Meta-Llama-3-70B-Instruct "Mistral-7B-Instruct-v0.3" "mpt-7b-chat" "oasst-sft-1-pythia-12b" "OLMo-7B-SFT-hf" Phi-3-medium-4k-instruct Qwen1.5-110B-Chat Qwen2-72B-Instruct "vicuna-7b-v1.5" "vicuna-13b-v1.5"   "WizardLM-13B-V1.2" Yi-1.5-34B-Chat  ;
do
python3 -m scripts.combine_references \
        --model_name ${model} \
        --prediction_dir results_model_outputs_11x25\
        --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data' \
        --save_dir tmp/references_11x25
done


##################################### 11X25 stronger Annotation ######################################################################################################################################


for model in 01-ai/Yi-1.5-34B-Chat ; 
do
python3 scripts/submit_through_interactive.py \
    --name download_${model} \
    --command "beaker session create --budget ai2/oe-adapt --cpus 4 --gpus 0 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs  --no-update-default-image --secret-env HF_TOKEN=huggingface_token" \
    --argument "python -m eval.alpaca_farm.download_only --model_name_or_path ${model} --tokenizer_name_or_path ${model} --save_dir /output --cache_dir cache" \
    --cluster ai2/jupiter-cirrascale-2
done

for version in 0 ; 
do
    for model in llama3-70b ; 
    do
        python -m eval.alpaca_farm.run_humanif_eval \
        --limit_eval_size 200 \
        --use_vllm \
        --output_path /references/mixed_b_references_11x25_stronger.json \
        --reference_path /references/mixed_a_references_11x25_stronger.json \
        --save_dir  /net/nfs.cirrascale/allennlp/xinxil/results_11x25_stronger/2_wo_v${version}_${model} \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data'\
        --config_dir eval/alpaca_farm/configs \
        --config_name 2_wo_v${version}_${model}
        # --save_dir  /net/nfs.cirrascale/allennlp/xinxil/results_11x25/2_wo_v${version}_${model} \
    done

    # # pip install pip install vllm==0.5.0
    # # pip install prometheus-eval
    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/mixed_a_references_11x25_stronger.json \
    # --reference_path /references/mixed_b_references_11x25_stronger.json \
    # --save_dir  /net/nfs.cirrascale/allennlp/xinxil/results_11x25_stronger/2_wo_v${version}_prometheus/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" "Reasoning Over Numerical Data" "Multi-Document Synthesis" "Fact Checking or Attributed QA" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus

    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/mixed_a_references_11x25_stronger.json \
    # --reference_path /references/mixed_b_references_11x25_stronger.json \
    # --save_dir  /net/nfs.cirrascale/allennlp/xinxil/results_11x25_stronger/2_wo_v${version}_prometheus_w_tie/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" "Reasoning Over Numerical Data" "Multi-Document Synthesis" "Fact Checking or Attributed QA" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus\
    # --use_tie


    # # # Use human reference
    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/mixed_a_references_11x25_stronger.json \
    # --reference_path /references/mixed_b_references_11x25_stronger.json \
    # --human_path /references/mixed_human_references_11x25_stronger.json \
    # --save_dir  /net/nfs.cirrascale/allennlp/xinxil/results_11x25_stronger/2_wo_v${version}_prometheus_w_reference/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" "Reasoning Over Numerical Data" "Multi-Document Synthesis" "Fact Checking or Attributed QA" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus \
    # --use_human_reference 


    # python -m eval.prometheus.run_humanif_eval \
    # --output_path /references/mixed_a_references_11x25_stronger.json \
    # --reference_path /references/mixed_b_references_11x25_stronger.json \
    # --human_path /references/mixed_human_references_11x25_stronger.json \
    # --save_dir  /net/nfs.cirrascale/allennlp/xinxil/results_11x25_stronger/2_wo_v${version}_prometheus_w_tie_w_reference/ \
    # --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" "Reasoning Over Numerical Data" "Multi-Document Synthesis" "Fact Checking or Attributed QA" \
    # --config_dir eval/alpaca_farm/configs \
    # --config_name 2_wo_v${version}_prometheus\
    # --use_tie \
    # --use_human_reference

done

for version in 2_w_v0 ; 
do
    for model in llama3.1 ; 
    do
        python -m eval.alpaca_farm.run_humanif_eval \
        --limit_eval_size 200 \
        --use_vllm \
        --output_path /references/mixed_b_references_11x25_stronger.json \
        --reference_path /references/mixed_a_references_11x25_stronger.json \
        --save_dir  /net/nfs.cirrascale/allennlp/xinxil/results_11x25_stronger/${version}_${model} \
        --use_chat_format \
        --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" "Reasoning Over Numerical Data" "Multi-Document Synthesis" "Fact Checking or Attributed QA"\
        --config_dir eval/alpaca_farm/configs \
        --config_name ${version}_${model} \
        --embed_human_response \
        --human_path /references/mixed_human_references_11x25_stronger.json
        # --human_path /net/nfs.cirrascale/allennlp/xinxil/mixed_human_references_11x25.json 
        # --save_dir  /net/nfs.cirrascale/allennlp/xinxil/results_11x25/${version}_${model} \
    done
done

for version in 2_wo_v0 2_w_v0 ;
do
    for model in llama3-70b llama3.1 llama3.1-70b llama3.1-405b ;
    do
        python3 scripts/create_config.py --template_name ${version} --model $model
    done
done

# Automatic
python -m eval.automatic.run_automatic \
    --output_path /references/mixed_a_references_11x25_stronger.json \
    --reference_path /references/mixed_b_references_11x25_stronger.json \
    --human_path /references/mixed_human_references_11x25_stronger.json \
    --save_dir  /net/nfs.cirrascale/allennlp/xinxil/results_11x25_stronger/ \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" "Reasoning Over Numerical Data" "Multi-Document Synthesis" "Fact Checking or Attributed QA" 

# Baselines
for type in longer shorter random_two random_three more_lines less_lines ; 
do
    python3 -m eval.baselines.run_baselines \
        --output_path /references/mixed_a_references_11x25_stronger.json \
        --reference_path /references/mixed_b_references_11x25_stronger.json \
        --save_dir  /net/nfs.cirrascale/allennlp/xinxil/results_11x25_stronger/${type} \
        --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" "Reasoning Over Numerical Data" "Multi-Document Synthesis" "Fact Checking or Attributed QA" \
        --type ${type}
done

# Analysis
python3 scripts/compare_disagreement_direct_human_annotation_11x25.py \
    --nr_category "Brainstorm" "Open QA" "Closed QA" "Extract" "Generation" "Rewrite" "Summarize" "Classify" "Reasoning Over Numerical Data" "Multi-Document Synthesis" "Fact Checking or Attributed QA" \
    --results_dir results_11x25_stronger \
    --models 2_wo_v0_gpt4 2_w_v0_gpt4 2_wo_v0_llama3 2_w_v0_llama3 2_wo_v0_prometheus 2_wo_v0_prometheus_w_reference 2_wo_v0_prometheus_w_tie 2_wo_v0_prometheus_w_tie_w_reference BERTScore rouge-1 rouge-2 rouge-l longer shorter random_two random_three more_lines less_lines \
    --preference_keys preference preference preference preference  preference preference preference preference preference preference preference preference preference preference preference preference preference preference \
    --annotation_file_path tmp/human_annotations_11x25_stronger.json  

# 
for version in 2_w_v0 ; 
    for model in llama3.1; 
    do
        python3 scripts/submit_through_interactive.py \
        --name humanif_annotation_${model} \
        --command "beaker session create --budget ai2/oe-adapt --gpus 1 --workspace ai2/pradeepd-open-instruct --image beaker://alrope/xinxil_open_instruct --mount beaker://alrope/references=/references --mount beaker://alrope/dataset=/dataset --mount beaker://alrope/xinxil-Meta-Llama-3.1-8B-Instruct=/model --no-update-default-image --secret-env OPENAI_API_KEY=openai_api_key --secret-env HF_TOKEN=huggingface_token" \
        --argument "python -m eval.alpaca_farm.run_humanif_eval --limit_eval_size 200 --use_vllm --output_path /references/mixed_b_references_11x25_stronger.json --reference_path /references/mixed_a_references_11x25_stronger.json --save_dir  /output --use_chat_format --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data' --config_dir eval/alpaca_farm/configs --config_name 2_wo_v0_llama3.1 && python -m eval.alpaca_farm.run_humanif_eval --limit_eval_size 200 --use_vllm --output_path /references/mixed_b_references_11x25_stronger.json --reference_path /references/mixed_a_references_11x25_stronger.json --save_dir /output --use_chat_format --chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template --nr_category 'Brainstorm' 'Open QA' 'Closed QA' 'Extract' 'Generation' 'Rewrite' 'Summarize' 'Classify' 'Fact Checking or Attributed QA' 'Multi-Document Synthesis' 'Reasoning Over Numerical Data' --config_dir eval/alpaca_farm/configs --config_name 2_w_v0_llama3.1 --embed_human_response --human_path /references/mixed_human_references_11x25_stronger.json" 
    done
done