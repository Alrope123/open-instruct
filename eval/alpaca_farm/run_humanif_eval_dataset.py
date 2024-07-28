import os
import json
import argparse
import logging
import random
from collections import defaultdict
import torch
import datasets
import vllm
from alpaca_eval import evaluate as alpaca_farm_evaluate
from openai import OpenAI

from eval.utils import query_openai_chat_model, query_openai_model, generate_completions, dynamic_import_function, load_hf_lm, load_hf_tokenizer


def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    logging.info("loading data and model...")

    raw_text_prompts = []  # category -> list of example dicts
    human_references = []  # category -> list of example dicts
    # Load local or huggingface no_robot dataset
    alpaca_data = datasets.load_dataset("json", data_files=args.dataset)["train"]
    for example in alpaca_data:
       
        raw_text_prompts.append({
                "instruction": example["instruction"],
                "output": example["output_1"],
                "generator": "unknown",
                "dataset": example["dataset"]
            }
        )
        human_references.append(
            {
                "instruction": example["instruction"],
                "output": example["output_2"],
                "generator": "unknown",
                "dataset": example["dataset"]
            }
        )

    logging.info(f"Running alpaca eval")
    output_path = os.path.join(args.save_dir)
    os.makedirs(output_path, exist_ok=True)
    df_leaderboard, _ = alpaca_farm_evaluate(
        model_outputs=raw_text_prompts,
        reference_outputs=human_references,
        human_outputs=None,
        annotators_config=args.config_name,
        output_path=output_path,
        is_return_instead_of_print=True,
        caching_path=os.path.join(output_path, "alpaca_eval_annotator_cache.json"),
        precomputed_leaderboard=None,
        is_cache_leaderboard=False,
        base_dir=args.config_dir,
        output_keys=("output_1", "output_2"),
    )
    print(df_leaderboard.to_string(float_format="%.2f"))
    metrics_dict = {}
    for key, value in df_leaderboard.to_dict().items():
        metrics_dict[f"{key}"] = value

    # save to json
    with open(os.path.join(args.save_dir, f"metrics.json"), "w") as fout:
        json.dump(metrics_dict, fout)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceH4/no_robots",
        help="Path to the reference outputs. If none is provided, will use human-written references."
    )
    parser.add_argument(
        "--nr_category",
        type=str,
        choices=["Generation", "Open QA", "Brainstorm", "Chat", "Rewrite", "Summarize",
                 "Coding", "Classify", "Closed QA", "Extract"],
        nargs="+",
        help="Categories in the No Robots dataset to include. If not specified, all categories will be used"
    )
    parser.add_argument(
        "--limit_eval_size",
        type=int,
        help="Evaluate only on these many prompt-response pairs per category. If not specified, all examples will be used."
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        help="Path to the reference outputs. If none is provided, will use human-written references."
    )
    parser.add_argument(
        "--save_dir",
        type=str, 
        default="results/alpaca_farm")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default="gpt-3.5-turbo",
        help="If specified, we will use the OpenAI API to generate the predictions.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If given, we will use vLLM to generate the predictions - much faster.",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="If specified, we will use the dir as the root directory for annotator configuration.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="alpaca_eval_gpt4",
        help="We will use the directory under configuration directory as the file that contains the annotator configuration file.",
    )
    parser.add_argument(
        "--embed_human_response",
        action="store_true",
        help="If given, we will embed human response into the prompt."
    )

    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is not None) or (args.openai_engine is not None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
