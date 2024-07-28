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

    # if args.config_name is not None:
    #     assert args.config_dir is not None

    if args.embed_human_response:
        assert args.reference_path is not None   

    if args.output_path is not None:
        assert args.model_name_or_path is None

    logging.info("loading data and model...")

    raw_text_prompts = defaultdict(list)  # category -> list of example dicts
    human_references = defaultdict(list)  # category -> list of example dicts
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None

    if args.output_path is None or args.reference_path is None or args.embed_human_response:
        # Load local or huggingface no_robot dataset
        if args.dataset.endswith('.json'):
            no_robots_data = datasets.load_dataset("json", data_files=args.dataset)["train"]
        elif args.dataset.endswith('.csv'):
            no_robots_data = datasets.load_dataset("csv", data_files=args.dataset)["train"]
        else:
            no_robots_data = datasets.load_dataset(args.dataset)["train"]
        for example in no_robots_data:
            category = example["category"]
            if args.nr_category and category not in args.nr_category:
                continue
            if len(example["messages"]) > 2:
                # Happens only in the chat category
                continue
            if args.limit_eval_size is not None and category in raw_text_prompts and len(raw_text_prompts[category]) >= args.limit_eval_size:
                continue
            raw_text_prompts[category].append(example["prompt"])
            human_references[category].append(
                {
                    "instruction": example["prompt"],
                    "output": example["messages"][1]["content"],
                    "generator": "human",
                    "dataset": f"no_robots_{category}"
                }
            )

    if args.model_name_or_path is not None:
        # we always load the tokenizer for vllm or hf models
        tokenizer = load_hf_tokenizer(
                model_name_or_path=args.model_name_or_path,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                use_fast_tokenizer=not args.use_slow_tokenizer,
            )
        if args.use_vllm:
            vllm_model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path is not None else args.model_name_or_path,
                tensor_parallel_size=torch.cuda.device_count(),
            )
            sampling_params = vllm.SamplingParams(
                temperature=0,  # greedy decoding
                max_tokens=args.max_new_tokens,
            )
        else:
            model = load_hf_lm(
                model_name_or_path=args.model_name_or_path,
                load_in_8bit=args.load_in_8bit,
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq,
            )
            # modify tokenizer if required
            from transformers import GPTNeoXForCausalLM, OPTForCausalLM
            if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
                tokenizer.model_max_length = model.config.max_position_embeddings
                print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))

        if args.use_chat_format:
            prompts = {}
            for category, category_prompts in raw_text_prompts.items():
                formatted_prompts = []
                for prompt in category_prompts:
                    messages = [{"role": "user", "content": prompt}]
                    prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
                    formatted_prompts.append(prompt)
                prompts[category] = formatted_prompts
                logging.info("Formatting prompts in chat format.")
                logging.info(f"Example: {prompts[category][0]}")
        else:
            prompts = dict(raw_text_prompts)
    elif args.output_path is None:
        openai_client = OpenAI()
        prompts = dict(raw_text_prompts)
    else:
        prompts = json.load(open(args.output_path))

    logging.info("Stats:")
    for category, category_prompts in prompts.items():
        logging.info(f"{category}\t{len(category_prompts)}")

    if args.reference_path is not None:
        logging.info(f"Using references from {args.reference_path}")
        # Assuming the file is json representing a dict where keys correspond to No Robots categories
        references = json.load(open(args.reference_path))
    else:
        logging.info("Using human-written references..")
        references = human_references

    print(len(prompts))
    metrics_dict = {}

    for category, category_prompts in prompts.items():
        logging.info(f"Running inference on category: {category}")
        if args.model_name_or_path is not None:
            if args.use_vllm:
                category_outputs = vllm_model.generate(category_prompts, sampling_params)
                category_outputs = [it.outputs[0].text for it in category_outputs]
            else:
                category_outputs = generate_completions(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=category_prompts,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0,
                    batch_size=args.eval_batch_size if args.eval_batch_size else 1,
                )
        elif args.output_path is None:
            assert not args.use_chat_format
            category_outputs = []
            for prompt in category_prompts:
                response = openai_client.chat.completions.create(
                    model=args.openai_engine,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.max_new_tokens,
                    temperature=0,
                )
                category_outputs.append(response.choices[0].message.content)

        if args.output_path is None:
            model_name = os.path.basename(os.path.normpath(args.model_name_or_path)) if args.model_name_or_path is not None else args.openai_engine
            model_results = []
            with open(os.path.join(args.save_dir, f"{model_name}-{category}-greedy-long-output.json"), "w") as fout:
                for prompt, output in zip(raw_text_prompts[category], category_outputs):
                    example = {
                        "instruction": prompt,
                        "output": output,
                        "generator": f"{model_name}-greedy-long",
                        "dataset": f"no_robots_{category}"
                    }
                    fout.write(json.dumps(example) + "\n")
                    model_results.append(example)
        else:
            model_results = category_prompts

        category_references = references[category]
        if args.embed_human_response:
            category_human_references = human_references[category]
        else:
            category_human_references = None
        logging.info(f"Running alpaca eval on category: {category}")
        output_path = os.path.join(args.save_dir, category.lower().replace(" ", "_"))
        os.makedirs(output_path, exist_ok=True)
        logging.info(os.environ.get('HF_TOKEN'))
        df_leaderboard, _ = alpaca_farm_evaluate(
            model_outputs=model_results,
            reference_outputs=category_references,
            human_outputs=category_human_references,
            annotators_config=args.config_name,
            output_path=output_path,
            is_return_instead_of_print=True,
            caching_path=os.path.join(output_path, "alpaca_eval_annotator_cache.json"),
            precomputed_leaderboard=None,
            is_cache_leaderboard=False,
            base_dir=args.config_dir,
            multiple_answer = args.multiple_answer,
            # annotation_kwargs = {"cache_dir": "/model", "token": os.environ.get('HF_TOKEN')},
            # annotation_kwargs = {"token": os.environ.get('HF_TOKEN')},
            output_keys=("output_1", "output_2", "output_human") if args.embed_human_response else ("output_1", "output_2")
        )
        print(df_leaderboard.to_string(float_format="%.2f"))
        for key, value in df_leaderboard.to_dict().items():
            metrics_dict[f"{category}_{key}"] = value

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
        "--output_path",
        type=str,
        default=None,
        help="Path to the file that contains outputs. If none is provided, will use model to generate."
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
    parser.add_argument(
        "--multiple_answer",
        action="store_true",
        help="If given, we will embed human response into the prompt."
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is not None) or (args.openai_engine is not None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
