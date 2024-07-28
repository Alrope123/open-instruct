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


def post_process(text):
    text_pieces = text.split("\n\n")
    if "rephrase" in text_pieces[0]:
        if len(text_pieces) > 1:
            text_pieces = text_pieces[1:]
    if "rephrase" in text_pieces[-1]:
        if len(text_pieces) > 1:
            text_pieces = text_pieces[:-1]
    return "\n\n".join(text_pieces)

def main(args):

    type_map = {
        "paraphrase": {
            "template": "Rephrase the following paragraph:\n{}\nPlease only output the paraphrased paragraph.",
            "key": "output"
        },
        "paraphrase_worse": {
            "template": "Paraphrase the following paragraph worse:\n{}\nPlease only output the paraphrased paragraph.",
            "key": "output"
        },
        "rephrase": {
            "template": "Rephrase the following paragraph without changing format and don't lose any details:\n{}\nPlease output the rephrased paragraph only.",
            "key": "output"
        },
        "rephrase_worse": {
            "template": "Rephrase the following paragraph while making some mistakes:\n{}\nPlease output the rephrased paragraph only.",
            "key": "output"
        },
        "longer": {
            "template": "{}. Please give a long response.",
            "key": "instruction"
        },
        "shorter": {
            "template": "{}. Please give a short response.",
            "key": "instruction"
        },
        "greet": {
            "template": "{}. Please greet the user.",
            "key": "instruction"
        },
        "no_greet": {
            "template": "{}. Please do not include any greeting.",
            "key": "instruction"
        }
    }
    if not args.save_dir:
        save_path = args.reference_path.split(".json")[0] + f"_{args.type}.json"
    else:
        save_path = os.path.join(args.save_dir, os.path.basename(args.reference_path.split(".json")[0] + f"_{args.type}.json"))
    print(f"Saving to {save_path}.")

    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    logging.info("loading data and model...")

    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None

    logging.info(f"Using references from {args.reference_path}")
    # Assuming the file is json representing a dict where keys correspond to No Robots categories
    raw_text_prompts = {}
    references = json.load(open(args.reference_path))
    raw_text_prompts = {k: [type_map[args.type]["template"].format(entry[type_map[args.type]["key"]]) for entry in v] for k, v in references.items()}
    print("Example of a paragraphing request:")
    print(raw_text_prompts[list(raw_text_prompts.keys())[0]][0])

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
    else:
        openai_client = OpenAI()
        prompts = dict(raw_text_prompts)

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
        else:
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

        for i, dp in enumerate(references[category]):
            dp["output"] = post_process(category_outputs[i])

    with open(save_path, "w") as f:
        json.dump(references, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--type",
        type=str
    )

    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is not None) or (args.openai_engine is not None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
