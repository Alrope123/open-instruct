import os
import json
import argparse
import random
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    os.environ['TRANSFORMERS_CACHE'] = f"{args.cache_dir}/models"
    os.environ['HF_HOME'] = args.cache_dir
    print("Cacheing into: " + args.cache_dir)

    model =  AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        token=os.getenv("HF_TOKEN", None),
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=os.getenv("HF_TOKEN", None))

    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
                
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--cache_dir",
        type=str,
        default="/net/nfs.cirrascale/allennlp/xinxil/",
        help="We will use the directory under configuration directory as the file that contains the annotator configuration file.",
    )
    args = parser.parse_args()

    main(args)
