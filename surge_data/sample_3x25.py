import os
import json
import argparse
import logging
import random
from collections import defaultdict
import csv
import uuid

def main(args):
    random.seed(42)

    prompt_file_path = os.path.join(args.prompt_file_path)
    data = json.load(open(prompt_file_path, 'r'))

    final_output = defaultdict(list)
    for category in args.nr_category:
        final_prompts = [dp['Prompt 1'] for dp in data if dp['category'] == category][:args.n_per_cat]
        for prompt in final_prompts:
            final_output[category].append({
                "instruction": prompt
            })
        
    json.dump(final_output, open(os.path.join(args.save_dir, "raw_prompts_3x25.json"), 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nr_category",
        type=str,
        choices=["Fact Checking / Attributed QA", "Multi-Document Synthesis", "Reasoning Over Numerical Data"],
        nargs="+",
        default=["Fact Checking / Attributed QA", "Multi-Document Synthesis", "Reasoning Over Numerical Data"],
        help="Categories in the No Robots dataset to include. If not specified, all categories will be used"
    )

    parser.add_argument(
        "--prompt_file_path",
        type=str,
        default=None,
        help="Path to the file that contains prompts."
    )

    parser.add_argument(
        "--n_per_cat",
        type=int,
        default=25,
        help="Number of sample per category."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="tmp/references",
        help="Path to the save dir."
    )

    args = parser.parse_args()
    main(args)
