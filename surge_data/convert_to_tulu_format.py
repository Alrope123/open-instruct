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

    final_output = []
    for dp in data:
        final_output.append({
            "prompt": dp['Prompt 1'],
            "category": dp["category"],
            "messages": [
                {
                    "content": dp["Prompt 1"],
                    "role": "user"
                },
                {
                    "content": dp["Response 1"],
                    "role": "assistant"
                }
            ],
            "prompt_id": dp["task_id"]
        })
    

    with open(os.path.join(args.save_dir, "no_robots_test_data_surge.json"), 'w') as f:
        for dp in final_output:
            f.write(json.dumps(dp))
            f.write("\n")


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
        default="/home/xinxil/open-instruct/surge_data/AI2_SFT_Science_Research_Categories_Cleaned.json",
        help="Path to the file that contains prompts."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="tmp/data",
        help="Path to the save dir."
    )

    args = parser.parse_args()
    main(args)
