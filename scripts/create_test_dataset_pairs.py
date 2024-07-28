import os
import json
import argparse
import logging
import datasets
from collections import defaultdict
import random
import csv

def similar(str1, str2):
    str1 = str1 + ' ' * (len(str2) - len(str1))
    str2 = str2 + ' ' * (len(str1) - len(str2))
    return sum(1 if i == j else 0
               for i, j in zip(str1, str2)) / float(len(str1))

def main(args):
    random.seed(42)

    prompt_file_path = args.prompt_file_path
    save_dir = args.save_dir
    nr_category = args.nr_category
    

    prompts = []
    if prompt_file_path:
        assert os.path.exists(prompt_file_path)
        with open(prompt_file_path, mode ='r') as f:    
            csvFile = csv.DictReader(f)
            for line in csvFile:
                shuffled = True if line["Shuffled"] == "TRUE" else False
                output1 = line["Output 1"] if not shuffled else line["Output 2"]
                output2 = line["Output 2"] if not shuffled else line["Output 1"] 
                prompts.append({
                    "instruction": line["Instruction"],
                    "output_1": output1,
                    "output_2": output2,
                    "generator": "unknown",
                    "dataset": "test_no_robots"
            })

    with open(os.path.join(save_dir, "no_robots_test_data_pairs.json"), 'w') as f:
        json.dump(prompts, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nr_category",
        type=str,
        choices=["Generation", "Open QA", "Brainstorm", "Chat", "Rewrite", "Summarize",
                 "Coding", "Classify", "Closed QA", "Extract"],
        nargs="+",
        help="Categories in the No Robots dataset to include. If not specified, all categories will be used"
    )

    parser.add_argument(
        "--prompt_file_path",
        type=str,
        default=None,
        help="Path to the file that contains prompts."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./",
        help="Path to the save dir."
    )

    args = parser.parse_args()
    main(args)
