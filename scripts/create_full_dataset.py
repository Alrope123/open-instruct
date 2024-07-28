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

    save_dir = args.save_dir
    nr_category = args.nr_category
    
    selected_examples = defaultdict(list)  # category -> list of example dicts
    no_robots_data = datasets.load_dataset("HuggingFaceH4/no_robots")["train"]
    for example in no_robots_data:
        category = example["category"]
        if nr_category and category not in nr_category:
            continue
        if len(example["messages"]) > 2:
            # Happens only in the chat category
            continue
        # if args.limit_eval_size is not None and category in selected_examples and len(selected_examples[category]) >= args.limit_eval_size:
        #     continue
        selected_examples[category].append(example)

    selected_examples_flat = []
    for _, v in selected_examples.items():
        selected_examples_flat.extend(v)


    output_datasets = selected_examples_flat

    with open(os.path.join(save_dir, "no_robots_full_data.json"), 'w') as f:
        for dp in output_datasets:
            f.write(json.dumps(dp))
            f.write('\n')


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
        "--save_dir",
        type=str,
        default="./",
        help="Path to the save dir."
    )

    args = parser.parse_args()
    main(args)
