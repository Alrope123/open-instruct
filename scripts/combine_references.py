import os
import json
import argparse
import logging
import random
from collections import defaultdict

def main(args):
    random.seed(42)

    prediction_dir = args.prediction_dir
    model_name = args.model_name
    save_dir = args.save_dir
    nr_category = args.nr_category

    references = {}
    for category in nr_category:
        references[category] = []
        with open(os.path.join(prediction_dir, f"{model_name}-{category}-greedy-long-output.json"), "r") as f:
            for line in f:     
                references[category].append(json.loads(line))

    json.dump(references, open(os.path.join(save_dir, f"{model_name}_references.json"), 'w'))

    

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
        "--prediction_dir",
        type=str,
        default=None,
        required=True,
        help="Path to the dir that contains the prediction file."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo",
        help="Path to the dir that contains the prediction file."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        required="./",
        help="Path to the save dir."
    )

    args = parser.parse_args()
    main(args)