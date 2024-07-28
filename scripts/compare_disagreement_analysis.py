import os
import json
import argparse
import logging
import random
from collections import defaultdict
import csv
from itertools import groupby
from operator import itemgetter
import copy
import math

def main(args):
    random.seed(42)

    nr_category = args.nr_category
    results_dir = args.results_dir
    names = args.names

    for name in names:
        annotations = defaultdict(list)
        for category in nr_category:
            category_name = category.lower().replace(' ', '_')
            cur_annotations = json.load(open(os.path.join(results_dir, f"{name}", category_name, "alpaca_eval_annotator_cache.json"), 'r'))
            annotations['tie_rate'].extend([int(cur_a["preference"]) == 0 if cur_a["preference"] is not None else False for cur_a in cur_annotations])
            annotations['response_1_win_rate'].extend([int(cur_a["preference"]) == 1 if cur_a["preference"] is not None else False for cur_a in cur_annotations])
            annotations['response_2_win_rate'].extend([int(cur_a["preference"]) == 2 if cur_a["preference"] is not None else False for cur_a in cur_annotations])
        
        for k, v in annotations.items():
            annotations[k] = f"{round(sum(v) / len(v) * 100, 2)}%"
        
        print(name)
        print(json.dumps(annotations, indent=4))
            

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
        "--results_dir",
        type=str,
        default=None,
        required=True,
        help="Path to the dir that contains the prediction file."
    )


    parser.add_argument(
        "--names",
        type=str,
        default=None,
        nargs="+",
        help="Path to the dir that contains the prediction file."
    )

    args = parser.parse_args()
    main(args)
