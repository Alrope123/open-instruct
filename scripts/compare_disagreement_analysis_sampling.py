import os
import json
import argparse
import logging
import random
from collections import defaultdict, Counter
import csv
from itertools import groupby
from operator import itemgetter
import copy
import math

def most_frequent_element(lst):
    if not lst:
        return None, 0
    counter = Counter(lst)
    most_common_element, frequency = counter.most_common(1)[0]
    if frequency < 1:
        print(lst)
        assert False
    return most_common_element, frequency

def main(args):
    random.seed(42)

    nr_category = args.nr_category
    results_dir = args.results_dir
    names = args.names

    n_samples = 10

    for name in names:
        annotations2 = []
        for category in nr_category:
            category_name = category.lower().replace(' ', '_')
            tmp_preference = []
            for i in range(n_samples):
                cur_annotations = json.load(open(os.path.join(results_dir, name, str(i), category_name, "alpaca_eval_annotator_cache.json"), 'r'))
                tmp_preference.append([cur_a["preference"] for cur_a in cur_annotations])
            annotations2.extend([[tmp_preference[j][i] for j in range(n_samples)] for i in range(len(tmp_preference[0]))])

        
        annotations = []
        for annot2 in annotations2:
            prediction, frequency = most_frequent_element(annot2)
            annotations.append({"preference": prediction, "agreement": frequency / n_samples})
        
        output = defaultdict(list)

    
        output['tie_rate'] = [int(cur_a["preference"]) == 0 if cur_a["preference"] is not None else False for cur_a in annotations]
        output['tie_agreement'] = [cur_a["agreement"] if cur_a["preference"] is not None else False for cur_a in annotations if cur_a["preference"] is not None and int(cur_a["preference"]) == 0]
        output['response_1_rate'] = [int(cur_a["preference"]) == 1 if cur_a["preference"] is not None else False for cur_a in annotations]
        output['response_1_agreement'] = [cur_a["agreement"] if cur_a["preference"] is not None else False for cur_a in annotations if cur_a["preference"] is not None and int(cur_a["preference"]) == 1]
        output['response_2_rate'] = [int(cur_a["preference"]) == 2 if cur_a["preference"] is not None else False for cur_a in annotations]
        output['response_2_agreement'] = [cur_a["agreement"] if cur_a["preference"] is not None else False for cur_a in annotations if cur_a["preference"] is not None and int(cur_a["preference"]) == 2]

        for k, v in output.items():
            output[k] = f"{round((sum(v) / len(v) * 100) if len(v) > 0 else 0, 2)}%"

        print(name)
        print(json.dumps(output, indent=4))
            

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
