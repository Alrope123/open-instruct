import os
import json
import argparse
import logging
import random
from collections import defaultdict
import csv
import uuid

def similar(str1, str2):
    """
    Determines if two strings are similar based on the Jaccard similarity.

    Args:
        str1 (str): The first string.
        str2 (str): The second string.

    Returns:
        float: Jaccard similarity index.
    """
    set1 = set(str1)
    set2 = set(str2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def main(args):
    random.seed(42)

    # prediction_dir = os.path.join(args.prediction_dir, f"chat-7b-against-gpt3.5-v{version_name}-prompt-test-set-new")
    # prompt_file_path = os.path.join(args.prompt_file_path, f"chat-7b-against-gpt3.5-v{version_name}-prompt-test-set-new")
    prompt_file_path = os.path.join(args.prompt_file_path)
    prediction_dir = args.prediction_dir
    save_dir = args.save_dir
    n_per_cat = args.n_per_cat
    nr_category = args.nr_category

    preference_map = {1: 2, 2:1, 0:0}

    prompts = {}
    if prompt_file_path:
        assert os.path.exists(prompt_file_path), prompt_file_path
        with open(prompt_file_path, mode ='r') as f:    
            csvFile = csv.DictReader(f)
            for line in csvFile:
                cat = line["Task"]
                if cat not in prompts:
                    prompts[cat] = []
                prompts[cat].append(line['Instruction'])

    final_output = defaultdict(list)
    for category in nr_category:
        category_name = category.lower().replace(' ', '_')
        annotations = json.load(open(os.path.join(prediction_dir, category_name, "alpaca_eval_annotator_cache.json"), 'r'))

        annotation_prompts = [annotation['instruction'] for annotation in annotations]

        must_have_prompts = prompts[category]
        needed_prompt_num = n_per_cat - len(must_have_prompts)
        # Check if all must have prompt are in the big data
        for must_have_prompt in must_have_prompts:
            found = False
            for annotation in annotations:
                # if must_have_prompt == "Who was the lead singer of Queen?\n" and annotation['instruction'] == "Who was the lead singer of Queen?\n":
                #     assert  similar(annotation['instruction'], must_have_prompt.strip()) > 0.99
                if similar(annotation['instruction'], must_have_prompt) > 0.99:
                    assert not found
                    found = True
            assert found, (category, must_have_prompt)
        additional_pool = [prompt for prompt in annotation_prompts if prompt not in must_have_prompts]
        assert len(additional_pool) > needed_prompt_num, (len(additional_pool), needed_prompt_num)
        random.shuffle(additional_pool)
        additional_prompts = additional_pool[:needed_prompt_num]
        final_prompts = must_have_prompts + additional_prompts
        assert len(final_prompts) == n_per_cat, (len(final_prompts), n_per_cat)
        print(len(final_prompts))
        print(len(final_output))

        for prompt in final_prompts:
            final_output[category].append({
                "instruction": prompt
            })
        
    json.dump(final_output, open(os.path.join(save_dir, "raw_prompts_8x25.json"), 'w'))


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
        default="./",
        help="Path to the save dir."
    )

    args = parser.parse_args()
    main(args)
