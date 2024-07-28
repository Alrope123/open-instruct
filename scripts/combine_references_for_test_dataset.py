import os
import json
import argparse
import logging
import random
from collections import defaultdict

def similar(str1, str2):
    str1 = str1 + ' ' * (len(str2) - len(str1))
    str2 = str2 + ' ' * (len(str1) - len(str2))
    return sum(1 if i == j else 0
        for i, j in zip(str1, str2)) / float(len(str1))

def main(args):
    random.seed(42)

    prompt_file_path = args.prompt_file_path
    prediction_dir = args.prediction_dir
    model_name = args.model_name
    save_dir = args.save_dir
    nr_category = args.nr_category 
            

    prompts = {}
    if prompt_file_path:
        assert os.path.exists(prompt_file_path)
        with open(prompt_file_path, mode ='r') as f:    
            for line in f:
                dp = json.loads(line)
                cat = dp["category"]
                if cat not in prompts:
                    prompts[cat] = []
                prompts[cat].append(dp['prompt'])


    references = {}
    for category in nr_category:
        references[category] = []
        with open(os.path.join(prediction_dir, f"{model_name}-{category}-greedy-long-output.json"), "r") as f:
            for line in f:
                dp = references[category].append(json.loads(line))

        cat_prompts = prompts[category]
        cat_references = []
        for prompt in cat_prompts:
            found = False
            for reference in references[category]:
                if similar(reference['instruction'], prompt.strip()) > 0.9:
                    cat_references.append(reference)
                    found = True
            assert found, (category, prompt)
        references[category] = cat_references


    json.dump(references, open(os.path.join(save_dir, f"{model_name}_references_test_dataset.json"), 'w'))

    

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
