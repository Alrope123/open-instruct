import os
import json
import argparse
import logging
import random
from collections import defaultdict
import csv

def similar(str1, str2):
    str1 = str1 + ' ' * (len(str2) - len(str1))
    str2 = str2 + ' ' * (len(str1) - len(str2))
    return sum(1 if i == j else 0
               for i, j in zip(str1, str2)) / float(len(str1))

def main(args):
    random.seed(42)

    version_name = args.version_name
    # prediction_dir = os.path.join(args.prediction_dir, f"chat-7b-against-gpt3.5-v{version_name}-prompt-test-set-new")
    # prompt_file_path = os.path.join(args.prompt_file_path, f"chat-7b-against-gpt3.5-v{version_name}-prompt-test-set-new")
    prompt_file_path = os.path.join(args.prompt_file_path)
    model_name = args.model_name
    prediction_dir = args.prediction_dir
    save_dir = args.save_dir
    n_per_cat = args.n_per_cat
    nr_category = args.nr_category

    preference_map = {1: 2, 2:1, 0:0}
    human_preference_path = args.human_preference_path

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

    human_preferences = {}
    assert os.path.exists(human_preference_path), human_preference_path
    with open(human_preference_path, 'r') as f:
        for line in f:
            dp = json.loads(line)
            human_preferences[dp['prompt']] = dp["messages"][1]["content"]

    final_output = []
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
                if similar(annotation['instruction'], must_have_prompt.strip()) > 0.9:
                    found = True
            assert found, (category, must_have_prompt)
        additional_pool = [prompt for prompt in annotation_prompts if prompt not in must_have_prompts]
        assert len(additional_pool) > needed_prompt_num, (len(additional_pool), needed_prompt_num)
        final_prompts = must_have_prompts + random.choices(additional_pool, k=needed_prompt_num)
        assert len(final_prompts) == n_per_cat, (len(final_prompts), n_per_cat)

        for i, annotation in enumerate(annotations):
            if annotation['instruction'] in final_prompts:
                response_1 = annotations['output_1']
                response_2 = annotations['output_2']
                assert annotation['instruction'] in human_preferences
                response_human = human_preferences[annotation['instruction']]
                final_output.add({
                    "category": category,
                    "messages": [{
                        "role": "user",
                        "content": annotation['instruction']
                    }],
                    "id": f"instance_{i}_{category}_from_chatgpt+llama",
                    "completions": [
                        response_1,
                        response_2
                    ],
                    "model": [
                        "chatgpt",
                        "llama"
                    ],
                })
                final_output.add({
                    "category": category,
                    "messages": [{
                        "role": "user",
                        "content": annotation['instruction']
                    }],
                    "id": f"instance_{i}_{category}_from_human+chatgpt",
                    "completions": [
                        response_human,
                        response_1
                    ],
                    "model": [
                        "human",
                        "chatgpt"
                    ],
                })
                final_output.add({
                    "category": category,
                    "messages": [{
                        "role": "user",
                        "content": annotation['instruction']
                    }],
                    "id": f"instance_{i}_{category}_from_human+llama",
                    "completions": [
                        response_human,
                        response_2
                    ],
                    "model": [
                        "human",
                        "llama"
                    ],
                })

    random.shuffle(final_output)
    with open(os.path.join(save_dir, "no_robot_samples_for_human_annotation.jsonl"), 'w') as f:
        for output in final_output:
            f.write(output)
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
        "--human_preference_path",
        type=str,
        default=None,
        help="Path to the file that contains prompts."
    )

    parser.add_argument(
        "--n_per_cat",
        type=int,
        default=20,
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
