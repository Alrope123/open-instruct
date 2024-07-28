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
                if similar(annotation['instruction'], must_have_prompt.strip()) > 0.99:
                    assert not found
                    found = True
            assert found, (category, must_have_prompt)
        additional_pool = [prompt for prompt in annotation_prompts if prompt not in must_have_prompts]
        assert len(additional_pool) > needed_prompt_num, (len(additional_pool), needed_prompt_num)
        final_prompts = must_have_prompts + random.choices(additional_pool, k=needed_prompt_num)
        assert len(final_prompts) == n_per_cat, (len(final_prompts), n_per_cat)
        print(len(final_prompts))
        print(len(final_output))

        for i, final_prompt in enumerate(final_prompts):
            found = False
            for annotation in annotations:
                if similar(annotation['instruction'], final_prompt.strip()) > 0.99:
                    assert not found, (annotation['instruction'], final_prompt.strip())
                    found = True
                    response_1 = annotation['output_1']
                    response_2 = annotation['output_2']
                    assert annotation['instruction'] in human_preferences
                    response_human = human_preferences[annotation['instruction']]
                    final_output.append({
                        "category": category,
                        "text": annotation['instruction'],
                        "comparison_id": uuid.uuid4().hex,
                        "index": i,
                        "completion_a": response_1,
                        "completion_b": response_2,
                        "model_a": "chatgpt",
                        "model_b": "llama",
                    })
                    final_output.append({
                        "category": category,
                        "text": annotation['instruction'],
                        "comparison_id": uuid.uuid4().hex,
                        "index": i,
                        "completion_a": response_human,
                        "completion_b": response_1,
                        "model_a": "human",
                        "model_b": "chatgpt",
                    })
                    final_output.append({
                        "category": category,
                        "text": annotation['instruction'],
                        "comparison_id": uuid.uuid4().hex,
                        "index": i,
                        "completion_a": response_human,
                        "completion_b": response_2,
                        "model_a": "human",
                        "model_b": "llama",
                    })
            assert found, (category, must_have_prompt)

        # for i, annotation in enumerate(annotations):
        #     if any([similar(annotation['instruction'].strip(), final_prompt.strip()) > 0.9 for final_prompt in final_prompts]):
        #         response_1 = annotation['output_1']
        #         response_2 = annotation['output_2']
        #         assert annotation['instruction'] in human_preferences
        #         response_human = human_preferences[annotation['instruction']]
        #         final_output.append({
        #             "category": category,
        #             "text": annotation['instruction'],
        #             "comparison_id": uuid.uuid4().hex,
        #             "index": i,
        #             "completion_a": response_1,
        #             "completion_b": response_2,
        #             "model_a": "chatgpt",
        #             "model_b": "llama",
        #         })
        #         final_output.append({
        #             "category": category,
        #             "text": annotation['instruction'],
        #             "comparison_id": uuid.uuid4().hex,
        #             "index": i,
        #             "completion_a": response_human,
        #             "completion_b": response_1,
        #             "model_a": "human",
        #             "model_b": "chatgpt",
        #         })
        #         final_output.append({
        #             "category": category,
        #             "text": annotation['instruction'],
        #             "comparison_id": uuid.uuid4().hex,
        #             "index": i,
        #             "completion_a": response_human,
        #             "completion_b": response_2,
        #             "model_a": "human",
        #             "model_b": "llama",
        #         })

    random.shuffle(final_output)
    with open(os.path.join(save_dir, "no_robot_samples_for_human_annotation.jsonl"), 'w') as f:
        for output in final_output:
            f.write(json.dumps(output))
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
