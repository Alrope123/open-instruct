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

    prediction_dir = args.prediction_dir
    prompt_file_path = args.prompt_file_path
    model_name = args.model_name
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
                prompts[cat].append((line['Instruction'], line["Output 1"]))

    examples = []
    for category in nr_category:
        category_name = category.lower().replace(' ', '_')
        annotations = json.load(open(os.path.join(prediction_dir, category_name, "alpaca_eval_annotator_cache.json"), 'r'))
        if prompts == {}:
            for i, annotation in enumerate(annotations):
                if i < n_per_cat:
                    examples.append([
                        category, annotation['instruction'], annotation['output_1'], annotation['output_2'], annotation['output_human'] if 'output_human' in annotation else "", int(annotation['preference'])
                    ])
                else:
                    break
        else:
            cat_prompts = prompts[category]
            for prompt, first_output in cat_prompts:
                found = False
                for annotation in annotations:
                    if similar(annotation['instruction'], prompt.strip()) > 0.9:
                        if "Shuffled" in annotation:
                            shuffled = True if annotation["Shuffled"] == "TRUE" else False
                        else:
                            shuffled = bool(random.getrandbits(1))

                        output_1, output_2 = (annotation['output_1'], annotation['output_2']) if not shuffled else (annotation['output_2'], annotation['output_1'])
                        # assert (first_output == output_1 and not shuffled) or (first_output == output_2 and shuffled), (output_1, output_2, first_output, shuffled)
                        preference = int(annotation['preference']) if not shuffled else preference_map[int(annotation['preference'])]
                        examples.append([
                            category, annotation['instruction'], output_1, output_2, annotation['output_human'] if 'output_human' in annotation else "", preference, shuffled
                        ])
                        found = True
                assert found, (category, prompt)

    f = csv.writer(open(os.path.join(save_dir, "examples.csv"), "w+"))
    titles = ["Task", "Instruction", "Output 1", "Output 2", "Output Human", "GPT4 Judgement", "Shuffled"]
    f.writerow(titles)

    assert len(examples) == n_per_cat * len(nr_category), [len(examples), n_per_cat * len(nr_category)]
    for example in examples:
        assert len(example) == len(titles), (example[-1], titles)
        f.writerow(example)
    

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
