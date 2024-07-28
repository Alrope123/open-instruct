import os
import json
import argparse
import logging
import random
from collections import defaultdict, Counter
import csv
import uuid
import nltk
from nltk.util import ngrams
import re

# Download necessary NLTK data files
nltk.download('punkt')

def check_generation(paragraph):
    if len(paragraph.strip()) <= 5 or len(paragraph.strip().split()) <= 1:
        return False

    # Clean and tokenize the paragraph
    paragraph = re.sub(r'\s+', ' ', paragraph)  # Normalize whitespace
    tokens = nltk.word_tokenize(paragraph.lower())

    for phrase_length, threshold in zip([15, 4, 3], [10, 15, 30]):
        # Generate n-grams
        n_grams = list(ngrams(tokens, phrase_length))

        # Count the frequency of each n-gram
        n_gram_counts = Counter(n_grams)

        # Find repeated phrases
        repeated_phrases = { ' '.join(gram): count for gram, count in n_gram_counts.items() if count >= threshold }

        if len(repeated_phrases) > 0:
            return False

    return True

def write_csv(data, filename):
    # Check if data is not empty
    if len(data) == 0:
        print("No data to write.")
        return

    # Get the keys from the first dictionary in the list as the CSV headers
    headers = data[0].keys()

    # Open the file in write mode
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)

        # Write the headers to the CSV file
        writer.writeheader()

        # Write the data to the CSV file
        for row in data:
            writer.writerow(row)

    print(f"Data has been written to {filename}")


def split_list(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


def post_process(model, output):
    output = output.strip()
    if model.startswith("Meta-Llama-3"):
        beginning_text = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        assert output.startswith(beginning_text)
        output = output[len(beginning_text):]
    return output
    

def main(args):
    random.seed(2024)

    models = args.models
    prediction_dir = args.prediction_dir
    save_dir = args.save_dir
    nr_category = args.nr_category

    model_outputs = {}
    for category in nr_category:
        category_name = category.lower().replace(' ', '_')
        for model in models:
            model_outputs[model] = json.load(open(os.path.join(prediction_dir, f"{model}_references_8x25.json"), 'r'))

    instruction_and_outputs = defaultdict(list)
    for category in nr_category:
        first_model_categorical_annotations = model_outputs[list(model_outputs.keys())[0]][category]
        for i in range(len(first_model_categorical_annotations)):
            assert all([annotations[category][i]['instruction'] == first_model_categorical_annotations[i]['instruction'] or first_model_categorical_annotations[i]['instruction'] in annotations[category][i]['instruction'] for _, annotations in model_outputs.items()]), first_model_categorical_annotations[i]['instruction']
            dp = {
                'instruction': first_model_categorical_annotations[i]['instruction'],
                'outputs': {model: post_process(model, annotations[category][i]['output']) for model, annotations in model_outputs.items()}
            }
            instruction_and_outputs[category].append(dp)

    final_output = []
    model_selections = defaultdict(int)
    for category in nr_category:
        categorical_instruction_and_outputs = instruction_and_outputs[category]
        for i, instruction_and_output in enumerate(categorical_instruction_and_outputs):
            models_and_lengths = [(model, len(output)) for model, output in instruction_and_output['outputs'].items() if check_generation(output)]
            models_and_lengths = sorted(models_and_lengths, key=lambda x: x[1])
            model_groups = split_list(models_and_lengths, 3)
            short_models = [m_and_l[0] for m_and_l in model_groups[0]]
            medium_models = [m_and_l[0] for m_and_l in model_groups[1]]
            long_models = [m_and_l[0] for m_and_l in model_groups[2]]
            comparison_groups = [(short_models, short_models), (long_models, long_models), (medium_models, medium_models), (models, models)]
            compared_models = set()
            for comparision_group in comparison_groups:
                group_a = comparision_group[0]
                group_b = comparision_group[1]
                model_a = ""
                model_b = ""
                while model_a == model_b or ((model_a, model_b) in compared_models or (model_b, model_a) in compared_models):
                    model_a = random.choice(group_a)
                    model_b = random.choice(group_b)
                compared_models.add((model_a, model_b))
                compared_models.add((model_b, model_a))
                final_output.append({
                    "category": category,
                    "text": instruction_and_output['instruction'],
                    "comparison_id": uuid.uuid4().hex,
                    "index": i,
                    "completion_a": instruction_and_output['outputs'][model_a],
                    "completion_b": instruction_and_output['outputs'][model_b],
                    "model_a": model_a,
                    "model_b": model_b,
                })
                model_selections[model_a] += 1
                model_selections[model_b] += 1

    print(json.dumps(model_selections, indent=4))

    random.shuffle(final_output)
    with open(os.path.join(save_dir, "no_robot_samples_for_human_annotation_8x25_stronger.jsonl"), 'w') as f:
        for output in final_output:
            f.write(json.dumps(output))
            f.write('\n')

    write_csv(final_output, os.path.join(save_dir, "no_robot_samples_for_human_annotation_8x25_stronger.csv"))

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
        "--models",
        type=str,
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
        "--save_dir",
        type=str,
        default="./",
        help="Path to the save dir."
    )

    args = parser.parse_args()
    main(args)
