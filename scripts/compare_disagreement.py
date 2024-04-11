import os
import json
import argparse
import logging
import random
from collections import defaultdict
import csv
from itertools import groupby
from operator import itemgetter

def group_by(data, key_to_group_by):
    total_disagreement = {}
    for disagreement_type, _ in data[0]['disagreements'].items():
        total_disagreement[disagreement_type] = sum([item['disagreements'][disagreement_type] for item in data])

    # Sort the data by the key
    sorted_data = sorted(data, key=itemgetter(key_to_group_by))

    # Group the data
    grouped_data = {}
    for key, group in groupby(sorted_data, key=itemgetter(key_to_group_by)):
        grouped_data[key] = list(group)

    # Display the grouped data
    results = {}
    for key, group in grouped_data.items():
        group_result = {}
        for disagreement_type, _ in group[0]['disagreements'].items():
            group_result[disagreement_type] = f"{sum([item['disagreements'][disagreement_type] for item in group]) / total_disagreement[disagreement_type] * 100}%" if total_disagreement[disagreement_type] != 0 else "0"
        results[key] = group_result
    print(results)


def main(args):
    random.seed(42)
    annotation_file_path = args.annotation_file_path
    # save_dir = args.save_dir
    output_name = "Output 1"
    reference_name = "Output 2" # "Human output"
    gpt4_judgement_name = "GPT4 Judgement"
    # judgement1_name = "Shane's Judgement"
    # judgement2_name = "Pradeep's Judgement"
    judgement1_name = "Your Judgement (Guideline: Instructions For Annotation)"
    judgement2_name = "Your Judgement (Guideline: Instructions For Annotation)"
    shuffled_name = "Shuffled"
    preference_map = {1: 2, 2:1, 0:0}

    annotations = []
    with open(annotation_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            annotations.append(row)

    for annotation in annotations:
        # unshuffle the outputs
        shuffled = bool(annotation[shuffled_name])
        annotation[output_name], annotation[reference_name] = (annotation[output_name], annotation[reference_name]) if not shuffled else (annotation[reference_name], annotation[output_name])
        annotation[judgement1_name] = int(annotation[judgement1_name]) if not shuffled else preference_map[int(annotation[judgement1_name])]
        annotation[judgement2_name] = int(annotation[judgement2_name]) if not shuffled else preference_map[int(annotation[judgement2_name])]

        annotation[gpt4_judgement_name] = int(annotation[gpt4_judgement_name])

        annotation['disagreements'] = {}
        annotation["disagreements"]['w_model'] = annotation[gpt4_judgement_name] != annotation[judgement1_name]
        annotation["disagreements"]['w_human'] = annotation[judgement1_name] != annotation[judgement2_name]


    print("Total results:")
    for disagreement_type, _ in annotations[0]['disagreements'].items():
        rate = sum([item['disagreements'][disagreement_type] for item in annotations]) / len(annotations)
        print(f"Disagreement {disagreement_type}: {rate * 100}%")
    print()

    for key in ["Task", judgement1_name]:
        print(f"Results grouped by {key}:")
        group_by(annotations, key)
        print()
    
    length_preferences = {}
    keys = [judgement1_name, judgement2_name, gpt4_judgement_name]
    for key in keys:
        length_preferences[key] = []
        for annotation in annotations:
            length_judgement = 2 if len(annotation[reference_name]) < len(annotation[output_name]) else 1
            length_preferences[key].append(str(annotation[key]) == str(length_judgement))
        length_preferences[key] = sum(length_preferences[key]) / len(length_preferences[key])
    print("Length preference:")
    print(length_preferences)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotation_file_path",
        type=str,
        default=None,
        help="Path to the file that contains annotations."
    )

    args = parser.parse_args()
    main(args)
