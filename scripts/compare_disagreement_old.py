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
            group_result[disagreement_type] = f"{sum([item['disagreements'][disagreement_type] for item in group]) / total_disagreement[disagreement_type] * 100}%"
        results[key] = group_result
    print(results)


def main(args):
    random.seed(42)
    annotation_file_path = args.annotation_file_path
    # save_dir = args.save_dir
    reference_name = "GPT-3.5 output" # "Human output"


    annotations = []
    with open(annotation_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            annotations.append(row)

    for annotation in annotations:
        annotation['disagreements'] = {}
        annotation["disagreements"]['w_model_w_tie'] = annotation["GPT4 Judgement"] != annotation["Overall (w/ tie)"]
        annotation["disagreements"]['w_model_wo_tie'] = annotation["GPT4 Judgement"] != annotation["Overall"]
        annotation["disagreements"]['w_human'] = annotation["Pradeep's judgment"] != annotation["Overall (w/ tie)"]

    for key in ["Task", "Overall (w/ tie)", "Overall"]:
        print(f"Results grouped by {key}:")
        group_by(annotations, key)
        print()
    
    length_preferences = {}
    keys = ["Overall (w/ tie)", "Overall", "GPT4 Judgement"]
    for key in keys:
        length_preferences[key] = []
        for annotation in annotations:
            length_judgement = 2 if len(annotation[reference_name]) < len(annotation["Llama2 7b Chat output"]) else 1
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

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./",
        help="Path to the save dir."
    )

    args = parser.parse_args()
    main(args)
