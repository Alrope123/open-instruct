import os
import json
import argparse
import logging
import random
from collections import defaultdict
import csv
from itertools import groupby
from operator import itemgetter

def group_by(data, key_to_group_by, absolute=False):
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
            group_result[disagreement_type] = f"{sum([item['disagreements'][disagreement_type] for item in group]) / (total_disagreement[disagreement_type] if not absolute else len(group)) * 100}%" if total_disagreement[disagreement_type] != 0 else "0"
        results[key] = group_result
    print(results)


def main(args):
    random.seed(42)
    annotation_file_path = args.annotation_file_path
    # save_dir = args.save_dir
    output2_name = "Output 2"
    output1_name = "Output 1" # "Human output"
    gpt4_judgement_name_no_ref = args.gpt4_judgement_name_no_ref
    gpt4_judgement_name_with_ref = args.gpt4_judgement_name_with_ref
    # judgement1_name = "Shane's Judgement"
    # judgement2_name = "Pradeep's Judgement"
    judgement1_name = args.judgement1_name
    judgement2_name = args.judgement2_name if args.judgement2_name is not None else args.judgement1_name
    if args.judgement2_name:
        judgement_at_least_one = "At least one"
    else:
        judgement_at_least_one = None
    shuffled_name = "Shuffled"
    preference_map = {1: 2, 2:1, 0:0}

    annotations = []
    with open(annotation_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            annotations.append(row)

    for i, annotation in enumerate(annotations):
        # unshuffle the outputs
        shuffled = True if annotation[shuffled_name] == "TRUE" else False
        annotation[judgement1_name] = int(annotation[judgement1_name])
        annotation[judgement2_name] = int(annotation[judgement2_name])
        annotation[gpt4_judgement_name_no_ref] = int(annotation[gpt4_judgement_name_no_ref])
        annotation[gpt4_judgement_name_with_ref] = int(annotation[gpt4_judgement_name_with_ref])
        if shuffled:
            new_annotation = annotation.copy()
            # annotation[output1_name], annotation[output2_name] = annotation[output2_name], annotation[output1_name]
            # annotation[judgement1_name] = preference_map[int(annotation[judgement1_name])]
            # annotation[judgement2_name] = preference_map[int(annotation[judgement2_name])]
            new_annotation[output1_name], new_annotation[output2_name] = annotation[output2_name], annotation[output1_name]
            new_annotation[judgement1_name] = preference_map[int(annotation[judgement1_name])]
            new_annotation[judgement2_name] = preference_map[int(annotation[judgement2_name])]
            new_annotation[gpt4_judgement_name_no_ref] = preference_map[int(annotation[gpt4_judgement_name_no_ref])]
            new_annotation[gpt4_judgement_name_with_ref] = preference_map[int(annotation[gpt4_judgement_name_with_ref])]
            annotation = new_annotation
            
        annotations[i] = annotation
    # assert False

    # Calculate overall win / tie rates
    for key in [gpt4_judgement_name_no_ref, gpt4_judgement_name_with_ref, judgement1_name, judgement2_name]:
        win_rates = defaultdict(list)
        tie_rates = defaultdict(list)
        # print([annotation[key] for annotation in annotations])
        for annotation in annotations:
            win_rates[annotation["Task"]].append(annotation[key] == 1)
            tie_rates[annotation["Task"]].append(annotation[key] == 0)
        print(f"Overall winrate from {key}:")
        for k, v in win_rates.items():
            print("\t{}: {}%".format(k, sum(v) / len(v) * 100))
        print(f"Overall tierate from {key}:")
        for k, v in tie_rates.items():
            print("\t{}: {}%".format(k, sum(v) / len(v) * 100))

    # Calculate categorized disagreements
    for i, annotation in enumerate(annotations):
        annotation['disagreements'] = {}
        for gpt4 in [gpt4_judgement_name_no_ref, gpt4_judgement_name_with_ref]:
            annotation["disagreements"][f'{gpt4}_w_model_neither'] = annotation[judgement1_name] != annotation[gpt4] and annotation[judgement2_name] != annotation[gpt4] 
            for judger in [judgement1_name, judgement2_name]:
                annotation["disagreements"][f'{judger}_against_{gpt4}'] = annotation[gpt4] != annotation[judger]
        annotation["disagreements"]['between judger'] = annotation[judgement1_name] != annotation[judgement2_name]
        
        annotations[i] = annotation

    # Calculate total disagreement not categorized 
    print("Total disagreements:")
    for disagreement_type, _ in annotations[0]['disagreements'].items():
        rate = sum([item['disagreements'][disagreement_type] for item in annotations]) / len(annotations)
        print(f"Disagreement {disagreement_type}: {rate * 100}%")
    print()

    # Calculate disagreement rates cateogrized 
    for absolute in [False, True]:
        for key in [judgement1_name, judgement2_name, "Task"]:
            print(f"disagreements grouped by {key}:")
            group_by(annotations, key, absolute)
            print()
    
    # Lenght Preference
    longer_length_preferences = {}
    shorter_length_preferences = {}
    # print([annotation[judgement1_name] for annotation in annotations])
    # print([annotation[judgement2_name] for annotation in annotations])
    keys = [judgement1_name, judgement2_name, gpt4_judgement_name_no_ref, gpt4_judgement_name_with_ref]
    for key in keys:
        longer_length_preferences[key] = []
        shorter_length_preferences[key] = []
        for annotation in annotations:
            longer_length_judgement = 2 if len(annotation[output1_name]) < len(annotation[output2_name]) else 1
            shorter_length_judgement = 2 if len(annotation[output1_name]) > len(annotation[output2_name]) else 1
            longer_length_preferences[key].append(str(annotation[key]) == str(longer_length_judgement))
            shorter_length_preferences[key].append(str(annotation[key]) == str(shorter_length_judgement))
        longer_length_preferences[key] = sum(longer_length_preferences[key]) / len(longer_length_preferences[key])
        shorter_length_preferences[key] = sum(shorter_length_preferences[key]) / len(shorter_length_preferences[key])
    print("Length preference:")
    print("Longer:")
    print(longer_length_preferences)
    print("Shorter:")
    print(shorter_length_preferences)


    human_agreement = []
    no_ref_gpt4_disagree_with_all_human = []
    ref_benefits = []
    ref_mess_up = []
    for i, annotation in enumerate(annotations):
        human_judgement1 = annotation[judgement1_name]
        human_judgement2 = annotation[judgement2_name]    
        gpt_judgement_no_ref = annotation[gpt4_judgement_name_no_ref]
        gpt_judgement_with_ref = annotation[gpt4_judgement_name_with_ref]
        if human_judgement1 == human_judgement2:
            human_agreement.append(i+2)
        if human_judgement1 == human_judgement2 and annotation[gpt4_judgement_name_no_ref] != human_judgement1 and human_judgement1 != 0:
            no_ref_gpt4_disagree_with_all_human.append(i+2)
        if judgement_at_least_one:
            if (human_judgement1 != gpt_judgement_no_ref and human_judgement2 != gpt_judgement_no_ref) and (human_judgement1 == gpt_judgement_with_ref or human_judgement2 == gpt_judgement_with_ref):
                ref_benefits.append(i+2)
            elif (human_judgement1 == gpt_judgement_no_ref or human_judgement2 == gpt_judgement_no_ref) and (human_judgement1 != gpt_judgement_with_ref and human_judgement2 != gpt_judgement_with_ref):
                ref_mess_up.append(i+2)
    
    print("No reference GPT4 disagree with all human: {}".format(no_ref_gpt4_disagree_with_all_human))
    print("Ref benefits: {}".format(ref_benefits))
    print("Ref messed up: {}".format(ref_mess_up))
    print(len(human_agreement))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotation_file_path",
        type=str,
        default=None,
        help="Path to the file that contains annotations."
    )

    parser.add_argument(
        "--gpt4_judgement_name_no_ref",
        type=str,
        required=True
    )

    parser.add_argument(
        "--gpt4_judgement_name_with_ref",
        type=str,
        required=True
    )

    parser.add_argument(
        "--judgement1_name",
        type=str,
        required=True
    )

    parser.add_argument(
        "--judgement2_name",
        type=str,
        default=None
    )

    args = parser.parse_args()
    main(args)
