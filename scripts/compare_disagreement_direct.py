import os
import json
import argparse
import logging
import random
from collections import defaultdict
import csv
from itertools import groupby
from operator import itemgetter
import copy
import math

def similar(str1, str2):
    str1 = str1 + ' ' * (len(str2) - len(str1))
    str2 = str2 + ' ' * (len(str1) - len(str2))
    return sum(1 if i == j else 0
               for i, j in zip(str1, str2)) / float(len(str1))

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
            rate = sum([item['disagreements'][disagreement_type] for item in group]) / (total_disagreement[disagreement_type] if not absolute else len(group))
            group_result[disagreement_type] = f"{round( (1- rate)* 100, 2)}%" if total_disagreement[disagreement_type] != 0 else "0"
        results[key] = group_result
    pretty_json = json.dumps(results, indent=4)
    print(pretty_json)

def main(args):
    random.seed(42)

    prompt_file_path = args.prompt_file_path
    prompt_file_general_path = args.prompt_file_general_path
    nr_category = args.nr_category
    results_dir = args.results_dir
    models = args.models
    models_general = args.models_general
    judgers = args.judgers

    if models_general:
        assert args.preference_keys_general and prompt_file_general_path

    assert len(models) == len(args.preference_keys), (len(models), len(args.preference_keys))

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
                prompts[cat].append(line)
    
    prompts_general = {}
    if prompt_file_general_path:
        with open(prompt_file_general_path, mode ='r') as f:    
            csvFile = csv.DictReader(f)
            for line in csvFile:
                cat = line["Task"]
                if cat not in prompts_general:
                    prompts_general[cat] = []
                prompts_general[cat].append(line)


    annotations = {}
    for i_m, model in enumerate(models):
        # print("Model: " + model)
        examples = []
        for category in nr_category:
            # print("Category: " + category)
            
            category_name = category.lower().replace(' ', '_')
            cur_annotations = json.load(open(os.path.join(results_dir, f"{model}", category_name, "alpaca_eval_annotator_cache.json"), 'r'))
            if models_general:
                general_annotations = json.load(open(os.path.join(results_dir, f"{models_general[i_m]}", category_name, "alpaca_eval_annotator_cache.json"), 'r'))
            # if category == "Open QA" and model.startswith("8_wo"):
            #     print(cur_annotations[5])
            #     assert False
            # print(os.path.join(results_dir, f"{model}", category_name, "alpaca_eval_annotator_cache.json"))
            cat_prompts = prompts[category]
            for cat_prompt in cat_prompts:
                prompt = cat_prompt["Instruction"]
                found = False
                for j, annotation in enumerate(cur_annotations):
                    if models_general:
                        annotation_general = general_annotations[j]
                    if similar(annotation['instruction'], prompt.strip()) > 0.9:
                        if models_general:
                            assert similar(annotation_general['instruction'], prompt.strip()) > 0.9
                        preference_key = sorted([k for k in annotation.keys() if k.startswith("preference")])[-1] if not args.preference_keys else args.preference_keys[i_m]
                        if models_general:
                            preference_key_general = args.preference_keys_general[i_m]
                        output_1, output_2 = annotation['output_1'], annotation['output_2']
                        # assert (first_output == output_1 and not shuffled) or (first_output == output_2 and shuffled), (output_1, output_2, first_output, shuffled)
                        preference = int(annotation[preference_key]) if annotation[preference_key] is not None else -1
                        if models_general:
                            preference_general = int(annotation_general[preference_key_general]) if annotation_general[preference_key_general] is not None else -1
                        else:
                            preference_general = None
                        output_dict = {
                            "Task": category, "Instruction": annotation['instruction'], "Output 1": output_1, "Output 2": output_2, "Output Human": annotation['output_human'] if 'output_human' in annotation else "", "Preference": preference, "Preference_general": preference_general
                        }
                        examples.append(output_dict)
                        found = True
                assert found, (category, prompt)
            # if model.startswith("8_w"):
            #     print([a["preference_4"] for a in cur_annotations])
            #     print([a["Preference"] for a in examples if a["Task"] == category])
        annotations[model] = examples

    for i, judger in enumerate(judgers):
        examples = []
        for category in nr_category:
            for j, cat_prompt in enumerate(prompts[category]):
                if models_general:
                    cat_prompt_general = prompts_general[category][j]
                    assert cat_prompt["Instruction"].strip() == cat_prompt_general["Instruction"].strip()
                if cat_prompt[judger] != "":
                    cat_prompt["Preference"] = int(cat_prompt[judger]) if cat_prompt["Shuffled"] == "FALSE" else preference_map[int(cat_prompt[judger])]
                else:
                    cat_prompt["Preference"] = -2
                if models_general:
                    if cat_prompt_general[judger] != "":
                        cat_prompt["Preference_general"] = int(cat_prompt_general[judger]) if cat_prompt_general["Shuffled"] == "FALSE" else preference_map[int(cat_prompt_general[judger])]
                    else:
                        cat_prompt["Preference_general"] = -3
                examples.append(cat_prompt)
        annotations[judger] = copy.deepcopy(examples)
    

    output2_name = "Output 2"
    output1_name = "Output 1"
    if len(judgers) > 1:
        judgement_at_least_one = "At least one"
    else:
        judgement_at_least_one = None
   
    preferences = []
    preferences_general = []
    for i in range(len(annotations[models[0]])):
        cur_preference = {}
        cur_preference_general = {}
        for k, v in annotations[models[0]][i].items():
            if k not in ["Preference", "Preference_general"]:
                cur_preference[k] = v
                cur_preference_general[k] = v
        for name in models + judgers:
            cur_preference[name] = annotations[name][i]["Preference"]
            if models_general:
                cur_preference_general[name] = annotations[name][i]["Preference_general"]
        preferences.append(cur_preference)
        preferences_general.append(cur_preference_general)
    # Calculate overall win / tie rates
    for key in models + judgers:
        win_rates = defaultdict(list)
        tie_rates = defaultdict(list)
        # print([annotation[key] for annotation in annotations])
        for preference in preferences:
            win_rates[preference["Task"]].append(preference[key] == 1)
            tie_rates[preference["Task"]].append(preference[key] == 0)
        print(f"Overall winrate from {key}:")
        for k, v in win_rates.items():
            print("\t{}: {}%".format(k, round(sum(v) / len(v) * 100, 2)))
        print(f"Overall tierate from {key}:")
        for k, v in tie_rates.items():
            print("\t{}: {}%".format(k, round(sum(v) / len(v) * 100, 2)))

    # Calculate categorized disagreements
    for i, preference in enumerate(preferences):
        preference['disagreements'] = {}
        for model in models:
            preference["disagreements"][f'{model}_w_any_human'] = all([preference[model] != preference[judger] for judger in judgers]) 
            # for judger in judgers:
            #     preference["disagreements"][f'{judger}_against_{model}'] = preference[model] != preference[judger]
        for j, judger1 in enumerate(judgers):
            for k, judger2 in enumerate(judgers):
                if j < k:
                    preference["disagreements"][f"{judger1}_against_{judger2}"] = preference[judger1] != preference[judger2]
    
    # # Calculate total disagreement not categorized 
    # print("Total disagreements:")
    # for disagreement_type, _ in preferences[0]['disagreements'].items():
    #     rate = sum([item['disagreements'][disagreement_type] for item in preferences]) / len(preferences)
    #     print(f"Disagreement {disagreement_type}: {rate * 100}%")

    # Calculate total Agreement not categorized 
    print("Total agreements:")
    for disagreement_type, _ in preferences[0]['disagreements'].items():
        rate = sum([item['disagreements'][disagreement_type] for item in preferences]) / len(preferences)
        print(f"Agreement {disagreement_type}: {round((1-rate) * 100, 2)}%")
    
    # # Calculate important disagreement
    # important_agreements = defaultdict(list)
    # for i, preference in enumerate(preferences):
    #     if all([preference[judger] == preference[judgers[0]] for judger in judgers]):
    #         for key in models:
    #             important_agreements[key].append(preference[key] == preference[judgers[0]])
    # print(f"Important Agreement with {len(important_agreements[models[0]])} example")
    # for key, agreements in important_agreements.items():
    #     print(f"{key}: {round(sum(agreements) / len(agreements) * 100, 2)}%")

    # Calculate Macro-average F1
    print("F1 scores")
    for key in models:
        print(f"\tFor {key}:")
        TPs = defaultdict(int)
        FPs = defaultdict(int)
        FNs = defaultdict(int)
        all_f1s = []
        for i, preference in enumerate(preferences):
            for judger in judgers:
                if preference[key] == preference[judger]: 
                    TPs[preference[judger]] += 1
                else:
                    FPs[preference[key]] += 1
                    FNs[preference[judger]] += 1
        for k in [0, 1, 2]:
            precision = TPs[k] / (TPs[k] + FPs[k]) if (TPs[k] + FPs[k]) != 0 else 0
            recall = TPs[k] / (TPs[k] + FNs[k]) if (TPs[k] + FNs[k]) != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            all_f1s.append(f1)
            print(f"\t\tF1 score for {k}: {round(f1,2)}")
        print(f"\t\tMacro-weighted F1 score: {round(sum(all_f1s) / 3, 2)}")

    # # Calculate disagreement rates cateogrized 
    # for absolute in [False, True]:
    #     for key in ["Task"]:
    #         print(f"Agreements grouped by {key}, Absolute = {absolute}:")
    #         group_by(preferences, key, absolute)
    #         print()
    
    # # Length Preference
    # longer_length_preferences = {}
    # shorter_length_preferences = {}
    # # print([preference[judgement1_name] for preference in preferences])
    # # print([preference[judgement2_name] for preference in preferences])
    # keys = models + judgers
    # for key in keys:
    #     longer_length_preferences[key] = []
    #     shorter_length_preferences[key] = []
    #     for preference in preferences:
    #         longer_length_judgement = 2 if len(preference[output1_name]) < len(preference[output2_name]) else 1
    #         shorter_length_judgement = 2 if len(preference[output1_name]) > len(preference[output2_name]) else 1
    #         longer_length_preferences[key].append(str(preference[key]) == str(longer_length_judgement))
    #         shorter_length_preferences[key].append(str(preference[key]) == str(shorter_length_judgement))
    #     longer_length_preferences[key] = round(sum(longer_length_preferences[key]) / len(longer_length_preferences[key]), 2)
    #     shorter_length_preferences[key] = round(sum(shorter_length_preferences[key]) / len(shorter_length_preferences[key]), 2)
    # print("Length preference:")
    # print("Longer:")
    # print(longer_length_preferences)
    # print("Shorter:")
    # print(shorter_length_preferences)

    # keys = models + judgers
    # if models_general:
    #     # Correlation between general and current
    #     match_ratios = {}
    #     for key in keys:
    #         match_ratios[key] = round(sum([pref_cur[key] == pref_general[key] for pref_cur, pref_general in zip(preferences, preferences_general)]) / len(preferences_general) * 100, 2)
    #         # print(key)
    #         # print(f"correctness: {[(z+2, pref_cur[key]) for z, pref_cur in enumerate(preferences)]}")
    #         # print(f"overall: {[(z+2, pref_general[key]) for z, pref_general in enumerate(preferences_general)]}")
    #     print("Correlations:")
    #     print(json.dumps(match_ratios, indent=4))

    
    # print("No reference GPT4 disagree with all human: {}".format(no_ref_gpt4_disagree_with_all_human))
    # print("Ref benefits: {}".format(ref_benefits))
    # print("Ref messed up: {}".format(ref_mess_up))
    # print(len(human_agreement))

    # print("Preferences details")
    # for key in models + judgers:
    #     print(f"{key}: {[(z+2, pref[key]) for z, pref in enumerate(preferences)]}")

    # print out preference changes
    # common_answers = [[pref[judger] for judger in judgers] for pref in preferences]
    # for key in models:
    #     if '_wo_' in key:
    #         def is_pair(key1, key2):
    #             key1_pieces = key1.split("_")
    #             key2_pieces = key2.split("_")
    #             if len(key1_pieces) != len(key2_pieces):
    #                 return False
    #             return all([key1_pieces[i] == key2_pieces[i] for i in range(len(key1_pieces)) if i != 1]) and key1_pieces[1] == "wo" and key2_pieces[1] == "w" 
    #         other_key = [k for k in models if is_pair(key, k) and '_w_' in k][0]
    #         print(f"Changes of preference from {key} to {other_key}:")
    #         preferences_changed = defaultdict(list)
    #         print(sum([pref[key] in common_answers[i] for i, pref in enumerate(preferences)]) / len(preferences))
    #         print(sum([pref[other_key] in common_answers[i] for i, pref in enumerate(preferences)]) / len(preferences))
    #         for i, pref in enumerate(preferences):
    #             if pref[key] != pref[other_key]:
    #                 if pref[key] in common_answers[i] and pref[other_key] in common_answers[i]:
    #                     change = "good_to_good"
    #                 elif pref[key] not in common_answers[i] and pref[other_key] not in common_answers[i]:
    #                     change = "bad_to_bad"
    #                 elif pref[key] in common_answers[i] and pref[other_key] not in common_answers[i]:
    #                     change = "good_to_bad"
    #                 elif pref[key] not in common_answers[i] and pref[other_key] in common_answers[i]:
    #                     change = "bad_to_good"
    #                 else:
    #                     assert False

    #                 preferences_changed[change].append((i+2, pref[key], pref[other_key]))
    #         for change, line in preferences_changed.items():
    #             print(f"\t{change}")
    #             for index, pref_before, pref_after in line:
    #                 print(f"\t\tLine {index}: {pref_before} -> {pref_after}")

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
        "--preference_keys",
        type=str,
        default=None,
        nargs="+",
        help="Path to the dir that contains the prediction file."
    )

    parser.add_argument(
        "--preference_keys_general",
        type=str,
        default=None,
        nargs="+",
        help="Path to the dir that contains the prediction file."
    )

    parser.add_argument(
        "--models",
        type=str,
        default=None,
        nargs="+",
        help="Path to the dir that contains the prediction file."
    )

    parser.add_argument(
        "--models_general",
        type=str,
        default=None,
        nargs="+",
        help="Path to the dir that contains the prediction file."
    )

    parser.add_argument(
        "--prompt_file_path",
        type=str,
        default=None,
        help="Path to the file that contains prompts."
    )

    parser.add_argument(
        "--prompt_file_general_path",
        type=str,
        default=None,
        help="Path to the file that contains prompts."
    )

    parser.add_argument(
        "--judgers",
        default=None,
        nargs="+",
        help="Path to the dir that contains the prediction file."
    )

    args = parser.parse_args()
    main(args)
