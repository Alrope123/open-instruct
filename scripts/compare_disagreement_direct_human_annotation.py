import os
import json
import argparse
import logging
import random
from collections import defaultdict
import csv
from itertools import groupby
import itertools
from operator import itemgetter
import copy
import math

def LOO_agreement(human_prefs, my_p):
    accs = []
    for i, pref in enumerate(human_prefs):
        for j, pref2 in enumerate(human_prefs):
            if i != j:
                accs.append(my_p == pref2)
    return sum(accs) / len(accs)


def deteremine_pair_consistency(model_pair_to_preference):
    def better_than(model1, model2):
        if f"{model1}_{model2}" in model_pair_to_preference:
            return model_pair_to_preference[f"{model1}_{model2}"] == 1
        else:
            return model_pair_to_preference[f"{model2}_{model1}"] == 2
    
    def tie_with(model1, model2):
        if f"{model1}_{model2}" in model_pair_to_preference:
            return model_pair_to_preference[f"{model1}_{model2}"] == 0
        else:
            return model_pair_to_preference[f"{model2}_{model1}"] == 0

    def worse_than(model1, model2):
        if f"{model1}_{model2}" in model_pair_to_preference:
            return model_pair_to_preference[f"{model1}_{model2}"] == 2
        else:
            return model_pair_to_preference[f"{model2}_{model1}"] == 1
    
    model_names = ["human", "chatgpt", "llama"]
    for three_names in list(itertools.permutations(model_names, 3)):
        model1 = three_names[0]
        model2 = three_names[1]
        model3 = three_names[2]
        if better_than(model1, model2) and better_than(model2, model3):
            return better_than(model1, model3)
        elif better_than(model1, model2) and tie_with(model2, model3):
            return better_than(model1, model3)
        elif tie_with(model1, model2) and better_than(model2, model3):
            return better_than(model1, model3)
        elif tie_with(model1, model2) and tie_with(model2, model3):
            return tie_with(model1, model3)
        elif tie_with(model1, model2) and worse_than(model2, model3):
            return worse_than(model1, model3)
        elif worse_than(model1, model2) and tie_with(model2, model3):
            return worse_than(model1, model3)
        elif worse_than(model1, model2) and worse_than(model2, model3):
            return worse_than(model1, model3)
    assert False
    return None

def similar(str1, str2):
    str1 = str1 + ' ' * (len(str2) - len(str1))
    str2 = str2 + ' ' * (len(str1) - len(str2))
    return sum(1 if i == j else 0
               for i, j in zip(str1, str2)) / float(len(str1))

def group_by(data, key_to_group_by, absolute=False):
    total_agreement = {}
    for agreement_type, _ in data[0]['agreements'].items():
        total_agreement[agreement_type] = sum([item['agreements'][agreement_type] for item in data])

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
        for agreement_type, _ in group[0]['agreements'].items():
            rate = sum([item['agreements'][agreement_type] for item in group]) / (total_agreement[agreement_type] if not absolute else len(group))
            group_result[agreement_type] = f"{round(rate* 100, 2)}%" if total_agreement[agreement_type] != 0 else "0"
        results[key] = group_result
    pretty_json = json.dumps(results, indent=4)
    print(pretty_json)

def main(args):
    random.seed(42)
    annotation_file_path = args.annotation_file_path
    nr_category = args.nr_category
    results_dir = args.results_dir
    models = args.models
    models_general = args.models_general
    comparison_keys = args.comparison_keys

    if models_general:
        assert args.preference_keys_general 

    assert len(models) == len(args.preference_keys), (len(models), len(args.preference_keys))

    human_annotations_data = json.load(open(annotation_file_path, 'r'))

    # models_prefered = defaultdict(int)
    # for model_pair, data in human_annotations_data.items():
    #     if model_pair != "chatgpt_llama":
    #         continue
    #     model1 = model_pair.split("_")[0]
    #     model2 = model_pair.split("_")[1]
    #     for k, v in data.items():
    #         for p in v:
    #             if int(p[1]) == 1:
    #                 models_prefered[model1] += 1
    #             elif int(p[1]) == 2:
    #                 models_prefered[model2] += 1
    # print(json.dumps(models_prefered, indent=4))
    # assert False

    human_annotations = {}
    annotations = {}
    for i_m, model in enumerate(models):
        # print("Model: " + model)
        examples = []
        h_annotations = []

        for category in nr_category:
            if category == "Coding":
                continue

            category_name = category.lower().replace(' ', '_')

            cur_annotations = json.load(open(os.path.join(results_dir, f"{model}", category_name, "alpaca_eval_annotator_cache.json"), 'r'))
            if models_general:
                general_annotations = json.load(open(os.path.join(results_dir, f"{models_general[i_m]}", category_name, "alpaca_eval_annotator_cache.json"), 'r'))
            
            # TODO: Remove this temporary fix
            instruction_set = set()
            for j, annotation in enumerate(cur_annotations):
                # TODO: Remove this temporary fix
                if annotation['instruction'] not in instruction_set:
                    instruction_set.add(annotation['instruction'])
                else:
                    print(f"Category: {category} has a duplicate")
                    continue
                preference_key = sorted([k for k in annotation.keys() if k.startswith("preference")])[-1] if not args.preference_keys else args.preference_keys[i_m]
                preference = int(annotation[preference_key]) if annotation[preference_key] is not None else -1
                preference_general = None

                # if model == "longer_chatgpt_llama" and annotation["output_1"].startswith("On January 5, 2018, Patricia Gualinga was the victim of a home invasion by"):
                #     print(preference)
                #     assert False

                if models_general:
                    annotation_general = general_annotations[j]
                    preference_key_general = args.preference_keys_general[i_m]
                    print(annotation_general[preference_key_general])
                    print(type(annotation_general[preference_key_general]))
                    preference_general = int(annotation_general[preference_key_general]) if annotation_general[preference_key_general] in [1.0, 2.0, 0.0] else -1
                
                output_dict = {
                    "Task": category, "Instruction": annotation['instruction'], "Output 1": annotation["output_1"], "Output 2": annotation["output_2"], "Output Human": annotation['output_human'] if 'output_human' in annotation else "", "Preference": preference, "Preference_general": preference_general
                }
                examples.append(output_dict)
                h_annotations.append([int(dp[1]) for dp in human_annotations_data[comparison_keys[i_m]][f"{category}_{j}"]])

        annotations[model] = examples
        human_annotations[model] = h_annotations

    output1_name = "Output 1"
    output2_name = "Output 2"
    
    # Merge the preferences
    preferences = []
    preferences_general = []
    assert all([len(annotations[m]) == len(annotations[models[0]]) for m in models])
    for i in range(len(annotations[models[0]])):
        assert all([annotations[m][i]["Instruction"] == annotations[models[0]][i]["Instruction"] for m in models])
        
        cur_preference = {}
        cur_preference_general = {}
        for k, v in annotations[models[0]][i].items():
            if k not in ["Preference", "Preference_general"]:
                cur_preference[k] = v
                cur_preference_general[k] = v
        for name in models:
            cur_preference[name] = annotations[name][i]["Preference"]
            if models_general:
                cur_preference_general[name] = annotations[name][i]["Preference_general"]
        preferences.append(cur_preference)
        preferences_general.append(cur_preference_general)

    # Calculate overall win / tie rates
    for key in models:
        win_rates = defaultdict(list)
        tie_rates = defaultdict(list)
        # print([annotation[key] for annotation in annotations])
        for preference in preferences:
            win_rates[preference["Task"]].append(preference[key] == 1)
            # tie_rates[preference["Task"]].append(preference[key] == 0)
        print(f"Winrate from {key}:")
        overall_v = []
        for k, v in win_rates.items():
            overall_v.extend(v)
            # print("\t{}: {}%".format(k, round(sum(v) / len(v) * 100, 2)))
        print("\tOverall: {}%".format(round(sum(overall_v) / len(overall_v) * 100, 2)))
        # print(f"Tie rate from {key}:")
        # overall_v = []
        # for k, v in tie_rates.items():
        #     overall_v.extend(v)
        #     # print("\t{}: {}%".format(k, round(sum(v) / len(v) * 100, 2)))
        # print("\tOverall: {}%".format(round(sum(overall_v) / len(overall_v) * 100, 2)))

    # Calculate categorized agreements
    assert all([len(preferences) == len(human_annotations[m]) for m in models]) 
    for i, preference in enumerate(preferences):
        preference['agreements'] = {}
        for model in models:
            # preference["agreements"][f'{model}_w_any_human'] = any([preference[model] == h_p for h_p in human_annotations[model][i]])
            preference["agreements"][f'{model}_leave_one_out'] =  LOO_agreement(human_annotations[model][i], preference[model])

    # Calculate total agreement not categorized 
    print("Total agreements:")
    for agreement_type, _ in preferences[0]['agreements'].items():
        rate = sum([item['agreements'][agreement_type] for item in preferences]) / len(preferences)
        print(f"Agreement {agreement_type}: {round(rate * 100, 2)}%")
    
    # Calculate Macro-average F1
    # print("F1 scores")
    # for key in models:
    #     print(f"\tFor {key}:")
    #     TPs = defaultdict(int)
    #     FPs = defaultdict(int)
    #     FNs = defaultdict(int)
    #     all_f1s = []
    #     for i, preference in enumerate(preferences):
    #         for judger in judgers:
    #             if preference[key] == preference[judger]: 
    #                 TPs[preference[judger]] += 1
    #             else:
    #                 FPs[preference[key]] += 1
    #                 FNs[preference[judger]] += 1
    #     for k in [0, 1, 2]:
    #         precision = TPs[k] / (TPs[k] + FPs[k]) if (TPs[k] + FPs[k]) != 0 else 0
    #         recall = TPs[k] / (TPs[k] + FNs[k]) if (TPs[k] + FNs[k]) != 0 else 0
    #         f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    #         all_f1s.append(f1)
    #         print(f"\t\tF1 score for {k}: {round(f1,2)}")
    #     print(f"\t\tMacro-weighted F1 score: {round(sum(all_f1s) / 3, 2)}")

    # Calculate agreement rates cateogrized 
    for absolute in [True]:
        for key in ["Task"]:
            print(f"Agreements grouped by {key}, Absolute = {absolute}:")
            group_by(preferences, key, absolute)
            print()
    
    # Length Preference
    longer_length_preferences = {}
    shorter_length_preferences = {}
    for key in models:
        longer_length_preferences[key] = []
        shorter_length_preferences[key] = []
        for preference in preferences:
            if len(preference[output1_name]) > len(preference[output2_name]):
                longer_length_judgement = 1
                shorter_length_judgement = 2 
            elif len(preference[output1_name]) == len(preference[output2_name]):
                longer_length_judgement = 0
                shorter_length_judgement = 0
            else:
                longer_length_judgement = 2
                shorter_length_judgement = 1
            longer_length_preferences[key].append(str(preference[key]) == str(longer_length_judgement))
            shorter_length_preferences[key].append(str(preference[key]) == str(shorter_length_judgement))
        longer_length_preferences[key] = round(sum(longer_length_preferences[key]) / len(longer_length_preferences[key]) * 100, 2)
        shorter_length_preferences[key] = round(sum(shorter_length_preferences[key]) / len(shorter_length_preferences[key]) * 100, 2)
    print("Length preference:")
    print("Longer:")
    print(json.dumps(longer_length_preferences, indent=4))
    print("Shorter:")
    print(json.dumps(shorter_length_preferences, indent=4))

    # Robust Reasoning
    if len(models) == 3:
        robust_reasoning = []
        for preference in preferences:
            model_pair_to_preference = {}
            for i, model in enumerate(models):
                model_pair_to_preference[comparison_keys[i]] = preference[model]
            reasoning_result = deteremine_pair_consistency(model_pair_to_preference)
            if reasoning_result is not None:
                robust_reasoning.append(reasoning_result)
        print(f"Robust Reasoning is: {round(sum(robust_reasoning) / len(robust_reasoning) * 100, 2)}%")
        

    # Correlation between general and 
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
        "--comparison_keys",
        type=str,
        default="chatgpt_llama",
        nargs="+",
    )

    parser.add_argument(
        "--models_general",
        type=str,
        default=None,
        nargs="+",
        help="Path to the dir that contains the prediction file."
    )
    parser.add_argument(
        "--annotation_file_path",
        type=str,
        default=None,
        help="Path to the file that contains prompts."
    )


    args = parser.parse_args()
    main(args)
