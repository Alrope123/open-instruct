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
from collections import Counter

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

def most_frequent_element(lst):
    if not lst:
        assert False
        return None, 0
    counter = Counter(lst)
    most_commons = counter.most_common(2)
    if len(most_commons) > 1 and most_commons[0][1] == most_commons[1][1]:
        return 0, 0.5
    most_common_element, frequency = counter.most_common(1)[0]
    if frequency < 1:
        print(lst)
        assert False
    return most_common_element, frequency


def main(args):
    random.seed(42)
    prompt_file_path = args.prompt_file_path
    annotation_file_path = args.annotation_file_path
    nr_category = args.nr_category
    results_dir = args.results_dir
    judgers = ["Shane", "Pradeep"]


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

    our_annotations = {}
    for i, judger in enumerate(judgers):
        examples = []
        for category in nr_category:
            for j, cat_prompt in enumerate(prompts[category]):
                if cat_prompt[judger] != "":
                    cat_prompt["Preference"] = int(cat_prompt[judger]) if cat_prompt["Shuffled"] == "FALSE" else preference_map[int(cat_prompt[judger])]
                else:
                    cat_prompt["Preference"] = -2
                if not cat_prompt["Shuffled"] == "FALSE":
                    tmp = prompts[category][j]["Output 1"]
                    prompts[category][j]["Output 1"] = prompts[category][j]["Output 2"]
                    prompts[category][j]["Output 2"] = tmp
                examples.append(cat_prompt)
        our_annotations[judger] = copy.deepcopy(examples)

    human_annotations_data = json.load(open(annotation_file_path, 'r'))
    human_annotations = []
    for category in nr_category:
        categorical_prompts = prompts[category]
        for j, _ in enumerate(categorical_prompts):
            human_annotations.append([int(dp[1]) for dp in human_annotations_data["chatgpt_llama"][f"{category}_{j}"]])
        
    assert len(our_annotations[judgers[0]]) == len(human_annotations)

    output1_name = "Output 1"
    output2_name = "Output 2"
    
    # Merge the preferences
    preferences = []
    for i in range(len(our_annotations[judgers[0]])):
        cur_preference = {}
        for k, v in our_annotations[judgers[0]][i].items():
            if k not in ["Preference", "Preference_general"]:
                cur_preference[k] = v
        for name in judgers:
            cur_preference[name] = our_annotations[name][i]["Preference"]
        cur_preference["Majority Human Annotator"] = most_frequent_element(human_annotations[i])[0]
        preferences.append(cur_preference)

    # Calculate overall win / tie rates
    for key in judgers + ["Majority Human Annotator"]:
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
            print("\t{}: {}%".format(k, round(sum(v) / len(v) * 100, 2)))
        print("\tOverall: {}%".format(round(sum(overall_v) / len(overall_v) * 100, 2)))
        # print(f"Tie rate from {key}:")
        # overall_v = []
        # for k, v in tie_rates.items():
        #     overall_v.extend(v)
        #     # print("\t{}: {}%".format(k, round(sum(v) / len(v) * 100, 2)))
        # print("\tOverall: {}%".format(round(sum(overall_v) / len(overall_v) * 100, 2)))

    # Calculate categorized agreements
    for i, preference in enumerate(preferences):
        preference['agreements'] = {}
        for j, judger1 in enumerate(judgers + ["Majority Human Annotator"]):
            for k, judger2 in enumerate(judgers + ["Majority Human Annotator"]):
                if j < k:
                    preference["agreements"][f"{judger1}_against_{judger2}"] = preference[judger1] == preference[judger2]

    # Calculate total agreement not categorized 
    print("Total agreements:")
    for agreement_type, _ in preferences[0]['agreements'].items():
        rate = sum([item['agreements'][agreement_type] for item in preferences]) / len(preferences)
        print(f"Agreement {agreement_type}: {round(rate * 100, 2)}%")
    
    
    # Length Preference
    longer_length_preferences = {}
    shorter_length_preferences = {}
    for key in judgers + ["Majority Human Annotator"]:
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
            if int(preference[key]) > 0:
                longer_length_preferences[key].append(str(preference[key]) == str(longer_length_judgement))
                shorter_length_preferences[key].append(str(preference[key]) == str(shorter_length_judgement))
        longer_length_preferences[key] = round(sum(longer_length_preferences[key]) / len(longer_length_preferences[key]) * 100, 2)
        shorter_length_preferences[key] = round(sum(shorter_length_preferences[key]) / len(shorter_length_preferences[key]) * 100, 2)
    print("Length preference:")
    print("Longer:")
    print(json.dumps(longer_length_preferences, indent=4))
    print("Shorter:")
    print(json.dumps(shorter_length_preferences, indent=4))

    
    print("Extreme Bad Examples:")
    f_shorter = csv.writer(open(os.path.join("tmp", "extreme_different_examples_shorter.csv"), "w+"))
    f_longer = csv.writer(open(os.path.join("tmp", "extreme_different_examples_longer.csv"), "w+"))
    titles = ["Instruction", "output_1", "output_2", "Shane", "Pradeep", "Crowd Worker Majority"]
    f_shorter.writerow(titles)
    f_longer.writerow(titles)
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
        if all([preference["Majority Human Annotator"] != preference[judger] for judger in judgers]):
            if preference["Majority Human Annotator"] == shorter_length_judgement:
                f_shorter.writerow([preference["Instruction"], preference["Output 1"], preference["Output 2"], preference["Shane"], preference["Pradeep"], preference["Majority Human Annotator"]])
            elif preference["Majority Human Annotator"] == longer_length_judgement:
                f_longer.writerow([preference["Instruction"], preference["Output 1"], preference["Output 2"], preference["Shane"], preference["Pradeep"], preference["Majority Human Annotator"]])


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
        "--prompt_file_path",
        type=str,
        default=None,
        help="Path to the file that contains prompts."
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
