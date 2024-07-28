import os
import json
import argparse
import logging
import random
from collections import defaultdict
import csv
import numpy as np

def LOO_agreement(human_prefs, my_p):
    accs = []
    for i, pref in enumerate(human_prefs):
        for j, pref2 in enumerate(human_prefs):
            if i != j:
                accs.append(my_p == pref2)
    return sum(accs) / len(accs)

def LOO_agreement_within(prefs):
    accs = []
    for i, pref in enumerate(prefs):
        for j, pref2 in enumerate(prefs):
            if i != j:
                accs.append(pref == pref2)
    return sum(accs) / len(accs)

def pretty_output(prefix, l):
    print(f"{prefix}: {round(sum(l) / len(l) * 100, 2)}%")


def eliminate_tie(p):
    if p == "0":
        return random.choice(['1', '2'])
    else:
        return p

def main(args):
    random.seed(42)
    preference_map = {
        "1": "2",
        "2": "1"
    }

    data = defaultdict(list)
    keys = []
    with open(args.annotation_file_path, mode ='r')as file:
        csvFile = csv.reader(file)
        for i, lines in enumerate(csvFile):
            if i == 0:
                keys = lines
            else:
                if lines[-1] not in ["0", "1", "2"]:
                    if lines[-1] != "":
                        print([lines[-1]])
                        assert False
                    continue
                for j, key in enumerate(keys):
                    if key != "Human":
                        if args.no_tie and key in ['Mine', 'GPT-4']:
                            data[key].append(eliminate_tie(lines[j]))
                        else:
                            data[key].append(lines[j])
                    else:
                        if args.no_tie:
                            data[key].append([eliminate_tie(p) for p in lines[j].split(',')])
                        else:
                            data[key].append(lines[j].split(','))


    print(len(data['Human']))

    for key in ['GPT-4', 'Mine']:
        matching_shorter = [data['Shorter'][i] == data[key][i] for i in range(len(data['Shorter']))]
        matching_longer = [preference_map[data['Shorter'][i]] == data[key][i] for i in range(len(data['Shorter']))]
        pretty_output(f"Rate of {key} for prefering shorter", matching_shorter)
        pretty_output(f"Rate of {key} for prefering longer", matching_longer)


    for key in ['GPT-4', 'Mine', 'Shorter']:
        matching = [LOO_agreement(data['Human'][i], data[key][i]) for i in range(len(data['Human']))]
        pretty_output(f"LOO of {key}", matching)

    matching = [data['Mine'][i] == data["GPT-4"][i] for i in range(len(data['Mine'])) if int(data[key][i]) != 0]
    pretty_output(f"Rate of me agreeing with GPT-4", matching)

    human_matching = [LOO_agreement_within(data['Human'][i]) for i in range(len(data['Human']))]
    pretty_output(f"LOO of Human", human_matching)

    matching_shorter = []
    matching_longer = []
    for i in range(len(data['Human'])):
        matching_shorter.extend([data['Human'][i][j] == data['Shorter'][i] for j in range(len(data['Human'][i]))])
        matching_longer.extend([data['Human'][i][j] == preference_map[data['Shorter'][i]] for j in range(len(data['Human'][i]))])
    pretty_output(f"Rate of Human for prefering shorter", matching_shorter)
    pretty_output(f"Rate of Human for prefering longer", matching_longer)

    matching_GPT4 = []
    for i in range(len(data['Human'])):
        matching_GPT4.extend([data['Human'][i][j] == data['GPT-4'][i] for j in range(len(data['Human'][i]))])
    pretty_output(f"Rate of Human agreeing with GPT4", matching_GPT4)

    for i in range(len(data['Human'])):
        # if not any([d == data['Mine'][i] for d in data['Human'][i]]) and data['Mine'][i] != data['GPT-4'][i]:
        #     print(f"{i}: Me doesn't agree with anyone else")
        # elif not any([d == data['GPT-4'][i] for d in data['Human'][i]]) and data['Mine'][i] != data['GPT-4'][i]:
        #     print(f"{i}: GPT-4 doesn't agree with anyone else")
        # elif not any([d == data['GPT-4'][i] for d in data['Human'][i]]) and data['Mine'][i] == data['GPT-4'][i]:
        #     print(f"{i}: Me and GPT-4 don't agree with other human")
        if not any([d == data['GPT-4'][i] for d in data['Human'][i]]) and not any([d == data['Mine'][i] for d in data['Human'][i]]) and data['Mine'][i] != data['Shorter'][i] and data['GPT-4'][i] != data['Shorter'][i]:
            print(f"{i+2}: Human Strongly prefer shorter")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotation_file_path",
        type=str,
        default="tmp/HumanIF Eval - 8x25 Stronger Annotations.csv",
        help="Path to the file that contains prompts."
    )

    parser.add_argument(
        "--no_tie",
        default=False,
        action="store_true"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="/net/nfs.cirrascale/allennlp/xinxil/tmp/",
        help="Path to the save dir."
    )

    args = parser.parse_args()
    main(args)
