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

def main(args):
    random.seed(42)

    reverse_map = {
        1: 2,
        2: 1
    }

    cross_annotation_file_path = args.cross_annotation_file_path
    save_dir = args.save_dir

    instruction_to_annotations = defaultdict(list)
    assert os.path.exists(cross_annotation_file_path)
    with open(cross_annotation_file_path, mode ='r') as f:    
        annotations = json.load(f)

    instruction_to_annotations_gpt4 = {}
    assert os.path.exists(args.gpt4_file_path)
    with open(args.gpt4_file_path, mode ='r') as f:    
        annotations_gpt4 = json.load(f)


    for annotation in annotations:
        instruction = annotation["instruction"]
        output1 = annotation["output_1"]
        output2 = annotation["output_2"]
        preference = annotation["preference"]
        if (instruction, output2, output1) not in instruction_to_annotations:
            instruction_to_annotations[annotation["instruction"], output1, output2].append(preference)
        else:
            preference = reverse_map[preference]
            instruction_to_annotations[annotation["instruction"], output2, output1].append(preference)


    for annotation in annotations_gpt4:
        instruction = annotation["instruction"]
        output1 = annotation["output_1"]
        output2 = annotation["output_2"]
        preference = int(annotation["preference"])
        if preference == 0:
            preference = 1 if bool(random.getrandbits(1)) else 2
        if (instruction, output2, output1) in instruction_to_annotations:
            preference = reverse_map[preference]
            instruction_to_annotations_gpt4[annotation["instruction"], output2, output1] = preference
        else:
            assert (instruction, output1, output2) in instruction_to_annotations
            instruction_to_annotations_gpt4[annotation["instruction"], output1, output2] = preference


    human_agreement = []
    longer_agreement = []
    shorter_agreement = []
    gpt4_agreement = []
    prefer_longer = []
    all_lengths = []
    print(f"Total # Annotations: {len(instruction_to_annotations)}")
    for (instruction, output1, output2), preference in instruction_to_annotations.items():
        if len(preference) != 4:
            random.shuffle(preference)
            instruction_to_annotations[(instruction, output1, output2)] = preference[:4]
            preference = preference[:4]
        
        longer_preference = 1 if len(output1) > len(output2) else 2
        shorter_preference = 2 if len(output1) > len(output2) else 1
        gpt4_preference = instruction_to_annotations_gpt4[instruction, output1, output2]
        for p in preference:
            prefer_longer.append(longer_preference == p)
        human_agreement.append(LOO_agreement_within(preference))
        longer_agreement.append(LOO_agreement(preference, longer_preference))
        shorter_agreement.append(LOO_agreement(preference, shorter_preference))
        gpt4_agreement.append(LOO_agreement(preference, gpt4_preference))
        all_lengths.append(np.abs(len(output1) - len(output2)))

    pretty_output("Length Bias (longer)", prefer_longer)
    pretty_output("Always longer LOO agreement", longer_agreement)
    pretty_output("Always shorter LOO agreement", shorter_agreement)
    pretty_output("Human LOO agreement", human_agreement)
    pretty_output("GPT4 LOO agreement", gpt4_agreement)
    print(f"Average Length: {sum(all_lengths) / len(all_lengths)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cross_annotation_file_path",
        type=str,
        default="/net/nfs.cirrascale/allennlp/xinxil/tmp/alpaca_farm_human_crossannotations.json",
        help="Path to the file that contains prompts."
    )

    parser.add_argument(
        "--gpt4_file_path",
        type=str,
        default="/home/xinxil/open-instruct/tmp/alpaca-eval-cross-annotations/alpaca_eval_annotator_cache.json",
        help="Path to the file that contains prompts."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="/net/nfs.cirrascale/allennlp/xinxil/tmp/",
        help="Path to the save dir."
    )

    args = parser.parse_args()
    main(args)
