import os
import json
import argparse
import logging
import random
from collections import defaultdict
import csv

def main(args):
    random.seed(42)

    cross_annotation_file_path = args.cross_annotation_file_path
    gpt4_file_path = args.gpt4_file_path
    save_dir = args.save_dir

    instruction_to_annotations = defaultdict(list)
    assert os.path.exists(cross_annotation_file_path)
    with open(cross_annotation_file_path, mode ='r') as f:    
        annotations = json.load(f)

    for annotation in annotations:
        instruction_to_annotations[annotation["instruction"], annotation["output_1"], annotation["output_2"]].append(annotation)

    instruction_to_annotations_gpt4 = defaultdict(list)
    assert os.path.exists(gpt4_file_path)
    with open(gpt4_file_path, mode ='r') as f:    
        annotations_gpt4 = json.load(f)

    for annotation in annotations_gpt4:
        instruction_to_annotations_gpt4[annotation["instruction"], annotation["output_1"], annotation["output_2"]].append(annotation)

    
    print(f"Total # Annotations: {len(instruction_to_annotations)}")

    f = csv.writer(open(os.path.join(save_dir, "alpaca_eval_diff.csv"), "w+"))
    titles = ["Instruction", "output_1", "output_2", "GPT-4's Judgement", "Human Judgement"]
    f.writerow(titles)

    i = 0
    for (instruct, output1, output2), cur_annotations in instruction_to_annotations.items():
        assert (instruct, output1, output2) in instruction_to_annotations_gpt4 or (instruct, output2, output1) in instruction_to_annotations_gpt4, (instruct, output1, output2)
        if (instruct, output1, output2) in instruction_to_annotations_gpt4:
            key = (instruct, output1, output2)
            reversed = False
        elif (instruct, output2, output1) in instruction_to_annotations_gpt4:
            key = (instruct, output2, output1)
            reversed = True
        
        preference_map = {1: 2, 2:1, 0:0}
        cur_annotations_gpt4 = instruction_to_annotations_gpt4[key]
        
        assert len(cur_annotations) >= 4, (key, len(cur_annotations))
        if all([cur_annotations[0]["preference"] == cur_annotation["preference"] for cur_annotation in cur_annotations]):
            i += 1
            if not all([cur_annotations_gpt4[0]["preference"] == cur_annotation_gpt4["preference"] for cur_annotation_gpt4 in cur_annotations_gpt4]):
                print(f"GPT-4 not consistant: {key}")
            else:
                gpt_4_prefernce = int(cur_annotations_gpt4[0]["preference"]) if not reversed else preference_map[int(cur_annotations_gpt4[0]["preference"])]
                if gpt_4_prefernce != int(cur_annotations[0]["preference"]): 
                    f.writerow([cur_annotations[0]["instruction"], cur_annotations[0]["output_1"], cur_annotations[0]["output_2"], gpt_4_prefernce, cur_annotations[0]["preference"]])
    print(i)

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
