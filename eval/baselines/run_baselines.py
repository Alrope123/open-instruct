# pip install rouge --quiet
# pip install bert_score --quiet

import os
import re
import argparse
import random
import json
from collections import defaultdict

def num_lines(text, characters_per_line = 56):
    text = text.strip()
    line_count = 0
    char_count = 0
    converages = []
    for c in text:
        if c == "\n":
            line_count += 1
            converages.append(char_count / characters_per_line)
            char_count = 0
        elif char_count >= characters_per_line:
            line_count += 1
            converages.append(1.0)
            char_count = 1
        else:
            char_count += 1
    if char_count > 0:
        converages.append(char_count / characters_per_line)
    converage_ratio = sum(converages) / len(converages) if len(converages) > 0 else 0
    assert 0.0 <= converage_ratio, (converage_ratio, text)
    assert 1.0 >= converage_ratio, (converage_ratio, text)
    return line_count + converage_ratio

def main(args):

    random.seed(2024)

    outputs_1 = json.load(open(args.output_path, 'r'))
    outputs_2 = json.load(open(args.reference_path, 'r'))
    for category in args.nr_category:
        print(f"Cateogry: {category}")
        data_1 = outputs_1[category]
        data_2 = outputs_2[category]

        scores = []
        for i, (dp1, dp2) in enumerate(zip(data_1, data_2)):
            if args.type == "longer":
                if len(dp1["output"]) > len(dp2["output"]):
                    scores.append(1.0)
                elif len(dp1["output"]) == len(dp2["output"]):
                    scores.append(0.0)
                else:
                    scores.append(2.0)
            elif args.type == "shorter":
                if len(dp1["output"]) < len(dp2["output"]):
                    scores.append(1.0)
                elif len(dp1["output"]) == len(dp2["output"]):
                    scores.append(0.0)
                else:
                    scores.append(2.0)
            elif args.type.startswith("more_lines"):
                characters_per_line = int(args.type.split('-')[-1]) if "-" in args.type else 56
                if num_lines(dp1["output"]) > num_lines(dp2["output"]):
                    scores.append(1.0)
                elif num_lines(dp1["output"]) == num_lines(dp2["output"]):
                    scores.append(0.0)
                else:
                    scores.append(2.0)
            elif args.type.startswith("less_lines"):
                characters_per_line = int(args.type.split('-')[-1]) if "-" in args.type else 56
                if num_lines(dp1["output"]) < num_lines(dp2["output"]):
                    scores.append(1.0)
                elif num_lines(dp1["output"]) == num_lines(dp2["output"]):
                    scores.append(0.0)
                else:
                    scores.append(2.0)
            elif args.type == "random_two":
                scores.append(1.0 if bool(random.getrandbits(1)) else 2.0)
            elif args.type == "random_three":
                scores.append(float(random.randint(0,2)))
            else:
                assert False

        output_file = []
        for i, (dp1, dp2, score) in enumerate(zip(data_1, data_2,scores)):
            output_file.append({
                "instruction": dp1["instruction"],
                "output_1": dp1["output"],
                "output_2": dp2["output"],
                "annotator": args.type,
                "preference": score,
                "id": dp1["dataset"]
            })

        output_path = os.path.join(args.save_dir, category.lower().replace(' ', '_'))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        json.dump(output_file, open(os.path.join(output_path, "alpaca_eval_annotator_cache.json"), "w"), indent=4)    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceH4/no_robots",
        help="Path to the reference outputs. If none is provided, will use human-written references."
    )
    parser.add_argument(
        "--nr_category",
        type=str,
        nargs="+",
        help="Categories in the No Robots dataset to include. If not specified, all categories will be used"
    )
    parser.add_argument(
        "--limit_eval_size",
        type=int,
        help="Evaluate only on these many prompt-response pairs per category. If not specified, all examples will be used."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to the file that contains outputs. If none is provided, will use model to generate."
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        help="Path to the reference outputs. If none is provided, will use human-written references."
    )
    parser.add_argument(
        "--save_dir",
        type=str, 
        default="results/alpaca_farm")
    
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        help="Path to the file that contains outputs. If none is provided, will use model to generate."
    )


    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    main(args)
