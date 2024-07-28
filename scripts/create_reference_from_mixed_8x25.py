
import os
import json
import argparse
import logging
import random
from collections import defaultdict

def main(args):
    data = []
    with open(args.data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    for g in ['a', 'b']:
        formatted_data = defaultdict(list)
        for dp in data:
            formatted_data[dp['category']].append({
                'instruction': dp['text'],
                'output': dp[f'completion_{g}'],
                "generator": dp[f'model_{g}'],
                "dataset": f"no_robots_8x25_stronger_{dp['category']}"
            })

        with open(os.path.join(args.save_dir, f"mixed_{g}_references_8x25_stronger.json"), 'w') as f:
            json.dump(formatted_data, f)
    

    # Output Human References
    human_data = []
    with open(args.human_data_path, 'r') as f:
        for line in f:
            human_data.append(json.loads(line))
    category_to_human_reference = defaultdict(list)
    for dp in human_data:
        category_to_human_reference[dp['category']].append(dp)

    output_human_data = defaultdict(list)
    for dp in data:
        matching_human_data = category_to_human_reference[dp['category']]
        found = False
        human_output = ""
        for dp_human in matching_human_data:
            if dp_human['prompt'] == dp['text']:
                found = True
                human_output = dp_human['messages'][1]['content']
        assert found
        output_human_data[dp['category']].append({
            'instruction': dp['text'],
            'output': human_output,
            "generator": 'human',
            "dataset": f"no_robots_8x25_{dp['category']}"
        })
    with open(os.path.join(args.save_dir, f"mixed_human_references_8x25_stronger.json"), 'w') as f:
        json.dump(output_human_data, f)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/xinxil/open-instruct/tmp/references/no_robot_samples_for_human_annotation_8x25.jsonl",
    )
    parser.add_argument(
        "--human_data_path",
        type=str,
        default="/home/xinxil/open-instruct/tmp/data/no_robots_full_data.json",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/xinxil/open-instruct/tmp/references/",
    )
    args = parser.parse_args()
    
    main(args)
