import argparse
import json
from collections import defaultdict
import os
import csv

def main(args):
    categories = defaultdict(list)
    data = json.load(open(args.json_path, 'r'))
    for dp in data:
        categories[dp['category']].append({
            "Prompt": dp['Prompt 1'],
            "Response": dp['Response 1'],
        })

    outputs = []
    for k, v in categories.items():
        outputs.extend([dict(dp, **{"Task": k}) for dp in v[:5]])

    f = csv.writer(open(os.path.join("surge_data", f"{os.path.basename(args.json_path).split('.')[0]}.csv"), "w+"))
    f.writerow(list(outputs[0].keys()))

    for dp in outputs:
        f.writerow(list(dp.values()))

    # with open(f'surge_data/{os.path.basename(args.json_path).split('.')[0]}.json', 'w') as f:
    #     json.dump(categories, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str)

    args = parser.parse_args()
    main(args)