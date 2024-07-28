import json
import argparse

def main(args):
    for type in args.models:
        file_path = f"/home/xinxil/open-instruct/tmp/references/{type}_references_8x25.json"

        data = json.load(open(file_path, 'r'))

        total_length = []
        for category, ls in data.items():
            if category == "Coding":
                continue
            for dp in ls:
                if len(dp["output"].strip()) > 0:
                    total_length.append(len(dp["output"]))
        
        print(f"{type}: {sum(total_length) / len(total_length)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Categories in the No Robots dataset to include. If not specified, all categories will be used"
    )
    args = parser.parse_args()
    main(args)