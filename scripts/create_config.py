import os
import json
import argparse
import yaml
import random
import copy

def remove_substring_between(original_string, start_substring, end_substring):
    while True:
        # Find the start and end positions of the substrings
        start_pos = original_string.find(start_substring)
        end_pos = original_string.find(end_substring, start_pos + len(start_substring))
        
        # If either start or end substring is not found, break the loop
        if start_pos == -1 or end_pos == -1:
            break
        
        # Remove the substrings and the text in between
        original_string = original_string[:start_pos] + original_string[end_pos + len(end_substring):]
    
    return original_string


def main(args):
    random.seed(42)

    model = args.model
    template_name = args.template_name
    config_dir = args.config_dir
    no_example = args.no_example
    keep_history = args.keep_history
    mutliple_answer = args.mutliple_answer
    do_sampling = args.do_sampling
    
    model_configs = json.load(open(os.path.join(config_dir, "base_template.json"), 'r'))[model]
    with open(os.path.join(config_dir, "templates", f"{template_name}.txt"), 'r') as f:
        template = f.read()
    with open(os.path.join(config_dir, "base_config.yaml"), 'r') as f:
        config = yaml.safe_load(f)
    
    template_name = template_name + "_" + model + ("" if not no_example else "_no_ex") + ("" if not do_sampling else "_sample")

    config[template_name] = copy.deepcopy(config["base"])
    del config["base"]
    config[template_name]["prompt_template"] = f"templates_generated/{template_name}.txt"
    if keep_history:
        config[template_name]["keep_history"] = True
    
    for k, v in model_configs.items():
        if type(v) == str:
            template = template.replace("{" + k + "}", v) 
        if k in ["fn_completions", "completions_kwargs"]:
            config[template_name][k] = v
    
    if mutliple_answer:
        config[template_name]["fn_completion_parser"] = "match_multiple_parser"
    
    if do_sampling and model != "gpt4":
        config[template_name]["completions_kwargs"]["do_sample"] = True
        config[template_name]["completions_kwargs"]["temperature"] = 0.9

    if no_example:
        template = remove_substring_between(template, "{" + "example_begin" + "}", "{" + "example_end" + "}")
    else:
        template = template.replace("{" + "example_begin" + "}", "").replace("{" + "example_end" + "}", "")

    with open(os.path.join(config_dir, "templates_generated", f"{template_name}.txt"), 'w') as f:
        f.write(template)
    if not os.path.exists(os.path.join(config_dir, template_name)):
        os.mkdir(os.path.join(config_dir, template_name))
    with open(os.path.join(config_dir, template_name, "configs.yaml"), 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="gpt4",
    )

    parser.add_argument(
        "--template_name",
        type=str,
        required=True
    )

    parser.add_argument(
        "--config_dir",
        type=str,
        default="eval/alpaca_farm/configs",
    )

    parser.add_argument(
        "--no_example",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--keep_history",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--mutliple_answer",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--do_sampling",
        default=False,
        action="store_true",
    )


    args = parser.parse_args()
    main(args)
