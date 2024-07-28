
import argparse
import json
import os
import random
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import RELATIVE_PROMPT_WO_REF, RELATIVE_PROMPT


def main(args):
    random.seed(2024)

    outputs_1 = json.load(open(args.output_path, 'r'))
    outputs_2 = json.load(open(args.reference_path, 'r'))
    
    with open(os.path.join(args.config_dir, "templates", f"{'_'.join(args.config_name.split('_')[:-1])}.txt"), 'r') as f:
        original_template = f.read()
    if args.use_human_reference:
        assert args.human_path
        outputs_human = json.load(open(args.human_path, 'r'))
        with open(os.path.join(args.config_dir, "templates", "prometheus_w_tie_w_reference.txt"), 'r') as f:
            prometheus_template = f.read()
    else:
        with open(os.path.join(args.config_dir, "templates", "prometheus_w_tie.txt"), 'r') as f:
            prometheus_template = f.read()

    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
    if args.use_tie:
        judge = PrometheusEval(model=model, relative_grade_template=prometheus_template)
    else:
        judge = PrometheusEval(model=model, relative_grade_template=RELATIVE_PROMPT if args.use_human_reference else RELATIVE_PROMPT_WO_REF)

    rubric = original_template.split("\n\n")[1]
    # print(rubric)
    # assert False

    reverse_map = {
        "A": "B",
        "B": "A",
        "tie": "tie"
    }

    preference_map = {
        "A": 1.0,
        "B": 2.0,
        "tie": 0.0
    }

    for category in args.nr_category:
        instructions = []
        responses_A = []
        responses_B = []
        reference_answers = []

        data_1 = outputs_1[category]
        data_2 = outputs_2[category]
        data_human = outputs_human[category] if args.use_human_reference else [None] * len(data_1)

        random_bits = [bool(random.getrandbits(1)) for _ in range(len(data_1))]

        for i, (dp1, dp2, dp_human) in enumerate(zip(data_1, data_2, data_human)):
            assert dp1["dataset"] == dp2["dataset"]
            assert dp1["instruction"] == dp2["instruction"]
            instructions.append(dp1["instruction"])
            output1 = dp2["output"] if random_bits[i] else dp1["output"]
            output2 = dp1["output"] if random_bits[i] else dp2["output"]
            responses_A.append(output1)
            responses_B.append(output2)
            if dp_human:
                reference_answers.append(dp_human["output"])

        feedbacks, scores = judge.relative_grade(
            instructions=instructions,
            responses_A=responses_A,
            responses_B=responses_B,
            rubric=rubric,
            reference_answers=reference_answers if args.use_human_reference else None
        )

        output_file = []
        for i, (dp1, dp2, score) in enumerate(zip(data_1, data_2, scores)):
            score = reverse_map[score] if random_bits[i] else score
            output_file.append({
                "instruction": dp1["instruction"],
                "output_1": dp1["output"],
                "output_2": dp2["output"],
                "annotator": args.config_name,
                "preference": preference_map[score],
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
        "--human_path",
        type=str,
        help="Path to the reference outputs. If none is provided, will use human-written references."
    )
    parser.add_argument(
        "--save_dir",
        type=str, 
        default="results/alpaca_farm")
    
    parser.add_argument(
        "--use_tie",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--use_human_reference",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default="gpt-3.5-turbo",
        help="If specified, we will use the OpenAI API to generate the predictions.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If given, we will use vLLM to generate the predictions - much faster.",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="If specified, we will use the dir as the root directory for annotator configuration.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="alpaca_eval_gpt4",
        help="We will use the directory under configuration directory as the file that contains the annotator configuration file.",
    )
    parser.add_argument(
        "--embed_human_response",
        action="store_true",
        help="If given, we will embed human response into the prompt."
    )
    parser.add_argument(
        "--multiple_answer",
        action="store_true",
        help="If given, we will embed human response into the prompt."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    main(args)
