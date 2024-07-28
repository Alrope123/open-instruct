# pip install rouge --quiet
# pip install bert_score --quiet

import os
import re
import pandas as pd
import argparse
import random
import json
from collections import defaultdict

# Python Implementation of the ROUGE Metric
from rouge import Rouge

# BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference sentences by cosine similarity.
from bert_score import BERTScorer

# function to calculate the Rouge score
def get_rouge_scores(rouge, text1, text2, ref_text):
    # Calculate the ROUGE scores for both summaries using reference
    if len(text1.strip()) == 0:
        eval_1_rouge = [{metric: {'f': 0} for metric in ["rouge-1", "rouge-2", "rouge-l"]}]
    else:
        eval_1_rouge = rouge.get_scores(text1, ref_text)
    if len(text2.strip()) == 0:
        eval_2_rouge = [{metric: {'f': 0} for metric in ["rouge-1", "rouge-2", "rouge-l"]}]
    else:
        eval_2_rouge = rouge.get_scores(text2, ref_text)

    scores = {}
    for metric in ["rouge-1", "rouge-2", "rouge-l"]:
        eval_1_score = eval_1_rouge[0][metric]["f"]
        eval_2_score = eval_2_rouge[0][metric]["f"]
        if eval_1_score > eval_2_score:
            scores[metric] = 1.0
        elif eval_1_score < eval_2_score:
            scores[metric] = 2.0
        else:
            scores[metric] = 0.0
    return scores

def get_bert_scores(bert_scorer, texts1, texts2, ref_texts):
    _, _, F1_1 = bert_scorer.score(texts1, ref_texts)
    _, _, F1_2 = bert_scorer.score(texts2, ref_texts)
    scores = []
    for f1_1, f1_2 in zip(F1_1, F1_2):
        if f1_1 > f1_2:
            scores.append(1.0)
        elif f1_1 < f1_2:
            scores.append(2.0)
        else:
            scores.append(0.0)
    return scores

def main(args):

    random.seed(2024)

    rouge = Rouge()
    scorer = BERTScorer(lang="en")

    outputs_1 = json.load(open(args.output_path, 'r'))
    outputs_2 = json.load(open(args.reference_path, 'r'))
    outputs_human = json.load(open(args.human_path, 'r'))

    for category in args.nr_category:
        print(f"Cateogry: {category}")
        data_1 = outputs_1[category]
        data_2 = outputs_2[category]
        data_human = outputs_human[category]

        all_scores = defaultdict()
        
        print("Getting Rouge Score...")
        for i, (dp1, dp2, dp_human) in enumerate(zip(data_1, data_2, data_human)):
            # Get Rouge scores 
            scores = get_rouge_scores(rouge, dp1["output"], dp2["output"], dp_human["output"])
            for k, v in scores.items():
                if k not in all_scores:
                    all_scores[k] = []
                all_scores[k].append(v)

        print("Getting Bert Score...")
        # Get BertScores
        cur_outputs_1 = [dp["output"] for dp in data_1]
        cur_outputs_2 = [dp["output"] for dp in data_2]
        cur_outputs_human = [dp["output"] for dp in data_human]
        all_scores["BERTScore"] = get_bert_scores(scorer, cur_outputs_1, cur_outputs_2, cur_outputs_human)

        for score_type, scores in all_scores.items():
            output_file = []
            for i, (dp1, dp2, dp_human, score) in enumerate(zip(data_1, data_2, data_human, scores)):
                output_file.append({
                    "instruction": dp1["instruction"],
                    "output_1": dp1["output"],
                    "output_2": dp2["output"],
                    "output_human": dp_human["output"],
                    "annotator": score_type,
                    "preference": score,
                    "id": dp1["dataset"]
                })

            output_path = os.path.join(args.save_dir, f"{score_type}", category.lower().replace(' ', '_'))
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

    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    main(args)
