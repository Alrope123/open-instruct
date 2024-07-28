from datasets import load_dataset
from collections import defaultdict
import json
from tqdm import tqdm
import numpy as np

# Replace 'imdb' with the name of the dataset you want to download
dataset_name = 'lmsys/chatbot_arena_conversations'

# Load the dataset
dataset = load_dataset(dataset_name, split="train")

model_statics = defaultdict(list)

for row in tqdm(dataset):
    model_statics[row['model_a']].append({
        "win": row['winner'] == "model_a",
        "length": len(row['conversation_a'][1]["content"])
    })
    model_statics[row['model_b']].append({
        "win": row['winner'] == "model_b",
        "length": len(row['conversation_b'][1]["content"])
    })

final_results = {}
for model, statics in model_statics.items():
    final_results[model] = {}
    for k in statics[0].keys():
        v_list = [static[k] for static in statics]
        final_results[model][k] = sum(v_list) / len(v_list)

win_rates = [final_results[model]["win"] for model in final_results.keys()]
lengths = [final_results[model]["length"] for model in final_results.keys()]

print(win_rates)
print(lengths)
assert len(win_rates) == len(lengths)
correlation_matrix = np.corrcoef(win_rates, lengths)
correlation = correlation_matrix[0, 1]

# Display the first few rows of the dataset
print(json.dumps(final_results, indent=4))
print(f"Correlation between win rate and lengths: {correlation}")
