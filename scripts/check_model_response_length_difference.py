import os
import json
import argparse
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def main(args):
    random.seed(2024)

    data_path = args.data_path

    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    length_differences = []
    for dp in data:
        length_differences.append(len(dp["completion_a"]) - len(dp["completion_b"]))

    plt.hist(length_differences, weights=np.ones(len(length_differences)) / len(length_differences), bins=50, range=[-4000, 4000], color='skyblue', edgecolor='black')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # Adding labels and title
    plt.xlabel('Length Difference')
    plt.ylabel('Percentage')
    plt.title('Lengths difference compared to GPT-4-Turbo')
    
    # Display the plot
    plt.show()
    plt.savefig("tmp/length difference mixed.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        required=True,
        help="Path to the dir that contains the prediction file."
    )

    args = parser.parse_args()
    main(args)
