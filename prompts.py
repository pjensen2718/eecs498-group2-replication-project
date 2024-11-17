"""Reads the first 50 stories from the validation set and splices them at random points to create prompts for our model."""

from datasets import load_dataset

import csv
import random 

dataset = load_dataset("roneneldan/TinyStories")
validation = dataset['validation'][0:50]["text"]

prompts = []

for story in validation:
    spaces = [i for i, ch in enumerate(story) if ch == ' ' and i > 0 and story[i-1] != '.']
    start = int(len(spaces) * 0.25)
    end = int(len(spaces) * 0.3)

    idx = random.choice(spaces[start:end])
    prompts.append(story[:idx].replace("\n", ""))

with open("prompts.csv", mode="w", newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Prompt"])

    for prompt in prompts:
        writer.writerow([prompt])
