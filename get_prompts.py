from datasets import load_dataset

import csv

dataset = load_dataset("roneneldan/TinyStories")
validation = dataset['validation'][0:50]["text"]

with open("stories.csv", mode="w", newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Story"])

    for story in validation:
        writer.writerow([story])
        writer.writerow(["..."])