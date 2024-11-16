import collections
import re

import numpy as np 
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from matplotlib.backends.backend_pdf import PdfPages

model_name = "roneneldan/TinyStories-33M"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

model.eval()

prompts = [
    "Once upon a time, in a big forest, there lived a rhinoceros named Roxy. Roxy loved to climb. She climbed trees, rocks, and",
    "Once upon a time, in a small yard, there was a small daisy. The daisy had a name. Her name was",
    "Once upon a time, there was a thoughtful girl named Sue. Sue loved to help her mom around",
    "Once upon a time, there was a kind farmer. He had a big cow. The cow was",
    "Once upon a time, there was a little girl named Lucy. She had a pet cat named Tom. They loved to play together in"
]

from evaluator import gen_all_completions, grade_completion_with_gpt

completions = gen_all_completions(model, tokenizer, prompts, 1)

num_gradings = 20

with PdfPages('grading_frequencies.pdf') as pdf:
    for (idx, _), completion in completions.items():
        prompt_scores = collections.defaultdict(list)
        
        print(f"Completion {idx}: {completion}")
        
        for i in range(num_gradings):
            _, grading = grade_completion_with_gpt(completion)
            scores = re.findall(r"(\d+)/10", grading)
            
            scores = [int(score) for score in scores]

            if len(scores) != 4:
                continue

            grammar, creativity, consistency, plot = scores

            prompt_scores["grammar"].append(grammar)
            prompt_scores["creativity"].append(creativity)
            prompt_scores["consistency"].append(consistency)
            prompt_scores["plot"].append(plot)

        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        axes = ax.flatten()

        for idx, (metric, scores) in enumerate(prompt_scores.items()):
            unique_scores, frequencies = np.unique(scores, return_counts=True)
            axes[idx].bar(unique_scores, frequencies, color='skyblue', edgecolor='black')
            axes[idx].set_title(metric.capitalize())
            axes[idx].set_xlabel("Scores")
            axes[idx].set_ylabel("Frequency")
            axes[idx].set_yticks(range(0, max(frequencies) + 1))
            axes[idx].set_xticks(range(min(scores), max(scores) + 1))

        fig.text(0.5, -0.12, f'Completion: {completion}', ha='center', va='top', fontsize=8, wrap=True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        pdf.savefig(fig)
        plt.close(fig)
