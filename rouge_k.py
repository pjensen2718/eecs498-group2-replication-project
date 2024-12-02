"""Python script for calculating and plotting rouge-k precision scores for diversity metrics."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import load_dataset

# Split a string into a list of k-grams
def get_k_grams(text, k):
    words = text.split()
    k_grams = []
    
    for i in range(len(words) - k + 1):
        k_gram = ' '.join(words[i:i + k])
        k_grams.append(k_gram)

    return k_grams  

# Formula for the Rouge-k precision score, calculated as the 
def rouge_k_precision_score(generated_story, original_story, k):
    generated_kgrams = get_k_grams(generated_story, k)

    original_kgrams = get_k_grams(original_story, k)

    original_kgrams_set = set(original_kgrams)

    overlap_count = 0
    for k_gram in generated_kgrams:
        if k_gram in original_kgrams_set:
            overlap_count += 1

    if len(original_kgrams_set) > 0:
        score = overlap_count / len(original_kgrams_set)
    else:  
        score = 0.0

    return round(score, 3)

# Fetch the first 50 stories from the validation split of the dataset, whose beginnings have been used as our test prompts
def get_validation_set():
    dataset = load_dataset("roneneldan/TinyStories")
    return dataset['validation'][0:100]["text"]

# Fetch the generated story completions 
def get_unique_completions(outs_file):
    outs_df = pd.read_csv(outs_file, delimiter='|')
    return list(zip(outs_df['prompt_id'], outs_df['prompt'], outs_df['completion']))[::4]

# Standardization of the format used for our completion file output for easier text processing
def standardize(story, prompt):
    return story.replace("\n", "").replace("\\n", "").replace(prompt, "")

# Uniqueness Section 1
def similarity_to_original(outs_file):
    validation = get_validation_set()
    unique_completions = get_unique_completions(outs_file)

    scores = []
    for prompt_id, prompt, completion in unique_completions:
        prompt_id = int(prompt_id)

        original = standardize(validation[prompt_id], prompt)
        generated = standardize(completion, prompt)

        scores.append(rouge_k_precision_score(generated, original, 2))
    
    return scores

# Uniqueness Section 2 
def similarity_to_generated(outs_file):
    unique_completions = get_unique_completions(outs_file) # prompt_id, prompt, completion
    scores = []

    for i, (prompt_id, prompt, completion) in enumerate(unique_completions):
        generated_focus = standardize(completion, prompt)
        max_score = 0
        
        for j, (other_prompt_id, other_prompt, other_completion) in enumerate(unique_completions):

            if j != i:
                generated_other = standardize(other_completion, other_prompt)
                max_score = max(max_score, rouge_k_precision_score(generated_focus, generated_other, 2))

        scores.append(max_score)

    return scores


# Uniqueness Section 3 

# ////////////////////////////////////////////////////////////////////////////////

# ////////////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":
    outs_file_1M = "evals/outs/out--TinyStories-1M--2024-11-16--17-04-55.csv"
    outs_file_33M = "evals/outs/out--TinyStories-33M--2024-11-16--17-21-08.csv"

    # Uniquness Section 1 (Similarity to original (sto)) **************************************************
    sto_scores_1M = similarity_to_original(outs_file_1M)
    sto_scores_33M = similarity_to_original(outs_file_33M)

    sto_mean_1M = sum(sto_scores_1M)/len(sto_scores_1M)
    sto_mean_33M = sum(sto_scores_33M)/len(sto_scores_33M)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the histogram for sto_scores_1M_model
    axes[0].hist(sto_scores_1M, bins=30, color='purple', edgecolor='black', alpha=0.7)
    axes[0].set_title('Rouge-2 Score Distribution For TinyStories-1M Model')
    axes[0].set_xlabel('Rouge-2 Score')
    axes[0].set_ylabel('Frequency')
    axes[0].text(0.2, 5, f'Mean: {sto_mean_1M:.3f}', color='blue', fontsize=16, fontweight='bold')

    # Plot the histogram for sto_scores_33M_model
    axes[1].hist(sto_scores_33M, bins=30, color='purple', edgecolor='black', alpha=0.7)
    axes[1].set_title('Rouge-2 Score Distribution For TinyStories-33M Model')
    axes[1].set_xlabel('Rouge-2 Score')
    axes[1].set_ylabel('Frequency')
    axes[1].text(0.2, 5, f'Mean: {sto_mean_33M:.3f}', color='blue', fontsize=16, fontweight='bold')

    plt.tight_layout() 
    plt.show()

    # Uniquness Section 2 (Similarity to generated (stg))  **************************************************
    stg_scores_1M = similarity_to_generated(outs_file_1M)
    stg_scores_33M = similarity_to_generated(outs_file_33M)

    stg_mean_1M = sum(stg_scores_1M)/len(stg_scores_1M)
    stg_mean_33M = sum(stg_scores_33M)/len(stg_scores_33M)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the histogram for sto_scores_1M_model
    axes[0].hist(stg_scores_1M, bins=30, color='blue', edgecolor='black', alpha=0.7)
    axes[0].set_title('Rouge-2 Score Distribution (OTHER) For TinyStories-1M Model')
    axes[0].set_xlabel('Rouge-2 Score')
    axes[0].set_ylabel('Frequency')
    axes[0].text(0.15, 5, f'Mean: {stg_mean_1M:.3f}', color='blue', fontsize=16, fontweight='bold')

    # Plot the histogram for sto_scores_33M_model
    axes[1].hist(stg_scores_33M, bins=30, color='blue', edgecolor='black', alpha=0.7)
    axes[1].set_title('Rouge-2 Score Distribution (OTHER) For TinyStories-33M Model')
    axes[1].set_xlabel('Rouge-2 Score')
    axes[1].set_ylabel('Frequency')
    axes[1].text(0.15, 5, f'Mean: {stg_mean_33M:.3f}', color='blue', fontsize=16, fontweight='bold')

    plt.tight_layout() 
    plt.show()

    # Uniquness Section 3 (Similarity to dataaset (stData))  **************************************************]