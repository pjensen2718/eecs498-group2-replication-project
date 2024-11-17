"""Python script for calculating and plotting rouge-k precision scores for diversity metrics."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import load_dataset

def get_k_grams(text, k):
    words = text.split()
    k_grams = []
    
    for i in range(len(words) - k + 1):
        k_gram = ' '.join(words[i:i + k])
        k_grams.append(k_gram)

    return k_grams  

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

def get_validation_set():
    dataset = load_dataset("roneneldan/TinyStories")
    return dataset['validation'][0:50]["text"]

def get_unique_completions(outs_file):
    outs_df = pd.read_csv(outs_file, delimiter='|')
    return list(zip(outs_df['prompt_id'], outs_df['prompt'], outs_df['completion']))[::4]

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

# ////////////////////////////////////////////////////////////////////////////////

# ////////////////////////////////////////////////////////////////////////////////

outs_file_1M = "evals/outs/out--TinyStories-1M--2024-11-16--17-04-55.csv"
outs_file_33M = "evals/outs/out--TinyStories-33M--2024-11-16--17-21-08.csv"

scores_1M = similarity_to_original(outs_file_1M)
scores_33M = similarity_to_original(outs_file_33M)

mean_1M = sum(scores_1M)/len(scores_1M)
mean_33M = sum(scores_33M)/len(scores_33M)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].hist(scores_1M, bins=30, color='purple', edgecolor='black', alpha=0.7)
axes[0].set_title('Rouge-2 Score For TinyStories-1M Model')
axes[0].set_xlabel('Rouge-2 Score')
axes[0].set_ylabel('Frequency')
axes[0].text(0.2, 5, f'Mean: {mean_1M:.3f}', color='blue', fontsize=16, fontweight='bold')

# Plot the histogram for scores_33M
axes[1].hist(scores_33M, bins=30, color='purple', edgecolor='black', alpha=0.7)
axes[1].set_title('Rouge-2 Score For TinyStories-33M Model')
axes[1].set_xlabel('Rouge-2 Score')
axes[1].set_ylabel('Frequency')
axes[1].text(0.2, 5, f'Mean: {mean_33M:.3f}', color='blue', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()

# def rouge_k_fmeasure(original_story, generated_story, k):
#     # Calculate precision in both directions
#     precision_otg= rouge_k_precision_score(original_story, generated_story, k)
#     precision_gto = rouge_k_precision_score(generated_story, original_story, k)

#     # Calculate F-measure using harmonic mean
#     if precision_otg + precision_gto > 0:
#         fmeasure = (2 * precision_otg * precision_gto) / (precision_otg + precision_gto)
#     else:
#         fmeasure = 0.0

#     return fmeasure

# ////////////////////////////////////////////////////////////////////////////////
#
# ////////////////////////////////////////////////////////////////////////////////


# outs_33M_run = "evals/outs/out--TinyStories-33M--2024-11-16--17-21-08.csv"

# outputs_filename = 'evals/outs/out--TinyStories-1M--2024-11-12--23-46-15.csv'


# #original_stories_filename = 'file.csv'

# df_outputs = pd.read_csv(outputs_filename, delimiter = '|')

# df_original = load_dataset("roneneldan/TinyStories")['train'].to_pandas()

# print(df_original)

# #df_original = pd.read_csv(original_stories_filename)

# print(df_outputs)

# original_stories = df_original.iloc[:,0].tolist()

# for i in range(len(df_outputs)):
#     g_story = .replace("***", "")
#     generated_stories[i] = g_story


# scores_generated_to_original = []
# scores_generated_to_genereated = []
# scores_generated_to_dataset = []
# scores_generated_to_closest = []

# # 
# for index, row in df_outputs.iterrrows():
#     # prompt_id|prompt|completion|analysis|grammar|creativity|consistency|plot|age_group
#     print(row)

#     values = row.split('|')
    
#     g_story = generated_stories[index]
#     original_story = original_stories[values[0]]

#     # 1. How much of the new generation is contained in the original story
#     scores_generated_to_original.append(rouge_k_precision_score(g_story, original_story, 2))

#     # 2. How similar the new generation is to the 99 other generated stories
#     # max_score = 0
#     # for i in generated_stories:
#     #     gen_to_gen_score = rouge_k_fmeasure(g_story, i, 2)
#     #     max_score = max(max_score, gen_to_gen_score)
#     # scores_generated_to_genereated.append(max_score)

#     # 3. Wo what extent the k-grams in the generated story copied from the training data


#     # 4. How similar the generated story is to the closest point
#     # max_score = 0
#     # for i in original_stories:
#     #     org_to_gen_score = rouge_k_fmeasure(i, g_story,2)
#     #     max_Score = max(max_score, org_to_gen_score)
#     # scores_generated_to_closest.append(max_score)


# fig, axs = plt.subplots(1,4,figsize = (15,5))
# labels = ['1', '2', '3', '4']

# axs[0].bar(labels, scores_generated_to_original)
# axs[0].set_title('Graph 1')

# # axs[1].bar(labels, scores_generated_to_genereated)
# # axs[1].set_title('Graph 2')

# # axs[2].bar(labels, scores_generated_to_dataset)
# # axs[2].set_title('Graph 3')

# # axs[3].bar(labels, scores_generated_to_closest)
# # axs[3].set_title('Graph 4')

# plt.tight_layout
# plt.show()