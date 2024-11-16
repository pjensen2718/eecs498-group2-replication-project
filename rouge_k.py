"""Python script for calculating rouge-k precision scores for diversity metrics"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_k_grams(text, k):
    words = text.split()
    k_grams = []
    
    for i in range(len(words) - k + 1):
        k_gram = ' '.join(words[i:i + k])
        k_grams.append(k_gram)

    return k_grams  

def rouge_k_precision_score(generated_story, original_story, k):
    generated_kgrams = get_k_grams(generated_story, k)

    original_kgrams = get_k_grams(original_story, k)    # Generate k-grams

    original_kgrams_set = set(original_kgrams)          # Set of original k-grams

    overlap_count = 0                                   # Find number of k-grams within the generated story
    for k_gram in generated_kgrams:                     #   that appear within the original story 
        if k_gram in original_kgrams_set:
            overlap_count += 1

    if len(original_kgrams_set) > 0:
        score = overlap_count / len(original_kgrams_set)    # Calculate Score
    else:  
        score = 0.0

    return score

def rouge_k_fmeasure(original_story, generated_story, k):
    # Calculate precision in both directions
    precision_otg= rouge_k_precision_score(original_story, generated_story, k)
    precision_gto = rouge_k_precision_score(generated_story, original_story, k)

    # Calculate F-measure using harmonic mean
    if precision_otg + precision_gto > 0:
        fmeasure = (2 * precision_otg * precision_gto) / (precision_otg + precision_gto)
    else:
        fmeasure = 0.0

    return fmeasure

# ////////////////////////////////////////////////////////////////////////////////
#
# ////////////////////////////////////////////////////////////////////////////////

outputs_filename = 'file.csv'
original_stories_filename = 'file.csv'

df_outputs = pd.read_csv(outputs_filename)
df_original = pd.read_csv(original_stories_filename)

original_stories = df_original.iloc[:,0].tolist()
generated_stories = df_outputs.iloc[:,0].tolist()
for i in range(len(generated_stories)):
    values = generated_stories[i].split('|')
    g_story = values[2].replace("***", "")
    generated_stories[i] = g_story

scores_generated_to_original = []
scores_generated_to_genereated = []
scores_generated_to_dataset = []
scores_generated_to_closest = []

# 
for index, row in df_outputs.iterrrows():
    # prompt_id|prompt|completion|analysis|grammar|creativity|consistency|plot|age_group
    print(row)

    values = row.split('|')
    
    g_story = generated_stories[index]
    original_story = original_stories[values[0]]

    # 1. How much of the new generation is contained in the original story
    scores_generated_to_original.append(rouge_k_precision_score(g_story, original_story, 2))

    # 2. How similar the new generation is to the 99 other generated stories
    max_score = 0
    for i in generated_stories:
        gen_to_gen_score = rouge_k_fmeasure(g_story, i, 2)
        max_score = max(max_score, gen_to_gen_score)
    scores_generated_to_genereated.append(max_score)

    # 3. Wo what extent the k-grams in the generated story copied from the training data


    # 4. How similar the generated story is to the closest point
    max_score = 0
    for i in original_stories:
        org_to_gen_score = rouge_k_fmeasure(i, g_story,2)
        max_Score = max(max_score, org_to_gen_score)
    scores_generated_to_closest.append(max_score)


fig, axs = plt.subplots(1,4,figsize = (15,5))
labels = ['1', '2', '3', '4']

axs[0].bar(labels, scores_generated_to_original)
axs[0].set_title('Graph 1')

axs[1].bar(labels, scores_generated_to_genereated)
axs[1].set_title('Graph 2')

axs[2].bar(labels, scores_generated_to_dataset)
axs[2].set_title('Graph 3')

axs[3].bar(labels, scores_generated_to_closest)
axs[3].set_title('Graph 4')

plt.tight_layout
plt.show()

