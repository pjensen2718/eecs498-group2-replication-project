"""Python script for calculating rouge-k precision scores for diversity metrics"""

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