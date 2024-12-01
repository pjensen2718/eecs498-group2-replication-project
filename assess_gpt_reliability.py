"""Assess the reliability of GPT-4 as a grader by making it grade the same story completion over and over again."""

import collections
from openai import OpenAI
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time

from evaluator import grade_completion_with_gpt, obtain_grades, get_all_completions, get_model

model_name = "roneneldan/TinyStories-33M"
model, tokenizer = get_model(model_name, False)

model.eval()

openai_client = OpenAI()

prompts = [
    "Once upon a time, in a big forest, there lived a rhinoceros named Roxy. Roxy loved to climb. She climbed trees, rocks, and",
    # "Once upon a time, in a small yard, there was a small daisy. The daisy had a name. Her name was",
    # "Once upon a time, there was a thoughtful girl named Sue. Sue loved to help her mom around",
    # "Once upon a time, there was a kind farmer. He had a big cow. The cow was",
    # "Once upon a time, there was a little girl named Lucy. She had a pet cat named Tom. They loved to play together in"
]

def grade_same_completion_repeatedly(completions, num_gradings, plot_metrics, prompt1_skeleton: str = "", prompt2: str = "", delimiter: str = "***"):
    for (idx, _), completion in completions.items():
        prompt_scores = collections.defaultdict(list)
        
        if plot_metrics:
            print(f"Completion {idx}: {completion}")
        
        for _ in range(num_gradings):
            _, grading = grade_completion_with_gpt(openai_client, completion[0], prompt1_skeleton, prompt2)
            grades, msg = obtain_grades(grading)

            if not msg:
                prompt_scores["grammar"].append(grades['grammar'])
                prompt_scores["creativity"].append(grades['creativity'])
                prompt_scores["consistency"].append(grades['consistency'])
                prompt_scores["plot"].append(grades['plot'])

        if plot_metrics:
            cur_time = time.time()
            with PdfPages(f'grading_frequencies_{cur_time}.pdf') as pdf:

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
        else:
            category_averages = collections.defaultdict(int)
            for category, scores in prompt_scores.items():
                category_averages[category] = sum(scores)/len(scores)
            return category_averages


def grade_same_story_with_different_prompts(completions):
    print(grade_same_completion_repeatedly(completions, 100, True))

    user_prompt_2 = """
    Now, grade the student's completion in terms of whether the plot makes sense, 
    consistency with the story's beginning, creativity, and grammar.
    The scores for each of these categories should be an integer out of 10. 
    Moreover, please provide your best guess of what the age of the student 
    might be, as reflected from the completion. Choose from possible age 
    groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E: 10-12. F: 13-16.

    Format your output as follows:
    Plot: X/10, Consistency: X/10, Creativity: X/10, Grammar: X/10, Age group: X (Y-Z)
    """

    print(grade_same_completion_repeatedly(completions, 100, True, "", user_prompt_2), "\n")

    custom_prompt1_skeleton = f"""
    Your task is to evaluate the performance of a student. The student is 
    given the following exercise. In the following exercise, the student is 
    given a beginning of a story. The student needs to complete it into a full 
    story. The exercise tests the student's language abilities and creativity. 
    The symbol ~ marks the separator between the prescribed beginning and 
    the student's completion:
    
    The student wrote the following story:

    XXXXXXX

    Please provide your general assessment about the story written by the 
    student (the one after the ~ symbol). Please be concise. Is it 
    gramatically correct? Is it consistent with the requirements in the 
    exercise? Is it consistent with the beginning of the story? Pay special 
    attention to whether the student manages to complete the sentence which is 
    split in the middle by the separator ~.
    """

    for _, cmps in completions.items():
        for i, cmp in enumerate(cmps):
            cmps[i] = cmp.replace("***", "~")

    user_prompt_2_2 = """
        Now, grade the student's completion in terms of grammar, creativity, 
        consistency with the story's beginning and whether the plot makes sense. 
        The scores for each of these categories should be an integer out of 10. 
        Moreover, please provide your best guess of what the age of the student 
        might be, as reflected from the completion. Choose from possible age 
        groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E: 10-12. F: 13-16.

        Format your output as follows:
        Grammar: X/10, Creativity: X/10, Consistency: X/10, Plot: X/10, Age group: X (Y-Z)
        """
    print(grade_same_completion_repeatedly(completions, 100, True, custom_prompt1_skeleton, delimiter="~"))
    # # grade the same completion with a modified prompt (in this case, we changed the delimiter)
    # print(grade_same_completion_repeatedly(completions, 3, False, custom_prompt1_skeleton, delimiter="~"))

def main():
    completions = get_all_completions(model, tokenizer, prompts, 1, False)
    # grade_same_completion_repeatedly(completions, 3, True) # grade the same completion with the same prompt over and over again
    
    grade_same_story_with_different_prompts(completions)

main()
