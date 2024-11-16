"""Python script for evaluating SLM output via GPT-4."""

# Scale up evaluation to all prompts.
# Look into interpretability of model.
# Explore alternate evaluation approach.

import os, collections

from openai import OpenAI

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import re
import csv
import numpy as np

model_name = "roneneldan/TinyStories-33M"
model = AutoModelForCausalLM.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

model.eval()

def generate(prompt, max_new_tokens=300, top_k=50):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    attention_mask = inputs['attention_mask']

    # uses temperature = 1 by default
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=attention_mask,
        do_sample=True,        # Enable sampling for varied outputs
        max_new_tokens=max_new_tokens,
        top_k=top_k,              # Limits sampling to top K tokens
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_output(prompts, output_per_prompt=3):
    outputs = collections.defaultdict(list)
    for i, prompt in enumerate(prompts):
        for _ in range(output_per_prompt):
            completion = generate(prompt).split(prompt)
            outputs[(i, prompt)].append(prompt + "***" + completion[1])
    return outputs

# prompts = [
#     "Once upon a time, in a big forest, there lived a rhinoceros named Roxy. Roxy loved to climb. She climbed trees, rocks, and",
#     "Once upon a time, in a small yard, there was a small daisy. The daisy had a name. Her name was",
#     "Once upon a time, there was a thoughtful girl named Sue. Sue loved to help her mom around",
#     "Once upon a time, there was a kind farmer. He had a big cow. The cow was",
#     "Once upon a time, there was a little girl named Lucy. She had a pet cat named Tom. They loved to play together in",
#     "Once upon a time, there was a little brown dog named Spot. He loved to play with his ball in",
#     "Once upon a time, there was a little boy named Tom. He loved to play with his red",
#     "Once upon a time, there was a big dog named Max. Max had a red collar that he wore",
#     "Once upon a time, there was a girl named Mia. Mia loved her jewelry. She had a big box full"
# ]

prompts = []

# with open("prompts.csv", mode="r", newline='', encoding="utf-8") as file:
#     header = next(file)
#     reader = csv.reader(file)
#     for row in reader:
#         prompts.append(row[0])

# for prompt in prompts:
#     print(prompt)


# Return two GPT prompts in the form
def construct_prompt(story_completion):
    user_prompt_1 = f"""
    Your task is to evaluate the performance of a student. The student is given the following exercise.
    In the following exercise, the student is given a beginning of a story. The student needs to complete it into a full story.
    The exercise tests the student´s language abilities and creativity. The symbol *** marks the separator between the prescribed beginning and the student’s completion:
    
    The student wrote the following story:

    {story_completion}

    Please provide your general assessment about the story written by the student (the one after the *** symbol). Please be concise.
    Is it gramatically correct? Is it consistent with the requirements in the exercise? Is it consistent with the beginning of the story? 
    Pay special attention to whether the student manages to complete the sentence which is split in the middle by the separator ***.
    """

    user_prompt_2 = f"""
    Now, grade the student’s completion in terms of grammar, creativity, consistency, with the story’s beginning and whether the plot makes sense.
    The scores for each of these categories should be an integer out of 10.
    Moreover, please provide your best guess of what the age of the student might be, as reflected from the completion. 
    Choose from possible age groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E: 10-12. F: 13-16.

    Format your output as follows:
    Grammar: X/10, Creativity: X/10, Consistency: X/10, Plot: X/10, Age group: X (Y-Z)
    """
    
    return user_prompt_1, user_prompt_2, story_completion

def grade_completion_with_gpt(user_prompt_1, user_prompt_2):
    client = OpenAI()
    
    messages = [{"role": "user", "content": user_prompt_1}]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_completion_tokens=300,
        temperature=1
    )
    
    response_1 = response.choices[0].message.content

    # print(f"Model's response to user prompt 1:\n{response_content}")

    messages.append({"role": "assistant", "content": response_1})
    messages.append({"role": "user", "content": user_prompt_2})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_completion_tokens=250,
        temperature=1
    )

    response_2 = response.choices[0].message.content
    # print(f"Model's response to user prompt 2:\n{response_content}")

    return response_1, response_2

output_file = "eval.csv"
avg_scores = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

age_groups = collections.defaultdict(int)
num_completions = 0

with open(output_file, "w", newline='') as out:
    # csv_writer = csv.writer(out)
    
    # headers = ["Prompt ID", "Prompt", "Completion", "Analysis", "Grammar", "Creativity", "Consistency", "Plot", "Age Group"]
    # csv_writer.writerow(headers)
    
    prompts = ['If there’s a fire in the house, what’s the first thing you should do?']

    completions_by_prompt = generate_output(prompts)
    for (prompt_id, prompt), completions in completions_by_prompt.items():
        for completion in completions:
            print(completion)
            # user_prompt_1, user_prompt_2, story_completion = construct_prompt(completion)
            # analysis, grading = grade_completion_with_gpt(user_prompt_1, user_prompt_2)

            # print(grading)
            
            # scores = re.findall(r"(\d+)/10", grading)
            # if len(scores) == 4:
            #     scores = [float(score) for score in scores]
            #     grammar, creativity, consistency, plot = scores
                
            #     if any(score > 10 for score in scores):
            #         print("Assigned score greater than 10.")
            #         continue
                
            #     print(grammar, creativity, consistency, plot)
                
            #     age_group = grading.split(":")[-1][1:]
            #     csv_writer.writerow([prompt_id, prompt.replace("\n", "\\n"), story_completion.replace("\n", "\\n"), analysis.replace("\n", "\\n"), grammar, creativity, consistency, plot, age_group])
                
            #     avg_scores += [float(grammar), float(creativity), float(consistency), float(plot)]
            #     age_groups[age_group] += 1
            #     num_completions += 1
            # else:
            #     print(f"Grading output was malformed — {grading}.")
    
    # avg_scores /= num_completions
    # print('Average scores [grammar, creativity, consistency, plot]:',avg_scores)
    # print('Age group frequency:', age_groups.items())