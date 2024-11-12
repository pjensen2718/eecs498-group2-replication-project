import collections
import csv
from pathlib import Path
import re
import sys
import time

import click
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


# NOTE: the below could probably be made better by allowing an input dataset or reading from a file
def get_prompts(dataset) -> list[str]:
    """Get prompts from dataset or pre-defined list."""
    # TODO: consider making a csvfile that maps prompt_id to prompt, timestamped
    return [
        "Once upon a time, in a big forest, there lived a rhinoceros named Roxy. Roxy loved to climb. She climbed trees, rocks, and",
        "Once upon a time, in a small yard, there was a small daisy. The daisy had a name. Her name was",
        "Once upon a time, there was a thoughtful girl named Sue. Sue loved to help her mom around",
        "Once upon a time, there was a kind farmer. He had a big cow. The cow was",
        "Once upon a time, there was a little girl named Lucy. She had a pet cat named Tom. They loved to play together in",
        "Once upon a time, there was a little brown dog named Spot. He loved to play with his ball in",
        "Once upon a time, there was a little boy named Tom. He loved to play with his red",
        "Once upon a time, there was a big dog named Max. Max had a red collar that he wore",
        "Once upon a time, there was a girl named Mia. Mia loved her jewelry. She had a big box full"
    ]


def intro(model: str, dataset: str, all_official: bool) -> list[str]:
    """Handle errors with options."""
    if model and all_official:
        sys.exit("Error: cannot both specify specific model and use all official models.")
    if dataset:
        sys.exit("Error: functionality for user-defined dataset not implemented.")
    else:  # specific dataset not specified, use our default
        return get_prompts(dataset)  # currently just returns a hard-coded list of strings


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 300, top_k: int = 50) -> str:
    """Generate an output based on a prompt from the model to be evaluated."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    attention_mask = inputs['attention_mask']

    # uses temperature = 1 by default  TODO: is temp of 1 correct???
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=attention_mask,
        do_sample=True,        # Enable sampling for varied outputs
        max_new_tokens=max_new_tokens,
        top_k=top_k,              # Limits sampling to top K tokens
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_all_completions(model, tokenizer, prompts: list[str], output_per_prompt: int) -> dict[tuple[int, str], list]:
    """Generate all completions from given prompts."""
    outputs = collections.defaultdict(list)
    for i, prompt in enumerate(prompts):
        for _ in range(output_per_prompt):
            completion = generate_completion(model, tokenizer, prompt).split(prompt)
            outputs[(i, prompt)].append(prompt + "***" + completion[1])
    return outputs


def construct_prompt(story_completion: str) -> tuple[str, str]:
    """Generate prompt to be forwarded to GPT-4o."""
    user_prompt_1 = f"""
    Your task is to evaluate the performance of a student. The student is given the following exercise.
    In the following exercise, the student is given a beginning of a story. The student needs to complete it into a full story.
    The exercise tests the student's language abilities and creativity. The symbol *** marks the separator between the prescribed beginning and the student's completion:
    
    The student wrote the following story:

    {story_completion}

    Please provide your general assessment about the story written by the student (the one after the *** symbol). Please be concise.
    Is it gramatically correct? Is it consistent with the requirements in the exercise? Is it consistent with the beginning of the story? 
    Pay special attention to whether the student manages to complete the sentence which is split in the middle by the separator ***.
    """

    user_prompt_2 = """
    Now, grade the student's completion in terms of grammar, creativity, consistency, with the story's beginning and whether the plot makes sense.
    The scores for each of these categories should be an integer out of 10.
    Moreover, please provide your best guess of what the age of the student might be, as reflected from the completion. 
    Choose from possible age groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E: 10-12. F: 13-16.

    Format your output as follows:
    Grammar: X/10, Creativity: X/10, Consistency: X/10, Plot: X/10, Age group: X (Y-Z)
    """
    
    return user_prompt_1, user_prompt_2


def grade_completion_with_gpt(story_completion: str) -> tuple[str, str]:
    """Grade a story completion from the SLM with GPT-4o."""
    user_prompt_1, user_prompt_2 = construct_prompt(story_completion)

    client = OpenAI()
    
    messages = [{"role": "user", "content": user_prompt_1}]  # ask to grade traits

    response = client.chat.completions.create(
        model="gpt-4o",  # cheaper than gpt-4
        messages=messages,
        max_completion_tokens=300,
        temperature=1
    )
    
    response_1 = response.choices[0].message.content

    messages.append({"role": "assistant", "content": response_1})  # give the context of the last GPT response
    messages.append({"role": "user", "content": user_prompt_2})  # ask to grade numerically

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_completion_tokens=250,
        temperature=1
    )

    response_2 = response.choices[0].message.content

    return response_1, response_2


# prompts may not be a list of strings in the end due to memory concerns, maybe a pandas dataframe or np array?
def write_to_csv(model, tokenizer, prompts: list[str], num_completions: int, out_csv, err_csv) -> None:
    headers = ["prompt_id", "prompt", "completion", "analysis", "grammar", "creativity", "consistency", "plot", "age_group"]
    # headers = ["Prompt ID", "Prompt", "Completion", "Analysis", "Grammar", "Creativity", "Consistency", "Plot", "Age Group"]
    # csv_writer.writerow(headers)
    
    # completions_by_prompt = generate_all_completions(prompts[0:1])
    
    # for (prompt_id, prompt), completions in completions_by_prompt.items():
    #     for completion in completions:
    #         user_prompt_1, user_prompt_2, story_completion = construct_prompt(completion)
    #         analysis, grading = grade_completion_with_gpt(user_prompt_1, user_prompt_2)
            
    #         scores = re.findall(r"(\d+)/10", grading)
    #         if len(scores) == 4:
    #             scores = [float(score) for score in scores]
                
    #             if any(score > 10 for score in scores):
    #                 print("Assigned score greater than 10.")
    #                 continue
                
    #             grammar, creativity, consistency, plot = scores
    #             age_group = grading.split(":")[-1][1:]

    #             prompt = prompt.replace("\n", "\\n")
    #             analysis = analysis.replace("\n", "\\n")
    #             story_completion = story_completion.replace("\n", "\\n")
            
    #             csv_writer.writerow([prompt_id, prompt, story_completion, analysis, grammar, creativity, consistency, plot, age_group])
                
    #             avg_scores += [float(grammar), float(creativity), float(consistency), float(plot)]
    #             age_groups[age_group] += 1
    #             num_completions += 1
    #         else:
    #             print(f"Grading output was malformed — {grading}.")
    
    # avg_scores /= num_completions

    # print('Average Scores (Grammar, Creativity, Consistency, Plot)]:', list(avg_scores))
    # print('Age Group Frequency:', age_groups.items())


# prompts may not be a list of strings in the end due to memory concerns, maybe a pandas dataframe or np array?
def evaluate_model(model_str: str, prompts: list[str], num_completions: int, verbose: bool) -> None:
    """Evaluate a SLM - combines all parts."""
    model = AutoModelForCausalLM.from_pretrained(model_str)
    # should we prompt for custom tokenizer?
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")  # page 2 of paper

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    model.eval()  # put the model into evaluation mode

    prompts = get_prompts(False)  # this currently just takes from a list of predefined prompts

    # NOTE: here we can either append to a previous CSV or create a new one; both without using unnecessary GPT prompts
    # currently opting for creating a new one at the timestamp
    cur_time = time.gmtime(time.time())
    time_str = ""
    for i in range(6):
        if i == 3:
            time_str += '-'
        time_str += f"-{cur_time[i]:02d}"
    # filenames will be of the form {out,err}--{model string, like TinyStories-1M}--YYYY-MM-DD--HH-MM-SS
    out_fname = f"out--{model_str.split('/')[1]}-{time_str}"
    err_fname = f"err--{model_str.split('/')[1]}-{time_str}"

    eval_dir = Path("evals")
    with open(eval_dir/"outs"/out_fname, "+w", newline="", encoding="utd-8") as out_csv:
        with open(eval_dir/"errs"/err_fname, "+w", newline="", encoding="utf-8") as err_csv:
            write_to_csv(model, tokenizer, prompts, num_completions, out_csv, err_csv)


@click.command()
@click.option("-m", "--model", multiple=True, type=str, help="Select specific model to evaluate; can be used multiple times.")
@click.option("-d", "--dataset", type=str, help="Specify dataset.")
@click.option("-a", "--all_official", is_flag=True, help="Use all official models; redundant if specific model requested.")
@click.option("-n", "--num_completions", default=10, help="Number of completions to be done by (each) model.")
@click.option("-v", "--verbose", is_flag=True, help="Print verbose output.")
def main(model, dataset, all_official, num_completions, verbose):
    """Main driver for evaluation script."""
    prompts = intro(model, dataset, all_official)  # option error handling

    # currently, use this as "python3 temp.py -m roneneldan/TinyStories-1M"

    if (model):  # input model specified
        evaluate_model(model, prompts, num_completions, verbose)

    else:  # input model not specified
        # here, either we can use our own model or run through all official models; this can be done better with click later
        # TODO: potential functionality for our own model
        if (all_official):
            if __debug__: sys.exit("debug: do not try to evaluate all models")
            official_models = [f"roneneldan/TinyStories-{n}M" for n in [1, 3, 8, 28, 33]]
            for model in official_models:
                evaluate_model(model, prompts, num_completions, verbose)

        else:  # use our own model?
            sys.exit("ERROR: functionality for our own model not implemented.") 

    return 0

if __name__ == "__main__":
    main()