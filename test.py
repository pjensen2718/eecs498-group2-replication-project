"""Python script for testing functionalities."""

from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluator import construct_prompt, gen_completions, get_all_completions, obtain_grades


def main():
    """Driver for testing script."""
    # user_prompt_1, user_prompt_2 = construct_prompt("Hello, world! Hello, world! Hello, world! Hello, world! Hello, world! Hello, world!")
    # print(user_prompt_1, "\n\n\n\n\n\n", user_prompt_2)

    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")  # page 2 of paper

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    model.eval()  # put the model into evaluation mode

    completions = gen_completions(model, tokenizer, "What is 3 + 5?", 1)
    print(completions)

    # # for completion in completions:
    # #     print(completion)
    # #     print("\n\n\n\n\n")

    # prompts = ["Hello world! Hello world! Hello world!", "Once upon a time there was"]
    # outputs = get_all_completions(model, tokenizer, prompts, 2)

    # print(outputs)


    # grading = "Grammar: 5/10, Consistency: 1/10, Creativity: 3.44/10, Plot: 5/10"
    # grades, msg = obtain_grades(grading)
    # print(grades)
    # print(msg)
    
    # return 0


if __name__ == "__main__":
    main()
