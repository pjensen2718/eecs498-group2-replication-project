"""Python script for testing functionalities."""

from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluator import construct_prompt, gen_completions, gen_all_completions


def main():
    """Driver for testing script."""
    # user_prompt_1, user_prompt_2 = construct_prompt("Hello, world! Hello, world! Hello, world! Hello, world! Hello, world! Hello, world!")
    # print(user_prompt_1, "\n\n\n\n\n\n", user_prompt_2)

    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")  # page 2 of paper

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    model.eval()  # put the model into evaluation mode

    # completions = gen_completion(model, tokenizer, "Once upon a time there was ", 5)

    # for completion in completions:
    #     print(completion)
    #     print("\n\n\n\n\n")

    prompts = ["Hello world! Hello world! Hello world!", "Once upon a time there was"]
    outputs = gen_all_completions(model, tokenizer, prompts, 2)

    print(outputs)
    
    return 0


if __name__ == "__main__":
    main()
