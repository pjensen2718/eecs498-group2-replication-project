import pathlib

from transformers import AutoModelForCausalLM, AutoTokenizer
import click


@click.command()
@click.option("-i", "--input_model", multiple=True, type=str, help="Select specific input model; can be used multiple times.")
@click.option("-u", "--use_official", is_flag=True, help="Use official models; redundant if specific model requested.")
@click.option("-v", "--verbose", is_flag=True, help="Print verbose output.")
def main(input_model, use_official, verbose):
    if (input_model):  # input model specified
        return -1 #todo
    else:  # input model not specified
        # here, either we can use our own model or run through all official models; this can be done better with click later
        # TODO: potential functionality for our own model

        official_models = [f"roneneldan/TinyStories-{n}M" for n in [1, 3, 8, 28, 33]]
        for model in official_models:
            model = AutoModelForCausalLM.from_pretrained(model)
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")  # page 2 of paper

        

    return 0

if __name__ == "__main__":
    main()