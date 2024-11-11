import pathlib
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
import click


def intro(model, dataset, all_official):
    """Handle errors with options."""
    if not model and not all_official:  # this can be changed depending on functionality for our own model
        sys.exit("Error: a model must be specified or all official models must be specified.")
    if model and all_official:
        sys.exit("Error: cannot both specify specific model and use all official models.")


@click.command()
@click.option("-m", "--model", multiple=True, type=str, help="Select specific model to evaluate; can be used multiple times.")
@click.option("-d", "--dataset", type=str, help="Specify dataset.")
@click.option("-a", "--all_official", is_flag=True, help="Use official models; redundant if specific model requested.")
@click.option("-n", "--num_completions", default=10, help="Number of completions to be done by (each) model.")
@click.option("-v", "--verbose", is_flag=True, help="Print verbose output.")
def main(model, dataset, all_official, num_completions, verbose):
    """Main driver for evaluation script."""
    intro(model, dataset, all_official)  # option error handling

    if (model):  # input model specified
        model = AutoModelForCausalLM.from_pretrained(model)
        # should we prompt for custom tokenizer?
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")  # page 2 of paper

    else:  # input model not specified
        # here, either we can use our own model or run through all official models; this can be done better with click later
        # TODO: potential functionality for our own model
        if (all_official):
            # official_models = [f"roneneldan/TinyStories-{n}M" for n in [1, 3, 8, 28, 33]]
            official_models = ["roneneldan/TinyStories-1M"]
            for model in official_models:
                model = AutoModelForCausalLM.from_pretrained(model)
                tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")  # page 2 of paper
                # is setting the EOS/pad token necessary?
                # is model.eval() necessary?
                

        else:  # use our own model?
            print("ERROR: functionality for our own model not implemented.", file=sys.stderr)
            return 1        

    return 0

if __name__ == "__main__":
    main()