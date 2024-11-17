"""Visualizes the attention heads an arbitrary TinyStories model."""

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib.backends.backend_pdf import PdfPages

model_name = "roneneldan/TinyStories-33M"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# input_seq = """One day, Lucy asks Tom: ”I am looking for a banana but I can’t find it”. Tom says: ”Don’t worry, I will help you”.
# Lucy and Tom go to the park. They look for the banana together. After a while, they found the banana. Lucy is happy. She says:"""

input_seq = "Once upon a time, in a big forest, there lived a rhinoceros named Roxy. Roxy loved to climb. She climbed trees, rocks, and"

inputs = tokenizer(input_seq, return_tensors="pt")

# Finds the token that is most attented in every attention head.
def find_most_attended_to_tokens(attentions):
    for layer_idx in range(len(attentions)):
        attention_layer = attentions[layer_idx][0]
        for head_idx in range(len(attention_layer)):
            head = attention_layer[head_idx].detach().numpy()
            
            # How much do all other tokens attend to some token T? 
            # Sum of weights assigned to T by other tokens.
            column_sums = np.sum(head, axis=0)

            # For a given token T, how many tokens can attend to it? A token can only be attended by itself and the tokens after it.
            num_possible = np.arange(len(head), 0, -1)
            
            # Index of the token that is attended to the most, on average.
            max_index = np.argmax(column_sums / num_possible)

            # Translate the index back to a token.
            token_id = inputs['input_ids'][0][max_index].item()
            token = tokenizer.decode(token_id)

            print(f"Layer {layer_idx}, Head {head_idx}: {token}")

def plot_attention_heatmaps(attentions):
    filename = f"visualizations/attention_heads_heatmaps.pdf"
    
    with PdfPages(filename) as pdf:
        for layer_to_visualize in range(len(attentions)):
            attention_layer = attentions[layer_to_visualize][0].detach().numpy()
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            num_heads = attention_layer.shape[0]
            filename = f"visualizations/attention_heads_layer_{layer_to_visualize}.pdf"

            for head_idx in range(num_heads):
                fig, ax = plt.subplots(figsize=(18, 18))
                attention = attention_layer[head_idx]

                sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, cmap="viridis", square=True, cbar=True, ax=ax)
                ax.set_title(f"Head {head_idx}")

                plt.suptitle(f"Layer {layer_to_visualize + 1}, Attention Head {head_idx}", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.96])

                pdf.savefig(fig)
                plt.close()

def visualize_attention_with_arrows(tokens, attentions, figsize=(10, 12)):
    filename = f"visualizations/attention_heads_arrows.pdf"
    
    with PdfPages(filename) as pdf:
        for layer_to_visualize in range(len(attentions)):
            attention_layer = attentions[layer_to_visualize][0].detach().numpy()
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            num_heads = attention_layer.shape[0]

            for head_idx in range(num_heads):
                attention_matrix = attention_layer[head_idx]

                fig, ax = plt.subplots(figsize=figsize)
                
                token_count = len(tokens)
                
                # Each token is assigned a position on the y-axis from 0 to len(tokens) - 1.
                y_positions = np.arange(token_count)
                
                # Source tokens placed at x = 0, destination tokens placed at x = 1.
                ax.scatter(np.zeros(token_count), y_positions, alpha=0)
                ax.scatter(np.ones(token_count), y_positions, alpha=0)
                
                # Label source and destination tokens.
                for idx, token in enumerate(tokens):
                    ax.text(-0.1, idx, token, ha='right', va='center', fontsize=10)
                    ax.text(1.1, idx, token, ha='left', va='center', fontsize=10)
                
                # An arrow from a source token to a destination token indicates that the source 
                # token is attending (or paying attention) to the destination token.

                # We only plot the arrow for the strongest attention connection (to minimize clutter).
                for row_idx in range(token_count):
                    # Find the token that receives the strongest attention from source token i
                    max_attention_idx = np.argmax(attention_matrix[row_idx])
                    max_attention_weight = attention_matrix[row_idx][max_attention_idx]
                    
                    # Plot the arrow. The thickness of the arrow scales with the strength of the connection.
                    plt.plot([0, 1], [row_idx, max_attention_idx], color='red', alpha=min(max_attention_weight * 2, 0.9),
                                    linewidth=max_attention_weight * 3, zorder=1)
                
                ax.set_xlim(-0.2, 1.2)
                ax.set_ylim(token_count, -1)
                
                ax.set_xticks([])
                ax.set_yticks([])

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                plt.suptitle(f"Layer {layer_to_visualize + 1}, Attention Head {head_idx}", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.96])

                pdf.savefig(fig)
                plt.close()

def main():
    # Unpacks inputs, which contains tokens and attention mask.
    # Output contains logits and attentions.
        # Logits -> tensor of shape [batch_size, len(input_seq), vocab_size]
        # Attention -> tensor of shape [num_layers, batch_size, num_heads, len(input_seq), len(input_seq)]
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions

    plot_attention_heatmaps(attentions)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    visualize_attention_with_arrows(tokens, attentions)

main()
