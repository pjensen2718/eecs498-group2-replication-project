import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib.backends.backend_pdf import PdfPages  # Import PdfPages to save multiple pages in one PDF

model_name = "roneneldan/TinyStories-33M"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

input_sentence = "One day, Lucy asks Tom: ”I am looking for a banana but I can’t find it”. Tom says: ”Don’t"

inputs = tokenizer(input_sentence, return_tensors="pt")
outputs = model(**inputs, output_attentions=True)

attentions = outputs.attentions

for layer_to_visualize in range(4):
    attention_layer = attentions[layer_to_visualize][0].detach().numpy()  # shape: [num_heads, seq_len, seq_len]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    pdf_filename = f"visualizations/attention_heads_layer_{layer_to_visualize}.pdf"
    with PdfPages(pdf_filename) as pdf:
        num_heads = attention_layer.shape[0]
        num_plots_per_page = 4
        page_count = (num_heads // num_plots_per_page) + (1 if num_heads % num_plots_per_page != 0 else 0)

        for page_num in range(page_count):
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))  # Create a 2x2 grid for 4 heads
            axes = axes.flatten()

            start_idx = page_num * num_plots_per_page
            end_idx = min((page_num + 1) * num_plots_per_page, num_heads)

            for head_idx in range(start_idx, end_idx):
                ax = axes[head_idx - start_idx]
                attention = attention_layer[head_idx]
                sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, cmap="viridis", square=True, cbar=True, ax=ax)
                ax.set_title(f"Head {head_idx}")

            plt.suptitle(f"Attention Heads in Layer {layer_to_visualize} (Page {page_num + 1})", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            pdf.savefig(fig)
            plt.close()
