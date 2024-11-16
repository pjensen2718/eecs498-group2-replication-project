from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import re
import csv
import numpy as np

model_name = "roneneldan/TinyStories-1M"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

model.eval()