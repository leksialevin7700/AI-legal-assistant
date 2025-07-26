from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import HfApi, HfFolder, login
import torch
import os

# Set HF token (mandatory for private repos)
HUGGINGFACE_TOKEN = "your_huggingface_token_here"  # Replace with your actual Hugging Face token

os.environ["HUGGINGFACE_TOKEN"] = HUGGINGFACE_TOKEN

SUMMARIZER_REPO = "sanatann/legal-summarizer-bart"
PARAPHRASER_REPO = "Vamsi/T5_Paraphrase_Paws"

para_tokenizer = AutoTokenizer.from_pretrained(PARAPHRASER_REPO)
para_model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASER_REPO)

sum_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_REPO, token=HUGGINGFACE_TOKEN)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_REPO, token=HUGGINGFACE_TOKEN)

para_tokenizer = AutoTokenizer.from_pretrained(PARAPHRASER_REPO, token=HUGGINGFACE_TOKEN)
para_model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASER_REPO, token=HUGGINGFACE_TOKEN)


def generate_summary(text: str) -> str:
    inputs = sum_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    summary_ids = sum_model.generate(inputs["input_ids"], max_length=256, num_beams=5, early_stopping=True)
    return sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def paraphrase(text: str) -> str:
    para_input = f"paraphrase: {text} </s>"
    inputs = para_tokenizer(para_input, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    output_ids = para_model.generate(inputs["input_ids"], max_length=256, num_beams=5, early_stopping=True)
    return para_tokenizer.decode(output_ids[0], skip_special_tokens=True)


def auto_pipeline(text: str) -> tuple[str, str]:
    summary = generate_summary(text)

    # Optional: Only paraphrase if summary > 250 chars
    if len(summary) > 250:
        easy = paraphrase(summary)
    else:
        easy = summary

    return summary, easy
