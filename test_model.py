import os
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

os.environ["WANDB_DISABLED"] = "true"

model_dir = "icebert_bias_2"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

def get_bio_tags(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, is_split_into_words=False)
    with torch.no_grad():
        outputs = model(**tokens)
    predictions = torch.argmax(outputs.logits, dim=2)
    predictions_list = predictions[0].flatten().tolist()

    labels = [model.config.id2label[p] for p in predictions_list]

    # Try to get word_ids, fallback if not available
    word_ids = tokens.word_ids(batch_index=0) if hasattr(tokens, "word_ids") else None
    if word_ids is None:
        word_ids = list(range(len(labels)))
    if len(word_ids) != len(labels):
        # Pad or truncate to match
        min_len = min(len(word_ids), len(labels))
        word_ids = word_ids[:min_len]
        labels = labels[:min_len]

    bio_tags = []
    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        label = labels[idx]
        token_id = int(tokens.input_ids[0][idx])
        bio_tags.append((tokenizer.convert_ids_to_tokens(token_id), label))
        previous_word_idx = word_idx
    return bio_tags

def test_bio_tags():
    text = "Gamli hommatitturinn og ljóta jussan með græna hárið eru fínt par. Júðasvínin sem þau eru."
    bio_tags = get_bio_tags(text)
    print("Tokens with BIO tags:")
    for token, tag in bio_tags:
        print(f"{token}\t{tag}")

if __name__ == "__main__":
    test_bio_tags()