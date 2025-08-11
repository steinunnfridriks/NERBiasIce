from datasets import Dataset
from collections import Counter
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import os
import psutil
import time
import torch

start_time = time.time()

os.environ["WANDB_DISABLED"] = "true"

# RAM
ram = psutil.virtual_memory().total / (1024 ** 3)  # in GB
print("RAM:", ram)

# Disk
disk = psutil.disk_usage('/').total / (1024 ** 3)  # in GB
print("Disk:", disk)

if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)
    print("Device:", device)

def read_conll(filepath):
    sentences, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        tokens, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                splits = line.split()
                tokens.append(splits[0])
                tags.append(splits[-1])
        if tokens:
            sentences.append(tokens)
            labels.append(tags)
    return sentences, labels

sentences, tags = read_conll("data/all_balanced.txt")

# Build label list
all_labels = [label for sent in tags for label in sent]
label_list = sorted(set(all_labels))
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# Prepare HuggingFace dataset
data = [{"tokens": s, "labels": [label2id[t] for t in l]} for s, l in zip(sentences, tags)]
dataset = Dataset.from_list(data)

tokenizer = AutoTokenizer.from_pretrained(
    "mideind/IceBERT",
    add_prefix_space=True  # <-- required for RoBERTa with pre-tokenized input
)

def tokenize_and_align_labels(example):
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    word_ids = tokenized.word_ids()
    labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(example["labels"][word_idx])
        else:
            labels.append(-100)
        previous_word_idx = word_idx
    tokenized["labels"] = labels
    return tokenized

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(-1)

    true_labels = [
        [id2label[label] for (label, pred) in zip(label_row, pred_row) if label != -100]
        for label_row, pred_row in zip(labels, predictions)
    ]
    pred_labels = [
        [id2label[pred] for (label, pred) in zip(label_row, pred_row) if label != -100]
        for label_row, pred_row in zip(labels, predictions)
    ]

    # Only keep positions where the true label is not "O"
    filtered_true = []
    filtered_pred = []
    for t_seq, p_seq in zip(true_labels, pred_labels):
        t_filtered = []
        p_filtered = []
        for t, p in zip(t_seq, p_seq):
            if t != "O":
                t_filtered.append(t)
                p_filtered.append(p)
        filtered_true.append(t_filtered)
        filtered_pred.append(p_filtered)

    macro_precision = precision_score(filtered_true, filtered_pred, average='macro')
    macro_recall = recall_score(filtered_true, filtered_pred, average='macro')
    macro_f1 = f1_score(filtered_true, filtered_pred, average='macro')

    micro_precision = precision_score(filtered_true, filtered_pred, average='micro')
    micro_recall = recall_score(filtered_true, filtered_pred, average='micro')
    micro_f1 = f1_score(filtered_true, filtered_pred, average='micro')

    # Get per-category scores
    report = classification_report(filtered_true, filtered_pred, output_dict=True)
    # Remove 'micro avg', 'macro avg', 'weighted avg' from the report
    per_category = {k: v for k, v in report.items() if k not in ['micro avg', 'macro avg', 'weighted avg', 'accuracy']}

    # Flatten per-category metrics for logging
    metrics = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }
    for tag, scores in per_category.items():
        metrics[f"{tag}_precision"] = scores["precision"]
        metrics[f"{tag}_recall"] = scores["recall"]
        metrics[f"{tag}_f1"] = scores["f1-score"]

    return metrics

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False)
# Load dev data
dev_sentences, dev_tags = read_conll("data/dev.txt")
dev_data = [{"tokens": s, "labels": [label2id[t] for t in l]} for s, l in zip(dev_sentences, dev_tags)]
dev_dataset = Dataset.from_list(dev_data)
tokenized_dev = dev_dataset.map(tokenize_and_align_labels, batched=False)

# Load test data
test_sentences, test_tags = read_conll("data/test.txt")
test_data = [{"tokens": s, "labels": [label2id[t] for t in l]} for s, l in zip(test_sentences, test_tags)]
test_dataset = Dataset.from_list(test_data)
tokenized_test = test_dataset.map(tokenize_and_align_labels, batched=False)


model = AutoModelForTokenClassification.from_pretrained(
    "mideind/IceBERT",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

training_args = TrainingArguments(
    output_dir="icebert_bias_2",
    per_device_train_batch_size=16,  # Increased for more stable gradients
    gradient_accumulation_steps=2,   # Effective batch size of 32
    num_train_epochs=5,              # More epochs for imbalanced data
    learning_rate=2e-5,              # Lower LR for fine-tuning
    weight_decay=0.01,               # Regularization
    warmup_steps=500,                # Gradual learning rate warmup
    logging_steps=50,
    evaluation_strategy="epoch",     
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=3,              # Save disk space
    fp16=True,                       # Memory efficiency
    dataloader_pin_memory=False,     # May help with memory issues
    label_smoothing_factor=0.1,      # Help with overconfident predictions
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dev,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
#results = trainer.evaluate()
#print(results)

# Save model and tokenizer
trainer.save_model("icebert_bias_2")
tokenizer.save_pretrained("icebert_bias_2")
end_time = time.time()
elapsed_time = end_time - start_time 
print("Time:", elapsed_time, "seconds")