from datasets import Dataset
from collections import Counter
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import os
import psutil
import time
import torch
import numpy as np
import logging
import argparse
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train or evaluate bias detection model')
parser.add_argument('--eval-only', action='store_true', help='Only evaluate, do not train')
parser.add_argument('--model-dir', type=str, default='icebert', help='Directory containing pre-trained model')
parser.add_argument('--output-dir', type=str, default='.', help='Directory to save results')
args = parser.parse_args()

start_time = time.time()

os.environ["WANDB_DISABLED"] = "true"

# System info
ram = psutil.virtual_memory().total / (1024 ** 3)
print(f"RAM: {ram:.1f} GB")

disk = psutil.disk_usage('/').total / (1024 ** 3)
print(f"Disk: {disk:.1f} GB")

if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)
    print(f"Device: {device}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def read_conll(filepath):
    """Read CoNLL format file with tab separation and better error handling"""
    sentences, labels = [], []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tokens, tags = [], []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    if tokens:
                        sentences.append(tokens)
                        labels.append(tags)
                        tokens, tags = [], []
                else:
                    # Split by tab first, then fall back to space if no tabs
                    if '\t' in line:
                        splits = line.split('\t')
                    else:
                        splits = line.split()
                    
                    if len(splits) < 2:
                        logger.warning(f"Malformed line {line_num} in {filepath}: {line}")
                        continue
                    
                    # For standard CoNLL: word \t tag
                    tokens.append(splits[0])
                    tags.append(splits[1])  # Use second column (index 1) for tag
            
            # Don't forget the last sentence if file doesn't end with empty line
            if tokens:
                sentences.append(tokens)
                labels.append(tags)
                
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error reading {filepath}: {str(e)}")
        raise
    
    logger.info(f"Loaded {len(sentences)} sentences from {filepath}")
    return sentences, labels

def analyze_label_distribution(tags, dataset_name=""):
    """Analyze and print label distribution"""
    all_labels = [label for sent in tags for label in sent]
    label_counts = Counter(all_labels)
    
    print(f"\n{dataset_name} Label Distribution:")
    print("-" * 40)
    for label, count in sorted(label_counts.items()):
        print(f"{label:20} {count:8d}")
    
    return label_counts

# Load data
if not args.eval_only:
    sentences, tags = read_conll("data/all_balanced.txt")
    dev_sentences, dev_tags = read_conll("data/dev.txt")
    # Analyze distributions
    train_counts = analyze_label_distribution(tags, "Training")
    dev_counts = analyze_label_distribution(dev_tags, "Development")
    
test_sentences, test_tags = read_conll("data/test.txt")

# Build label list
if args.eval_only:
    # Load label mappings from pre-trained model
    model_config_path = os.path.join(args.model_dir, "config.json")
    with open(model_config_path, 'r') as f:
        config = json.load(f)
    label2id = config.get("label2id", {})
    id2label = config.get("id2label", {})
    # Convert string keys to int for id2label
    id2label = {int(k): v for k, v in id2label.items()}
    label_list = sorted(label2id.keys())
else:
    all_labels = [label for sent in tags for label in sent]
    label_list = sorted(set(all_labels))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

print(f"\nTotal labels: {len(label_list)}")
print(f"Labels: {label_list}")

# Calculate class weights for imbalanced data (only for training)
if not args.eval_only:
    flat_labels = [label2id[label] for label in all_labels]
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(flat_labels),
        y=flat_labels
    )
    class_weights_tensor = torch.FloatTensor(class_weights)
    print(f"\nClass weights computed: {len(class_weights)} classes")
else:
    class_weights_tensor = None

# Custom Trainer with weighted loss
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device),
                ignore_index=-100
            )
        else:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                       labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# Initialize tokenizer
if args.eval_only:
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
else:
    tokenizer = AutoTokenizer.from_pretrained(
        "vesteinn/ScandiBERT",
        add_prefix_space=True
    )

def tokenize_and_align_labels(example):
    """Improved tokenization with better label alignment"""
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_overflowing_tokens=False,  # Handle long sequences better
        return_offsets_mapping=False
    )
    
    word_ids = tokenized.word_ids()
    labels = []
    previous_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is None:
            # Special tokens get -100
            labels.append(-100)
        elif word_idx != previous_word_idx:
            # First subtoken of a word gets the label
            if word_idx < len(example["labels"]):
                labels.append(example["labels"][word_idx])
            else:
                labels.append(-100)
        else:
            # Subsequent subtokens get -100 (ignored in loss)
            labels.append(-100)
        previous_word_idx = word_idx
    
    tokenized["labels"] = labels
    return tokenized

def compute_metrics(p):
    """Enhanced metrics computation with entity-level evaluation"""
    predictions, labels = p
    predictions = predictions.argmax(-1)

    # Convert to string labels, filtering out ignored tokens
    true_labels = []
    pred_labels = []
    
    for label_row, pred_row in zip(labels, predictions):
        true_seq = []
        pred_seq = []
        for label, pred in zip(label_row, pred_row):
            if label != -100:
                true_seq.append(id2label[label])
                pred_seq.append(id2label[pred])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq)

    # Compute standard seqeval metrics (entity-level)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    
    # Compute macro averages (per-class)
    macro_precision = precision_score(true_labels, pred_labels, average='macro')
    macro_recall = recall_score(true_labels, pred_labels, average='macro')
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    
    # Compute micro averages (overall performance across all entities)
    micro_precision = precision_score(true_labels, pred_labels, average='micro')
    micro_recall = recall_score(true_labels, pred_labels, average='micro')
    micro_f1 = f1_score(true_labels, pred_labels, average='micro')

    # Get detailed classification report
    try:
        report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
        per_category = {k: v for k, v in report.items() 
                       if k not in ['micro avg', 'macro avg', 'weighted avg', 'accuracy']}
    except:
        per_category = {}

    # Compile metrics
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }
    
    # Add per-category metrics
    for tag, scores in per_category.items():
        if isinstance(scores, dict):
            metrics[f"{tag}_precision"] = scores.get("precision", 0.0)
            metrics[f"{tag}_recall"] = scores.get("recall", 0.0)
            metrics[f"{tag}_f1"] = scores.get("f1-score", 0.0)

    return metrics

# Prepare datasets
print("\nTokenizing datasets...")
if not args.eval_only:
    data = [{"tokens": s, "labels": [label2id[t] for t in l]} for s, l in zip(sentences, tags)]
    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False, remove_columns=dataset.column_names)

    dev_data = [{"tokens": s, "labels": [label2id[t] for t in l]} for s, l in zip(dev_sentences, dev_tags)]
    dev_dataset = Dataset.from_list(dev_data)
    tokenized_dev = dev_dataset.map(tokenize_and_align_labels, batched=False, remove_columns=dev_dataset.column_names)

test_data = [{"tokens": s, "labels": [label2id[t] for t in l]} for s, l in zip(test_sentences, test_tags)]
test_dataset = Dataset.from_list(test_data)
tokenized_test = test_dataset.map(tokenize_and_align_labels, batched=False, remove_columns=test_dataset.column_names)

if args.eval_only:
    print(f"Test dataset size: {len(tokenized_test)}")

# Load model
if args.eval_only:
    print(f"\nLoading pre-trained model from {args.model_dir}")
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
else:
    model = AutoModelForTokenClassification.from_pretrained(
        "vesteinn/ScandiBERT",
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

# Check if CUDA is available for FP16
use_fp16 = torch.cuda.is_available()

# Training arguments
if args.eval_only:
    # Minimal arguments for evaluation only
    training_args = TrainingArguments(
        output_dir="./eval_output",
        per_device_eval_batch_size=32,
        eval_strategy="no",              # No evaluation during training (we're not training)
        report_to="none",                # Disable all logging integrations
        fp16=use_fp16,                   # Only use FP16 if CUDA is available
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        use_mps_device=False,            # Disable MPS to avoid issues
    )
else:
    # Full training arguments
    training_args = TrainingArguments(
        output_dir="scandibert_bias_2",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,    # Larger for eval since no gradients
        gradient_accumulation_steps=2,    # Effective batch size of 32
        num_train_epochs=8,               # More epochs for better convergence
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,                 # 10% warmup
        lr_scheduler_type="cosine",       # Better than linear for longer training
        logging_steps=50,
        eval_strategy="steps",            # More frequent evaluation
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=3,
        fp16=use_fp16,                   # Only use FP16 if CUDA is available
        dataloader_pin_memory=False,
        label_smoothing_factor=0.1,
        report_to="none",                 # Disable wandb completely
        seed=42,                          # Reproducibility
        data_seed=42,
        remove_unused_columns=True,       # Remove original columns after tokenization
        use_mps_device=False,            # Disable MPS to avoid issues
    )

# Initialize trainer
if args.eval_only:
    # For evaluation only, we don't need training dataset or callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,  # Updated from tokenizer to processing_class
        compute_metrics=compute_metrics,
    )
else:
    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dev,
        processing_class=tokenizer,  # Updated from tokenizer to processing_class
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    print("\nStarting training...")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Total optimization steps: ~{len(tokenized_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
    
    # Train the model
    trainer.train()
    
    # Final evaluation on dev set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON DEV SET")
    print("="*50)
    dev_results = trainer.evaluate()
    for key, value in dev_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")

# Evaluate on test set
print("\n" + "="*50)
print("EVALUATION ON TEST SET")
print("="*50)
test_results = trainer.evaluate(tokenized_test)
for key, value in test_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")

# Save model and results
if args.eval_only:
    # Save test results to JSON for plotting
    output_file = os.path.join(args.output_dir, "test_results.json")
    print(f"\nSaving test results to {output_file}")
    with open(output_file, "w") as f:
        json.dump(test_results, f, indent=2)
    
    # Also save a human-readable report
    report_file = os.path.join(args.output_dir, "evaluation_report.txt")
    with open(report_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("TEST SET EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        
        # Overall metrics
        f.write("Overall Metrics:\n")
        f.write("-"*30 + "\n")
        for key in ["eval_loss", "eval_macro_precision", "eval_macro_recall", "eval_macro_f1", 
                   "eval_micro_precision", "eval_micro_recall", "eval_micro_f1"]:
            if key in test_results:
                metric_name = key.replace("eval_", "").replace("_", " ").title()
                f.write(f"{metric_name:20} {test_results[key]:.4f}\n")
        
        f.write("\n\nPer-Category Metrics:\n")
        f.write("-"*30 + "\n")
        
        # Collect per-category metrics
        categories = set()
        for key in test_results.keys():
            if "_precision" in key or "_recall" in key or "_f1" in key:
                if not any(x in key for x in ["macro", "micro", "eval_precision", "eval_recall", "eval_f1"]):
                    cat = key.replace("eval_", "").replace("_precision", "").replace("_recall", "").replace("_f1", "")
                    categories.add(cat)
        
        # Sort by F1 score
        cat_f1_scores = [(cat, test_results.get(f"eval_{cat}_f1", 0)) for cat in categories]
        sorted_cats = sorted(cat_f1_scores, key=lambda x: x[1], reverse=True)
        
        f.write(f"{'Category':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
        f.write("-"*50 + "\n")
        for cat, _ in sorted_cats:
            prec = test_results.get(f"eval_{cat}_precision", 0)
            rec = test_results.get(f"eval_{cat}_recall", 0)
            f1 = test_results.get(f"eval_{cat}_f1", 0)
            f.write(f"{cat:<15} {prec:10.4f} {rec:10.4f} {f1:10.4f}\n")
    
    print(f"Evaluation report saved to {report_file}")
else:
    print("\nSaving model...")
    trainer.save_model("scandibert_bias_2")
    tokenizer.save_pretrained("scandibert_bias_2")
    
    # Save training log
    log_data = {
        "training_time": time.time() - start_time,
        "final_dev_results": dev_results,
        "final_test_results": test_results,
        "training_args": training_args.to_dict(),
        "label_distribution": dict(train_counts),
        "num_labels": len(label_list),
    }
    
    with open("scandibert_bias_2/training_log.json", "w") as f:
        json.dump(log_data, f, indent=2)

end_time = time.time()
elapsed_time = end_time - start_time
if args.eval_only:
    print(f"\nTotal evaluation time: {elapsed_time:.1f} seconds")
else:
    print(f"\nTotal training time: {elapsed_time/3600:.2f} hours ({elapsed_time:.1f} seconds)")
    print(f"Model saved to: scandibert_bias_2/")

# Memory cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"CUDA memory cleared")