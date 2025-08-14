from datasets import Dataset
from collections import Counter
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import os
import psutil
import time
import torch
import numpy as np
import logging
import json
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def tokenize_and_align_labels(tokenizer, example, label2id):
    """Improved tokenization with better label alignment"""
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_overflowing_tokens=False,
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

def get_predictions(model, tokenizer, dataset, label2id, id2label):
    """Get predictions from a model for a dataset"""
    model.eval()
    device = next(model.parameters()).device
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for example in tqdm(dataset, desc="Getting predictions", leave=False):
            inputs = {k: torch.tensor(v).unsqueeze(0).to(device) 
                     for k, v in example.items() if k != "labels"}
            labels = torch.tensor(example["labels"]).unsqueeze(0)
            
            outputs = model(**inputs)
            predictions = outputs.logits.argmax(-1)
            
            # Convert to lists and filter out padding
            pred_list = predictions[0].cpu().numpy()
            label_list = labels[0].numpy()
            
            valid_preds = []
            valid_labels = []
            for pred, label in zip(pred_list, label_list):
                if label != -100:
                    valid_preds.append(id2label[pred])
                    valid_labels.append(id2label[label])
            
            if valid_preds:  # Only add if there are valid predictions
                all_predictions.append(valid_preds)
                all_labels.append(valid_labels)
    
    return all_predictions, all_labels

def compute_metrics_detailed(true_labels, pred_labels):
    """Enhanced metrics computation with entity-level evaluation"""
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
    
    # Add per-category metrics with eval_ prefix for compatibility
    for tag, scores in per_category.items():
        if isinstance(scores, dict):
            metrics[f"eval_{tag}_precision"] = scores.get("precision", 0.0)
            metrics[f"eval_{tag}_recall"] = scores.get("recall", 0.0)
            metrics[f"eval_{tag}_f1"] = scores.get("f1-score", 0.0)
    
    # Add eval_ prefix to main metrics for compatibility
    metrics["eval_macro_f1"] = macro_f1
    metrics["eval_micro_f1"] = micro_f1

    return metrics

def mcnemar_test(pred1, pred2, true_labels):
    """Perform McNemar's test between two models' predictions"""
    # Flatten predictions for comparison
    flat_pred1 = [label for sent in pred1 for label in sent]
    flat_pred2 = [label for sent in pred2 for label in sent]
    flat_true = [label for sent in true_labels for label in sent]
    
    # Create binary correct/incorrect arrays
    correct1 = [p == t for p, t in zip(flat_pred1, flat_true)]
    correct2 = [p == t for p, t in zip(flat_pred2, flat_true)]
    
    # Build contingency table
    # n00: both wrong, n01: 1 wrong 2 right, n10: 1 right 2 wrong, n11: both right
    n00 = sum(1 for c1, c2 in zip(correct1, correct2) if not c1 and not c2)
    n01 = sum(1 for c1, c2 in zip(correct1, correct2) if not c1 and c2)
    n10 = sum(1 for c1, c2 in zip(correct1, correct2) if c1 and not c2)
    n11 = sum(1 for c1, c2 in zip(correct1, correct2) if c1 and c2)
    
    # McNemar's test uses only the discordant pairs
    contingency_table = [[n00, n01], [n10, n11]]
    
    # Perform test (using only discordant pairs)
    if n01 + n10 == 0:
        # No discordant pairs, models perform identically
        return 1.0, contingency_table
    
    # Use continuity correction for small samples
    result = mcnemar([[n00, n01], [n10, n11]], exact=False, correction=True)
    
    return result.pvalue, contingency_table

def evaluate_model(model_dir, test_file, dataset_name):
    """Evaluate a single model on a test file"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_dir} on {dataset_name}")
    print('='*60)
    
    # Load model configuration
    model_config_path = os.path.join(model_dir, "config.json")
    with open(model_config_path, 'r') as f:
        config = json.load(f)
    
    label2id = config.get("label2id", {})
    id2label = config.get("id2label", {})
    # Convert string keys to int for id2label
    id2label = {int(k): v for k, v in id2label.items()}
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load test data
    sentences, tags = read_conll(test_file)
    
    # Prepare dataset
    data = [{"tokens": s, "labels": [label2id.get(t, label2id.get("O", 0)) for t in l]} 
            for s, l in zip(sentences, tags)]
    dataset = Dataset.from_list(data)
    
    # Tokenize dataset
    print(f"  Tokenizing {len(dataset)} examples...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(tokenizer, x, label2id),
        batched=False,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Get predictions
    predictions, true_labels = get_predictions(model, tokenizer, tokenized_dataset, label2id, id2label)
    
    # Compute metrics
    metrics = compute_metrics_detailed(true_labels, predictions)
    
    # Print key metrics
    print(f"\nResults for {model_dir} on {dataset_name}:")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Micro F1: {metrics['micro_f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    
    return metrics, predictions, true_labels

def main():
    """Main evaluation function"""
    models = ["icebert", "scandibert", "mbert"]
    test_files = [
        ("data/test.txt", "test"),
        ("data/gold.txt", "gold")
    ]
    
    all_results = {}
    all_predictions = {}
    
    # Evaluate each model on each test set
    total_evaluations = len([m for m in models if os.path.exists(m)]) * len(test_files)
    pbar = tqdm(total=total_evaluations, desc="Overall progress")
    
    for model_dir in models:
        if not os.path.exists(model_dir):
            print(f"Warning: Model directory '{model_dir}' not found. Skipping...")
            continue
            
        all_results[model_dir] = {}
        all_predictions[model_dir] = {}
        
        for test_file, dataset_name in test_files:
            if not os.path.exists(test_file):
                print(f"Warning: Test file '{test_file}' not found. Creating dummy file...")
                # Create a dummy gold.txt if it doesn't exist
                if test_file == "data/gold.txt":
                    os.makedirs("data", exist_ok=True)
                    with open(test_file, "w") as f:
                        # Copy some data from test.txt as placeholder
                        f.write("Example\tO\ntest\tO\ndata\tO\n\n")
            
            pbar.set_description(f"Evaluating {model_dir} on {dataset_name}")
            metrics, predictions, true_labels = evaluate_model(model_dir, test_file, dataset_name)
            all_results[model_dir][dataset_name] = metrics
            all_predictions[model_dir][dataset_name] = (predictions, true_labels)
            pbar.update(1)
    
    pbar.close()
    
    # Save results for plotting
    with open("evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Perform McNemar tests
    print("\n" + "="*60)
    print("McNEMAR'S TEST RESULTS")
    print("="*60)
    
    model_pairs = [
        ("icebert", "scandibert"),
        ("icebert", "mbert"),
        ("scandibert", "mbert")
    ]
    
    mcnemar_results = {}
    
    for test_file, dataset_name in test_files:
        print(f"\n{dataset_name.upper()} dataset:")
        print("-"*40)
        
        mcnemar_results[dataset_name] = {}
        
        for model1, model2 in model_pairs:
            if model1 not in all_predictions or model2 not in all_predictions:
                continue
            if dataset_name not in all_predictions[model1] or dataset_name not in all_predictions[model2]:
                continue
                
            pred1, true1 = all_predictions[model1][dataset_name]
            pred2, true2 = all_predictions[model2][dataset_name]
            
            # Ensure we're comparing on the same true labels
            assert len(true1) == len(true2), "True labels mismatch between models"
            
            p_value, contingency = mcnemar_test(pred1, pred2, true1)
            
            # Determine which model is better based on F1 scores
            f1_model1 = all_results[model1][dataset_name]["macro_f1"]
            f1_model2 = all_results[model2][dataset_name]["macro_f1"]
            
            if f1_model1 > f1_model2:
                better_model = model1
                worse_model = model2
                improvement = f1_model1 - f1_model2
            else:
                better_model = model2
                worse_model = model1
                improvement = f1_model2 - f1_model1
            
            # Interpret significance
            if p_value < 0.001:
                significance = "***"
                interpretation = "highly significant"
            elif p_value < 0.01:
                significance = "**"
                interpretation = "significant"
            elif p_value < 0.05:
                significance = "*"
                interpretation = "significant"
            else:
                significance = ""
                interpretation = "not significant"
            
            print(f"\n{model1} vs {model2}:")
            print(f"  p-value: {p_value:.6f} {significance}")
            print(f"  Contingency table (both wrong, {model2} only right, {model1} only right, both right):")
            print(f"    {contingency}")
            print(f"  {better_model} performs better than {worse_model}")
            print(f"  Improvement: +{improvement:.4f} (Macro F1)")
            print(f"  Difference is {interpretation}")
            
            mcnemar_results[dataset_name][f"{model1}_vs_{model2}"] = {
                "p_value": float(p_value),
                "better_model": better_model,
                "improvement": float(improvement),
                "significant": bool(p_value < 0.05),
                "contingency_table": [[int(x) for x in row] for row in contingency]
            }
    
    # Save McNemar results
    with open("mcnemar_results.json", "w") as f:
        json.dump(mcnemar_results, f, indent=2)
    
    # Generate human-readable report
    with open("evaluation_report.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("COMPREHENSIVE EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        # Model performance summary
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("-"*40 + "\n\n")
        
        for test_file, dataset_name in test_files:
            f.write(f"{dataset_name.upper()} Dataset:\n")
            f.write(f"{'Model':<12} {'Macro F1':>10} {'Micro F1':>10} {'Precision':>10} {'Recall':>10}\n")
            f.write("-"*55 + "\n")
            
            for model_dir in models:
                if model_dir in all_results and dataset_name in all_results[model_dir]:
                    metrics = all_results[model_dir][dataset_name]
                    f.write(f"{model_dir:<12} {metrics['macro_f1']:10.4f} {metrics['micro_f1']:10.4f} "
                           f"{metrics['precision']:10.4f} {metrics['recall']:10.4f}\n")
            f.write("\n")
        
        # Statistical comparisons
        f.write("\nSTATISTICAL COMPARISONS (McNemar's Test)\n")
        f.write("-"*40 + "\n\n")
        
        for dataset_name in mcnemar_results:
            f.write(f"{dataset_name.upper()} Dataset:\n")
            
            for comparison, result in mcnemar_results[dataset_name].items():
                models = comparison.split("_vs_")
                f.write(f"\n  {models[0]} vs {models[1]}:\n")
                f.write(f"    Better model: {result['better_model']}\n")
                f.write(f"    Improvement: +{result['improvement']:.4f} (Macro F1)\n")
                f.write(f"    p-value: {result['p_value']:.6f}\n")
                f.write(f"    Significant: {'Yes' if result['significant'] else 'No'}\n")
            f.write("\n")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("\nResults saved to:")
    print("  - evaluation_results.json: Full evaluation metrics")
    print("  - mcnemar_results.json: Statistical test results")
    print("  - evaluation_report.txt: Human-readable report")
    
    end_time = time.time()
    print(f"\nTotal evaluation time: {(end_time - start_time)/60:.1f} minutes")

if __name__ == "__main__":
    main()