"""
Evaluates a trained NER model on a test dataset and provides detailed metrics.

- Per-entity-type F1 scores (Person, Location)
- Overall precision, recall, F1
- Detailed classification report
- Per-class token accuracy
- Optional sample predictions display

Quickstart usage:
    # Evaluate using experiment shortcut:
    python scripts/evaluate/evaluate_ner.py --experiment normalised/norm_menota_ihpc

    # Custom paths:
    python scripts/evaluate/evaluate_ner.py \\
        --model models/my_model \\
        --test-file data/normalised/normalised_ner_data/test/norm_test.txt \\
        --output results.txt
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Read the CoNLL-formatted data (i.e. BIO-format) and convert to tokens and labels
def read_conll_file(filepath: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Read CoNLL file and return lists of tokens and labels."""
    all_tokens = []
    all_labels = []
    current_tokens = []
    current_labels = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line == '':
                if current_tokens:
                    all_tokens.append(current_tokens)
                    all_labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
            else:
                parts = line.split('\t')
                if len(parts) >= 2:
                    current_tokens.append(parts[0])
                    current_labels.append(parts[1])

    if current_tokens:
        all_tokens.append(current_tokens)
        all_labels.append(current_labels)

    return all_tokens, all_labels

# Initialise dataset
class NERDataset(Dataset):
    """Dataset for token classification."""

    def __init__(
        self,
        tokens: List[List[str]],
        labels: List[List[str]],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 512
    ):
        self.tokens = tokens
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                if word_idx < len(labels):
                    aligned_labels.append(self.label2id.get(labels[word_idx], 0))
                else:
                    aligned_labels.append(-100)
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

# Get sample predictions and compare to GT
def show_sample_predictions(
    model,
    test_dataset,
    tokenizer,
    label_list: List[str],
    num_samples: int = 5
):
    """Display sample predictions vs ground truth."""
    print("\n" + "="*70)
    print("Sample Predictions vs Ground Truth")
    print("="*70)

    model.eval()
    device = next(model.parameters()).device

    indices = random.sample(range(len(test_dataset)),
                           min(num_samples, len(test_dataset)))

    for idx in indices:
        sample = test_dataset[idx]

        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        true_labels = sample['labels']

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu()

        # Decode tokens and align predictions
        tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])

        print(f"\nSample {idx}:")
        print(f"{'Token':<25} {'True':<20} {'Pred':<20} {'Match'}")
        print("-" * 70)

        for i, (tok, true_id, pred_id) in enumerate(zip(tokens, true_labels, predictions)):
            if true_id == -100:
                continue
            if tok in ['<s>', '</s>', '<pad>']:
                continue

            true_label = label_list[true_id]
            pred_label = label_list[pred_id]
            match = "✓" if true_label == pred_label else "✗"

            # Highlight mismatches
            if true_label != pred_label:
                print(f"{tok:<25} {true_label:<20} {pred_label:<20} {match} ← MISMATCH")
            else:
                print(f"{tok:<25} {true_label:<20} {pred_label:<20} {match}")

    print("="*70 + "\n")

# Evaluate model
def evaluate_model(
    model_path: str,
    test_file: str,
    output_file: str = None,
    show_samples: bool = False,
    num_samples: int = 5,
    max_length: int = 512,
    batch_size: int = 16
):
    """
    Evaluate trained NER model on test set data.

    Args:
        model_path: Path to saved model directory
        test_file: Path to CoNLL-formatted test file
        output_file: Optional path to save results
        show_samples: Show sample predictions
        num_samples: Number of sample predictions to show
        max_length: Maximum sequence length
        batch_size: Batch size for evaluation
    """
    print("="*70)
    print("NER Model Evaluation")
    print("="*70)

    # Load label mapping
    print(f"\n1. Loading model from: {model_path}")
    label_map_path = os.path.join(model_path, 'label_map.json')

    if os.path.exists(label_map_path):
        with open(label_map_path, 'r') as f:
            label_data = json.load(f)
            label2id = label_data['label2id']
            id2label = {int(k): v for k, v in label_data['id2label'].items()}
            label_list = label_data['label_list']
    else:
        print(f"   Warning: label_map.json not found, loading from model config")
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        label2id = model.config.label2id
        id2label = model.config.id2label
        label_list = [id2label[i] for i in range(len(id2label))]

    print(f"   Labels: {label_list}")
    print(f"   Number of labels: {len(label_list)}")

    # Load tokenizer and model
    print(f"\n2. Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"   Device: {device}")

    # Load test data
    print(f"\n3. Loading test data from: {test_file}")
    test_tokens, test_labels = read_conll_file(test_file)
    print(f"   Test sentences: {len(test_tokens)}")

    # Analyze test data distribution
    flat_labels = [l for sent in test_labels for l in sent]
    label_counts = Counter(flat_labels)
    total_tokens = len(flat_labels)

    print(f"\n   Label distribution in test data:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = 100 * count / total_tokens
        print(f"     {label:<20}: {count:>8,} ({pct:5.2f}%)")

    # Create dataset
    print(f"\n4. Creating test dataset...")
    test_dataset = NERDataset(test_tokens, test_labels, tokenizer, label2id, max_length)

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Create trainer for evaluation
    training_args = TrainingArguments(
        output_dir='./tmp_eval',
        per_device_eval_batch_size=batch_size,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
    )

    # Run predictions
    print(f"\n5. Running evaluation...")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=2)

    # Align predictions with labels
    true_labels = []
    true_predictions = []

    # Also track per-class accuracy to get PER / LOC comparison
    correct_by_class = Counter()
    total_by_class = Counter()

    for pred_seq, label_seq in zip(preds, predictions.label_ids):
        pred_labels = []
        gold_labels = []

        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id != -100:
                pred_label = label_list[pred_id]
                gold_label = label_list[label_id]
                pred_labels.append(pred_label)
                gold_labels.append(gold_label)

                total_by_class[gold_label] += 1
                if pred_label == gold_label:
                    correct_by_class[gold_label] += 1

        true_labels.append(gold_labels)
        true_predictions.append(pred_labels)

    # Compute metrics
    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)

    # Prepare output
    output_lines = []
    output_lines.append("="*70)
    output_lines.append("EVALUATION RESULTS")
    output_lines.append("="*70)
    output_lines.append(f"\nModel: {model_path}")
    output_lines.append(f"Test file: {test_file}")
    output_lines.append(f"Test sentences: {len(test_tokens)}")
    output_lines.append(f"Test tokens: {total_tokens}")

    output_lines.append(f"\n--- Overall Metrics ---")
    output_lines.append(f"Precision: {precision:.4f}")
    output_lines.append(f"Recall:    {recall:.4f}")
    output_lines.append(f"F1 Score:  {f1:.4f}")

    output_lines.append(f"\n--- Per-Class Token Accuracy ---")
    for label in sorted(total_by_class.keys()):
        acc = correct_by_class[label] / total_by_class[label] if total_by_class[label] > 0 else 0
        output_lines.append(
            f"{label:<20}: {acc*100:5.1f}% ({correct_by_class[label]:>6,}/{total_by_class[label]:>6,})"
        )

    # Check if model is predicting mostly O
    o_predictions = sum(1 for sent in true_predictions for l in sent if l == 'O')
    total_predictions = sum(len(sent) for sent in true_predictions)
    o_pred_pct = 100 * o_predictions / total_predictions if total_predictions > 0 else 0

    if o_pred_pct > 95:
        output_lines.append(f"\n  WARNING: Model predicting 'O' for {o_pred_pct:.1f}% of tokens!")

    output_lines.append(f"\n--- Classification Report (seqeval) ---")
    detailed_report = classification_report(true_labels, true_predictions)
    output_lines.append(detailed_report)

    output_lines.append("="*70)

    # Print results
    output_text = '\n'.join(output_lines)
    print(output_text)

    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\nResults saved to: {output_file}")

    # Show sample predictions if requested
    if show_samples:
        show_sample_predictions(model, test_dataset, tokenizer, label_list, num_samples)

    # Save results as JSON file
    results_dict = {
        'model_path': model_path,
        'test_file': test_file,
        'num_sentences': len(test_tokens),
        'num_tokens': total_tokens,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'per_class_accuracy': {
            label: float(correct_by_class[label] / total_by_class[label])
            for label in total_by_class.keys()
        },
        'label_distribution': {
            label: int(count) for label, count in label_counts.items()
        }
    }

    if output_file:
        json_file = output_file.replace('.txt', '.json')
        with open(json_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"Results JSON saved to: {json_file}")

    return results_dict

# Main fuction
def main():
    parser = argparse.ArgumentParser(
        description='Evaluate NER model on test data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate using predefined experiment:
    python scripts/evaluate/evaluate_ner.py --experiment normalised/norm_menota_ihpc

    # Custom paths:
    python scripts/evaluate/evaluate_ner.py \\
        --model models/my_model \\
        --test-file data/normalised/normalised_ner_data/test/norm_test.txt
        """
    )

    # Predefined experiment
    parser.add_argument(
        '--experiment',
        type=str,
        help='Predefined experiment (e.g., "normalised/norm_menota_ihpc")'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Path to saved model directory'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        help='Path to CoNLL test file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save evaluation results (optional)'
    )
    parser.add_argument(
        '--show-samples',
        action='store_true',
        help='Show sample predictions'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of sample predictions to show (default: 5)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length (default: 512)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for evaluation (default: 16)'
    )

    args = parser.parse_args()

    # If experiment specified, create paths automatically
    if args.experiment:
        exp_parts = args.experiment.split('/')
        if len(exp_parts) != 2:
            parser.error(f"Experiment format should be 'type/name', got: {args.experiment}")

        exp_type, exp_name = exp_parts

        # Validate experiment type
        if exp_type not in ['normalised', 'diplomatic']:
            parser.error(f"Experiment type must be 'normalised' or 'diplomatic', got: {exp_type}")

        # Create paths
        base = Path('data') / exp_type / f'{exp_type}_ner_data'

        args.model = args.model or str(Path('models') / exp_name)
        args.test_file = args.test_file or str(base / 'test' / f'{exp_type[:4]}_test.txt')

    # Validate required arguments
    if not args.model:
        parser.error("Either --experiment or --model is required")

    if not args.test_file:
        parser.error("Either --experiment or --test-file is required")

    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model path does not exist: {args.model}")
        sys.exit(1)

    if not os.path.exists(args.test_file):
        print(f"Error: Test file does not exist: {args.test_file}")
        sys.exit(1)

    # Run evaluation
    evaluate_model(
        model_path=args.model,
        test_file=args.test_file,
        output_file=args.output,
        show_samples=args.show_samples,
        num_samples=args.num_samples,
        max_length=args.max_length,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
