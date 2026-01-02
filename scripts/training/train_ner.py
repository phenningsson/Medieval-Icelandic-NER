"""
Trains a NER model on Medieval Icelandic data. 
Diagnostics and evaluation is part of the code and will run both during and after training.
For standalone evaluation, use the evaluate_ner.py script.

- Evaluates every 10% of training (configurable)
- Class weights are used to handle imbalanced data (default 30.0 for entites, 0.1 for non-entities)
- Logs predictions during evaluation to help troubleshooting
- Metrics for each entity type

Quickstart usage:
    # Use predefined experiments:
    python scripts/training/train_ner.py --experiment normalised/norm_menota_ihpc

    # Custom paths:
    python scripts/training/train_ner.py \\
        --train-file path/to/train.txt \\
        --dev-file path/to/dev.txt \\
        --output-dir models/my_model

    # Custom hyperparameters example:
    python scripts/training/train_ner.py \\
        --experiment normalised/norm_menota_ihpc \\
        --epochs 10 \\
        --batch-size 32 \\
        --learning-rate 3e-5
"""

import os
import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForTokenClassification,
    TrainerCallback,
)
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Data paths, use custom paths in CLI to override
TRAIN_FILE = 'data/normalised/normalised_ner_data/norm_menota/menota_only.txt'
DEV_FILE = 'data/normalised/normalised_ner_data/dev/norm_dev.txt'
TEST_FILE = 'data/normalised/normalised_ner_data/test/norm_test.txt'

# Model configuration, use custom paths in CLI to override
MODEL_NAME = 'mideind/IceBERT'
OUTPUT_DIR = 'models/model_output'

# Training hyperparameters
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEPS = 1
EARLY_STOPPING_PATIENCE = 3

# Evaluation frequency: evaluate every X% of training, standard is every 10%
EVAL_PERCENT = 10

# Use class weights to handle imbalanced data, change to False to disable or change values
USE_CLASS_WEIGHTS = True
O_CLASS_WEIGHT = 0.1
ENTITY_CLASS_WEIGHT = 30.0

# Use FP16 mixed precision to manage memory and faster computation
USE_FP16 = True

# Show sample predictions during evaluation to help troubleshoot
SHOW_SAMPLE_PREDICTIONS = True
NUM_SAMPLES_TO_SHOW = 5

# Parse CLI arguments
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train NER model on Medieval Icelandic data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        
Examples:
    # Train on predefined experiment:
    python scripts/training/train_ner.py --experiment normalised/norm_menota_ihpc

    # Train with custom paths:
    python scripts/training/train_ner.py \\
        --train-file data/normalised/normalised_ner_data/norm_menota/menota_only.txt \\
        --dev-file data/normalised/normalised_ner_data/dev/norm_dev.txt \\
        --output-dir models/my_model

    # Override hyperparameters:
    python scripts/training/train_ner.py \\
        --experiment normalised/norm_menota_ihpc \\
        --epochs 10 \\
        --batch-size 32
        """
    )

    # Predefined experiment
    parser.add_argument(
        '--experiment',
        type=str,
        help='Predefined experiment (e.g., "normalised/norm_menota_ihpc")'
    )

    # Data paths
    parser.add_argument('--train-file', type=str, help='Training data file')
    parser.add_argument('--dev-file', type=str, help='Development data file')
    parser.add_argument('--test-file', type=str, help='Test data file')
    parser.add_argument('--output-dir', type=str, help='Output directory for model')

    # Model configuration
    parser.add_argument(
        '--model-name',
        type=str,
        default=MODEL_NAME,
        help=f'Pre-trained model name (default: {MODEL_NAME})'
    )

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=EPOCHS, help=f'Number of epochs (default: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--max-length', type=int, default=MAX_LENGTH, help=f'Max sequence length (default: {MAX_LENGTH})')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY, help=f'Weight decay (default: {WEIGHT_DECAY})')
    parser.add_argument('--warmup-ratio', type=float, default=WARMUP_RATIO, help=f'Warmup ratio (default: {WARMUP_RATIO})')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument('--early-stopping-patience', type=int, default=EARLY_STOPPING_PATIENCE)

    # Advanced options
    parser.add_argument('--no-class-weights', action='store_true', help='Disable class weights')
    parser.add_argument('--no-fp16', action='store_true', help='Disable FP16 mixed precision')
    parser.add_argument('--no-sample-predictions', action='store_true', help='Disable sample predictions logging')
    parser.add_argument('--num-samples-to-show', type=int, default=NUM_SAMPLES_TO_SHOW, help=f'Number of sample predictions to show (default: {NUM_SAMPLES_TO_SHOW})')

    args = parser.parse_args()

    # If experiment is specified, create paths automatically
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

        args.train_file = args.train_file or str(base / exp_name / 'train.txt')
        args.dev_file = args.dev_file or str(base / 'dev' / f'{exp_type[:4]}_dev.txt')
        args.test_file = args.test_file or str(base / 'test' / f'{exp_type[:4]}_test.txt')
        args.output_dir = args.output_dir or str(Path('models') / exp_name)

    # Validate required arguments
    if not args.train_file:
        parser.error("Either --experiment or --train-file is required")

    if not args.output_dir:
        parser.error("Either --experiment or --output-dir is required")

    # Check if files exist
    if not Path(args.train_file).exists():
        print(f"\nERROR: Training file not found: {args.train_file}")
        print(f"\nAvailable experiments in data/:")
        for exp_type in ['normalised', 'diplomatic']:
            exp_dir = Path('data') / exp_type / f'{exp_type}_ner_data'
            if exp_dir.exists():
                print(f"\n  {exp_type}:")
                for subdir in sorted(exp_dir.iterdir()):
                    if subdir.is_dir() and subdir.name not in ['dev', 'test', 'config']:
                        # Check if directory has any .txt files
                        txt_files = list(subdir.glob('*.txt'))
                        if txt_files:
                            print(f"    - {exp_type}/{subdir.name}")
        sys.exit(1)

    return args

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

# Build list of unique labels
def build_label_list(*filepaths) -> List[str]:
    """Build sorted list of unique labels from data files."""
    labels = set()
    
    for filepath in filepaths:
        if filepath and os.path.exists(filepath):
            _, file_labels = read_conll_file(filepath)
            for sentence_labels in file_labels:
                labels.update(sentence_labels)
    
    label_list = sorted(labels, key=lambda x: (
        0 if x == 'O' else (1 if x.startswith('B-') else 2),
        x
    ))
    
    return label_list

# Get class weights
def compute_class_weights(train_labels: List[List[str]], label2id: Dict[str, int]) -> torch.Tensor:
    """Compute class weights based on label frequency."""
    flat_labels = [l for sent in train_labels for l in sent]
    label_counts = Counter(flat_labels)
    
    weights = torch.ones(len(label2id))
    
    for label, idx in label2id.items():
        if label == 'O':
            weights[idx] = O_CLASS_WEIGHT
        else:
            weights[idx] = ENTITY_CLASS_WEIGHT
    
    return weights

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

# Trainer using weighted loss
class WeightedLossTrainer(Trainer):
    """Trainer with weighted loss for imbalanced classes."""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# Logging to help troubleshoot
class PredictionLoggingCallback(TrainerCallback):
    """Callback to log sample predictions during evaluation."""
    
    def __init__(self, eval_dataset, tokenizer, label_list, num_samples=5):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.num_samples = num_samples
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or not SHOW_SAMPLE_PREDICTIONS:
            return
        
        print("\n" + "="*60)
        print("Sample Predictions vs Ground Truth")
        print("="*60)
        
        model.eval()
        device = next(model.parameters()).device
        
        indices = random.sample(range(len(self.eval_dataset)), 
                               min(self.num_samples, len(self.eval_dataset)))
        
        for idx in indices:
            sample = self.eval_dataset[idx]
            
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            true_labels = sample['labels']
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=2)[0].cpu()
            
            # Decode tokens and align predictions
            tokens = self.tokenizer.convert_ids_to_tokens(sample['input_ids'])
            
            print(f"\nSample {idx}:")
            print(f"{'Token':<20} {'True':<15} {'Pred':<15} {'Match'}")
            print("-" * 55)
            
            for i, (tok, true_id, pred_id) in enumerate(zip(tokens, true_labels, predictions)):
                if true_id == -100:
                    continue
                if tok in ['<s>', '</s>', '<pad>']:
                    continue
                    
                true_label = self.label_list[true_id]
                pred_label = self.label_list[pred_id]
                match = "✓" if true_label == pred_label else "✗"
                
                # Highlight mismatches
                if true_label != pred_label:
                    print(f"{tok:<20} {true_label:<15} {pred_label:<15} {match} ← MISMATCH")
                else:
                    print(f"{tok:<20} {true_label:<15} {pred_label:<15} {match}")
        
        print("="*60 + "\n")

# Compute evaluation metrics
def compute_metrics(eval_preds, label_list):
    """Compute seqeval metrics with detailed breakdown."""
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)
    
    true_labels = []
    true_predictions = []
    
    # Also track per-class accuracy for PER / LOC comparison
    correct_by_class = Counter()
    total_by_class = Counter()
    
    for pred_seq, label_seq in zip(predictions, labels):
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
    
    # Print per-class accuracy
    print("\n--- Per-Class Accuracy ---")
    for label in sorted(total_by_class.keys()):
        acc = correct_by_class[label] / total_by_class[label] if total_by_class[label] > 0 else 0
        print(f"  {label:<15}: {acc*100:5.1f}% ({correct_by_class[label]}/{total_by_class[label]})")
    
    # Check if model is just predicting O which can be problematic (depending on data imbalance)
    o_predictions = sum(1 for sent in true_predictions for l in sent if l == 'O')
    total_predictions = sum(len(sent) for sent in true_predictions)
    o_pred_pct = 100 * o_predictions / total_predictions if total_predictions > 0 else 0
    
    if o_pred_pct > 95:
        print(f"\n Model predicting 'O' for {o_pred_pct:.1f}% of tokens!")
    
    return {
        'precision': precision_score(true_labels, true_predictions),
        'recall': recall_score(true_labels, true_predictions),
        'f1': f1_score(true_labels, true_predictions),
    }

# Main function
def main():
    # Parse arguments
    args = parse_args()

    # Print configuration
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Train file: {args.train_file}")
    print(f"  Dev file: {args.dev_file if args.dev_file else 'None'}")
    print(f"  Test file: {args.test_file if args.test_file else 'None'}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Model: {args.model_name}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")

    # Build and print label list
    print("\n1. Building label vocabulary...")
    label_list = build_label_list(args.train_file, args.dev_file, args.test_file)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"   Labels: {label_list}")
    print(f"   Number of labels: {len(label_list)}")

    # Load tokenizer and model
    print(f"\n2. Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    # Load datasets
    print("\n3. Loading datasets...")

    train_tokens, train_labels = read_conll_file(args.train_file)
    print(f"   Train: {len(train_tokens)} sentences")

    # Analyze and print class distribution
    flat_train_labels = [l for sent in train_labels for l in sent]
    label_counts = Counter(flat_train_labels)
    total_tokens = len(flat_train_labels)

    print(f"\n   Class distribution in training data:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_tokens
        print(f"     {label:<15}: {count:>8,} ({pct:5.2f}%)")

    dev_tokens, dev_labels = [], []
    if args.dev_file and os.path.exists(args.dev_file):
        dev_tokens, dev_labels = read_conll_file(args.dev_file)
        print(f"   Dev: {len(dev_tokens)} sentences")

    test_tokens, test_labels = [], []
    if args.test_file and os.path.exists(args.test_file):
        test_tokens, test_labels = read_conll_file(args.test_file)
        print(f"   Test: {len(test_tokens)} sentences")

    # Create datasets
    print("\n4. Tokenizing and creating datasets...")

    train_dataset = NERDataset(train_tokens, train_labels, tokenizer, label2id, args.max_length)
    
    dev_dataset = None
    if dev_tokens:
        dev_dataset = NERDataset(dev_tokens, dev_labels, tokenizer, label2id, args.max_length)

    test_dataset = None
    if test_tokens:
        test_dataset = NERDataset(test_tokens, test_labels, tokenizer, label2id, args.max_length)

    # Compute class weights
    use_class_weights = not args.no_class_weights
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(train_labels, label2id)
        print(f"\n   Using class weights:")
        for label, idx in label2id.items():
            print(f"     {label:<15}: {class_weights[idx]:.1f}")

    # Calculate evaluation steps
    total_steps = (len(train_dataset) // args.batch_size) * args.epochs
    eval_steps = max(1, int(total_steps * EVAL_PERCENT / 100))

    print(f"\n   Total training steps: {total_steps}")
    print(f"   Evaluating every {eval_steps} steps ({EVAL_PERCENT}%)")
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Training arguments
    print("\n5. Setting up training...")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps" if dev_dataset else "no",
        eval_steps=eval_steps if dev_dataset else None,
        save_strategy="steps",
        save_steps=eval_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True if dev_dataset else False,
        metric_for_best_model="f1" if dev_dataset else None,
        greater_is_better=True,
        fp16=not args.no_fp16 and torch.cuda.is_available(),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to="none",
    )
    
    # Callbacks
    callbacks = []
    if dev_dataset and args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    if dev_dataset and not args.no_sample_predictions:
        callbacks.append(PredictionLoggingCallback(
            dev_dataset, tokenizer, label_list, args.num_samples_to_show
        ))
    
    # Create metric function
    def compute_metrics_fn(eval_preds):
        return compute_metrics(eval_preds, label_list)
    
    # Create trainer
    if use_class_weights:
        trainer = WeightedLossTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn if dev_dataset else None,
            callbacks=callbacks,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn if dev_dataset else None,
            callbacks=callbacks,
        )
    
    # Train
    print("\n6. Training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Class weights: {'Enabled' if use_class_weights else 'Disabled'}")
    print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    trainer.train()
    
    # Save model
    print(f"\n7. Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, 'label_map.json'), 'w') as f:
        json.dump({'label2id': label2id, 'id2label': id2label, 'label_list': label_list}, f, indent=2)
    
    # Evaluate on test set
    if test_dataset:
        print("\n8. Evaluating on test set...")
        results = trainer.evaluate(test_dataset)
        print(f"\nTest Results:")
        print(f"   Precision: {results.get('eval_precision', 0):.4f}")
        print(f"   Recall: {results.get('eval_recall', 0):.4f}")
        print(f"   F1: {results.get('eval_f1', 0):.4f}")
        
        # Classification report
        print("\n   Classification Report:")
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=2)
        
        true_labels = []
        true_predictions = []
        
        for pred_seq, label_seq in zip(preds, predictions.label_ids):
            pred_labels = []
            gold_labels = []
            
            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id != -100:
                    pred_labels.append(label_list[pred_id])
                    gold_labels.append(label_list[label_id])
            
            true_labels.append(gold_labels)
            true_predictions.append(pred_labels)
        
        print(classification_report(true_labels, true_predictions))

        with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()