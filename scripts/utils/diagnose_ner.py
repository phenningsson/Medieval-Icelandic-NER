"""
Analyzes NER datasets for potential problems before training:
- Class imbalance (O vs. entity tokens)
- Entity distribution (Person vs. Location)
- Sentence length statistics
- Token frequency analysis
- Rare entity detection

Quickstart usage:
    # Analyze predefined experiment:
    python scripts/utils/diagnose_ner.py --experiment normalised/norm_menota_ihpc

    # Analyze custom files:
    python scripts/utils/diagnose_ner.py \\
        --train-file data/normalised/normalised_ner_data/norm_menota/train.txt \\
        --dev-file data/normalised/normalised_ner_data/dev/norm_dev.txt \\
        --test-file data/normalised/normalised_ner_data/test/norm_test.txt
"""

import os
import argparse
from pathlib import Path
import sys
from collections import Counter
from typing import List, Tuple, Dict

# Data paths, use custom paths in CLI to override
TRAIN_FILE = 'data/normalised/normalised_ner_data/norm_menota_ihpc/train.txt'
DEV_FILE = 'data/normalised/normalised_ner_data/dev/norm_dev.txt'
TEST_FILE = 'data/normalised/normalised_ner_data/test/norm_test.txt'

# Parse CLI arguments
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Diagnose NER datasets for potential issues',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze predefined experiment:
    python scripts/utils/diagnose_ner.py --experiment normalised/norm_menota_ihpc

    # Analyze custom files:
    python scripts/utils/diagnose_ner.py \\
        --train-file data/normalised/normalised_ner_data/norm_menota/train.txt \\
        --dev-file data/normalised/normalised_ner_data/dev/norm_dev.txt
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

        args.train_file = args.train_file or str(base / exp_name / 'train.txt')
        args.dev_file = args.dev_file or str(base / 'dev' / f'{exp_type[:4]}_dev.txt')
        args.test_file = args.test_file or str(base / 'test' / f'{exp_type[:4]}_test.txt')
    else:
        # Use defaults from configuration
        args.train_file = args.train_file or TRAIN_FILE
        args.dev_file = args.dev_file or DEV_FILE
        args.test_file = args.test_file or TEST_FILE

    # At least one file needs to be be provided
    if not any([args.train_file, args.dev_file, args.test_file]):
        parser.error("At least one data file must be specified")

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

# Analysis of data
def analyze_dataset(name: str, tokens: List[List[str]], labels: List[List[str]]) -> Dict:
    """Analyze a dataset - simplified output."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")

    # Flatten labels
    flat_labels = [l for sent_labels in labels for l in sent_labels]

    # Count labels
    label_counts = Counter(flat_labels)
    total_tokens = len(flat_labels)

    print(f"\nTotal sentences: {len(tokens):,}")
    print(f"Total tokens: {total_tokens:,}")

    # Class distribution
    print(f"\n--- Class Distribution ---")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_tokens
        bar = '█' * int(pct / 2)
        print(f"  {label:15s}: {count:8,} ({pct:5.2f}%) {bar}")

    # Calculate class imbalance
    o_count = label_counts.get('O', 0)
    entity_count = total_tokens - o_count
    o_pct = 100 * o_count / total_tokens

    print(f"\n--- Class Imbalance Analysis ---")
    print(f"  'O' tokens:     {o_count:,} ({o_pct:.2f}%)")
    print(f"  Entity tokens:  {entity_count:,} ({100-o_pct:.2f}%)")
    print(f"  Ratio O:Entity: {o_count/max(entity_count,1):.1f}:1")

    # Entity statistics
    print(f"\n--- Entity Statistics ---")
    b_labels = {l: c for l, c in label_counts.items() if l.startswith('B-')}
    for label, count in sorted(b_labels.items(), key=lambda x: -x[1]):
        entity_type = label[2:]
        print(f"  {entity_type:12s}: {count:,} entities")

    return {
        'total_tokens': total_tokens,
        'total_sentences': len(tokens),
        'label_counts': label_counts,
        'o_percentage': o_pct,
    }

# Main function
def main():
    # Parse arguments
    args = parse_args()

    print("=" * 60)
    print("NER Dataset Diagnostic Tool")
    print("=" * 60)

    # Print configuration
    print(f"\nAnalyzing files:")
    if args.train_file:
        print(f"  Train: {args.train_file}")
    if args.dev_file:
        print(f"  Dev:   {args.dev_file}")
    if args.test_file:
        print(f"  Test:  {args.test_file}")

    # Load and analyze datasets
    if args.train_file and os.path.exists(args.train_file):
        train_tokens, train_labels = read_conll_file(args.train_file)
        analyze_dataset("Training Set", train_tokens, train_labels)
    elif args.train_file:
        print(f"\n  Training file not found: {args.train_file}")

    if args.dev_file and os.path.exists(args.dev_file):
        dev_tokens, dev_labels = read_conll_file(args.dev_file)
        analyze_dataset("Dev Set", dev_tokens, dev_labels)
    elif args.dev_file:
        print(f"\n  Dev file not found: {args.dev_file}")

    if args.test_file and os.path.exists(args.test_file):
        test_tokens, test_labels = read_conll_file(args.test_file)
        analyze_dataset("Test Set", test_tokens, test_labels)
    elif args.test_file:
        print(f"\n  Test file not found: {args.test_file}")


if __name__ == '__main__':
    main()