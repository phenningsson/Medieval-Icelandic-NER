"""
Add MIM-GOLD-NER to the experiments (datasets) that require it.

This script combines filtered MIM-GOLD-NER with Old Icelandic data to create
all the experimental datasets.

LICENSE NOTE: This script does not redistribute MIM-GOLD-NER. Users must obtain
MIM-GOLD-NER separately and then run prepare_mim_data.py.

Quickstart usage:
    # After running prepare_mim_data.py, create MIM experiments:
    python scripts/add_mim_to_experiments.py \
        --mim_train external_data/mim_filtered/mim_gold_ner_only_train_filtered.txt \
        --base_dir data/

    # Or specify custom paths:
    python scripts/add_mim_to_experiments.py \
        --mim_train /path/to/mim_filtered_train.txt \
        --base_dir data/ \
        --output_dir data/

"""

import argparse
from pathlib import Path
from typing import Dict, List
import shutil

def read_conll_file(file_path: Path) -> str:
    """Read a CoNLL file and return its content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def write_conll_file(file_path: Path, content: str):
    """Write content to a CoNLL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def count_entities(content: str) -> Dict:
    """Count entities in CoNLL format content."""
    total_entities = 0
    person_count = 0
    location_count = 0

    for line in content.split('\n'):
        if line.strip() and not line.startswith('#'):
            parts = line.split('\t')
            if len(parts) >= 2:
                label = parts[1]
                if label.startswith('B-'):
                    total_entities += 1
                    entity_type = label[2:]
                    if entity_type == 'Person':
                        person_count += 1
                    elif entity_type == 'Location':
                        location_count += 1

    return {
        'total_entities': total_entities,
        'person_count': person_count,
        'location_count': location_count
    }


def combine_datasets(base_file: Path, mim_content: str, output_file: Path):
    """
    Combine base dataset with MIM-GOLD-NER.

    Args:
        base_file: Path to base Old Icelandic train dataset
        mim_content: Content of MIM-GOLD-NER file
        output_file: Path to output combined file
    """
    # Read base file
    base_content = read_conll_file(base_file)

    # Combine: MIM first, then Old Icelandic train data
    combined = mim_content.strip() + '\n\n' + base_content.strip() + '\n'

    # Write combined file
    write_conll_file(output_file, combined)

    # Count entities for all datasets
    base_stats = count_entities(base_content)
    mim_stats = count_entities(mim_content)
    combined_stats = count_entities(combined)

    return {
        'base_total': base_stats['total_entities'],
        'base_person': base_stats['person_count'],
        'base_location': base_stats['location_count'],
        'mim_total': mim_stats['total_entities'],
        'mim_person': mim_stats['person_count'],
        'mim_location': mim_stats['location_count'],
        'combined_total': combined_stats['total_entities'],
        'combined_person': combined_stats['person_count'],
        'combined_location': combined_stats['location_count'],
        'base_file': base_file.name,
        'output_file': output_file.name
    }


def create_mim_experiments(mim_train: Path, base_dir: Path, output_dir: Path = None) -> List[Dict]:
    """
    Create all experiments that include MIM-GOLD-NER.

    Args:
        mim_train: Path to filtered MIM-GOLD-NER train.txt
        base_dir: Base directory containing the data
        output_dir: Output directory (defaults to base_dir)

    Returns:
        List of dictionaries with creation statistics
    """
    if output_dir is None:
        output_dir = base_dir

    print("="*70)
    print("Creating MIM Experiments")
    print("="*70)
    print(f"\nMIM train file: {mim_train}")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}\n")

    # Read MIM-GOLD-NER content
    mim_content = read_conll_file(mim_train)
    mim_stats = count_entities(mim_content)
    print(f"MIM-GOLD-NER entities: {mim_stats['total_entities']:,} total ({mim_stats['person_count']:,} Person, {mim_stats['location_count']:,} Location)\n")

    # Define experiments that need the MIM-GOLD-NER data as part of the training data
    experiments = [
        # Normalised experiments with MIM
        {
            'name': 'Normalised: Menota+IHPC+MIM',
            'base': base_dir / 'normalised/normalised_ner_data/norm_menota_ihpc/train.txt',
            'output': output_dir / 'normalised/normalised_ner_data/norm_menota_ihpc_mim/train.txt',
        },
        {
            'name': 'Normalised: Menota+IHPC+MIM (resampled)',
            'base': base_dir / 'normalised/normalised_ner_data/norm_menota_ihpc_resamp/train.txt',
            'output': output_dir / 'normalised/normalised_ner_data/norm_menota_ihpc_resamp_mim/train.txt',
        },

        # Diplomatic experiments with MIM
        {
            'name': 'Diplomatic: Menota+MIM',
            'base': base_dir / 'diplomatic/diplomatic_ner_data/dipl_menota/train.txt',
            'output': output_dir / 'diplomatic/diplomatic_ner_data/dipl_menota_mim/train.txt',
        },
        {
            'name': 'Diplomatic: Menota+IHPC+MIM',
            'base': base_dir / 'diplomatic/diplomatic_ner_data/dipl_menota_ihpc/train.txt',
            'output': output_dir / 'diplomatic/diplomatic_ner_data/dipl_menota_ihpc_mim/train.txt',
        },
        {
            'name': 'Diplomatic: Menota+MIM (resampled)',
            'base': base_dir / 'diplomatic/diplomatic_ner_data/dipl_menota_resamp/train.txt',
            'output': output_dir / 'diplomatic/diplomatic_ner_data/dipl_menota_resamp_mim/train.txt',
        },
        {
            'name': 'Diplomatic: Menota+IHPC+MIM (resampled)',
            'base': base_dir / 'diplomatic/diplomatic_ner_data/dipl_menota_ihpc_resamp/train.txt',
            'output': output_dir / 'diplomatic/diplomatic_ner_data/dipl_menota_ihpc_resamp_mim/train.txt',
        },
    ]

    results = []
    success_count = 0
    skip_count = 0

    print("Creating experiments:\n")

    for exp in experiments:
        print(f"{'='*70}")
        print(f"Experiment: {exp['name']}")
        print(f"{'='*70}")

        # Check if base file exists
        if not exp['base'].exists():
            print(f"  WARNING: Base file not found: {exp['base']}")
            print(f"   Skipping this experiment.\n")
            skip_count += 1
            continue

        # Check if output already exists
        if exp['output'].exists():
            print(f"  Output file already exists: {exp['output']}")
            response = input("   Overwrite? (y/n): ").strip().lower()
            if response != 'y':
                print("   Skipping.\n")
                skip_count += 1
                continue

        # Combine datasets
        stats = combine_datasets(exp['base'], mim_content, exp['output'])

        print(f"Created: {exp['output']}")
        print(f"  Base entities:     {stats['base_total']:,} total ({stats['base_person']:,} Person, {stats['base_location']:,} Location)")
        print(f"  MIM entities:      {stats['mim_total']:,} total ({stats['mim_person']:,} Person, {stats['mim_location']:,} Location)")
        print(f"  Combined entities: {stats['combined_total']:,} total ({stats['combined_person']:,} Person, {stats['combined_location']:,} Location)\n")

        results.append({**exp, **stats})
        success_count += 1

    # Print summary of datasets created, including failed ones
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Successfully created: {success_count} experiments")
    print(f"Skipped: {skip_count} experiments")
    print(f"Output directory: {output_dir}\n")

    if success_count > 0:
        print("MIM-enhanced experiments are ready to use!")
        print("\nYou can now train models with these datasets:")
        print("  python scripts/training/train_ner.py --experiment <experiment_name>\n")

    if skip_count > 0:
        print(f"  {skip_count} experiments were skipped.")
        print("   Check that all base datasets exist before running this script.\n")

    return results

# Main function
def main():
    parser = argparse.ArgumentParser(
        description='Add MIM-GOLD-NER to Medieval Icelandic NER experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard usage (after running prepare_mim_data.py):
    python scripts/mim_processing/add_mim_to_experiments.py \\
        --mim_train external_data/mim_filtered/mim_gold_ner_only_train_filtered.txt \\
        --base_dir data/

    # With custom output directory:
    python scripts/mim_processing/add_mim_to_experiments.py \\
        --mim_train /path/to/mim_filtered_train.txt \\
        --base_dir data/ \\
        --output_dir /custom/output/

Prerequisites:
    1. Obtain MIM-GOLD-NER and adhere to their licensing agreement
    2. Run prepare_mim_data.py to filter it
    3. Run this script to create the MIM experiments

See MIM_GOLD_NER_INSTRUCTIONS.md for detailed instructions.
        """
    )

    parser.add_argument(
        '--mim_train',
        type=Path,
        required=True,
        help='Path to filtered MIM-GOLD-NER train file'
    )
    parser.add_argument(
        '--base_dir',
        type=Path,
        default=Path('data'),
        help='Base directory containing existing datasets (default: data/)'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        help='Output directory (default: same as base_dir)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing files without asking'
    )

    args = parser.parse_args()

    # Validate MIM train file exists
    if not args.mim_train.exists():
        print(f"\nERROR: MIM-GOLD-NER file not found: {args.mim_train}\n")
        print("Please run prepare_mim_data.py first:")
        print("  python scripts/prepare_mim_data.py \\")
        print("      --dir /path/to/mim_gold_ner/ \\")
        print("      --output-dir external_data/mim_filtered/\n")
        print("See MIM_GOLD_NER_INSTRUCTIONS.md for detailed instructions.\n")
        return 1

    # Validate base directory exists
    if not args.base_dir.exists():
        print(f"\nERROR: Base directory not found: {args.base_dir}\n")
        print("Please ensure you're running this from the repository root")
        print("and that the data/ directory exists.\n")
        return 1

    # Create MIM experiments
    try:
        results = create_mim_experiments(
            args.mim_train,
            args.base_dir,
            args.output_dir
        )
        return 0
    except Exception as e:
        print(f"\nERROR: {e}\n")
        return 1


if __name__ == '__main__':
    exit(main())
