# Medieval Icelandic Named Entity Recognition

[![Code License: GPL v3](https://img.shields.io/badge/Code%20License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Data License: CC BY-SA 4.0](https://img.shields.io/badge/Data%20License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](#)

This repository contains code and data for training BERT-based Named Entity Recognition (NER) models on Medieval Icelandic texts. We view NER as a token classification task, where the model identifies either [Person] or [Location] entities in both normalised and diplomatic transcriptions of Old Icelandic texts. The development and test sets consist strictly of data derived from Old Icelandic texts (with the majority written between the years 1250 - 1400; one excpetion is the 18th century version of Íslendingabók) stored in the Medieval Nordic Text Archive (Menota). The training data, however, consists of various variations (experiments): strictly Old Icelandic texts from Menota; resampling of Old Icelandic texts and their annotated entities; additions of annotated normalised Old Icelandic texts from the Icelandic Parsed Historical Corpus (IcePaHC, see below); modern texts written and annotated in contemporary Icelandic (MIM-GOLD-NER, see below). 

To address issues of a general nature relating to NER on historical texts, as well as specific to Old Icelandic in particular, we create and compare 12 different experiment configurations through different data combinations, model transfer learning and finetuning, as well as entity class balancing and weighting. For more information about configurations and augmentations, see our paper *(coming soon)*. For our best performing models, see our HuggingFace space *(coming soon)*.

For our best performing NER model on normalised texts, we achieve a **0.926** F1 score.
For our best performing NER model on diplomatic texts, we achieve a **0.789** F1 score.
In the cases above, both NER models use the maximum amount of available training data; in the case of the diplomatic NER model, it leverages a customised finetuned version of IceBERT that is finetuned for diplomatic Old Icelandic texts derived from Menota (see our HuggingFace space for more information about our finetuned IceBERT model). 

## Repository Structure

```
Medieval-Icelandic-NER/
├── data/                    # Training, dev, and test datasets
│   ├── normalised/          # Normalised transcription experiments (4 base + 2 MIM)
│   └── diplomatic/          # Diplomatic transcription experiments (4 base + 4 MIM)
├── scripts/
│   ├── training/            # Model training (train_ner.py)
│   ├── evaluate/            # Model evaluation (evaluate_ner.py)
│   ├── utils/               # Dataset diagnostics (diagnose_ner.py)
│   └── mim_processing/      # MIM-GOLD-NER processing scripts (we do not distribute MIM-GOLD-NER)
```

## Resources

- **Paper:** [Link to paper] *(coming soon)*
- **Models:** [HuggingFace Hub] *(coming soon)*
- **Data:** [Zenodo] *(coming soon)*

## Get Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Medieval-Icelandic-NER.git
cd Medieval-Icelandic-NER
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Quickstart Use Examples

#### 1. Explore the Data

Analyze a selected dataset:
```bash
python scripts/utils/diagnose_ner.py --experiment normalised/norm_menota_ihpc
```

#### 2. Train a Model

Train on a specific experiment:
```bash
python scripts/training/train_ner.py --experiment normalised/norm_menota_ihpc
```

Customize training parameters (optional):
```bash
python scripts/training/train_ner.py \
    --experiment diplomatic/dipl_menota \
    --epochs 10 \
    --learning-rate 2e-5 \
    --batch-size 16
```

#### 3. Evaluate a Model

Evaluate on test set:
```bash
python scripts/evaluate/evaluate_ner.py --experiment normalised/norm_menota_ihpc
```

### MIM-GOLD-NER Modern Icelandic Training Data 

To augment training data with Modern Icelandic NER data from MIM-GOLD-NER, please see the [MIM_GOLD_NER_INSTRUCTIONS.md] file for instructions. Due to license restrictions, we cannot redistribute MIM-GOLD-NER. Users must obtain it separately and integrate it using our provided scripts in order to reproduce some of our training data and thereby results. 

## Available Experiments (training data and model variations)

### Base Experiments (6)
- **Normalised:** `norm_menota`, `norm_menota_resamp`, `norm_menota_ihpc`, `norm_menota_ihpc_resamp`
- **Diplomatic:** `dipl_menota`, `dipl_menota_ihpc`, `dipl_menota_resamp`, `dipl_menota_ihpc_resamp`

### MIM-GOLD-NER Experiments (6)
- **Normalised:** `norm_menota_ihpc_mim`, `norm_menota_ihpc_resamp_mim`
- **Diplomatic:** `dipl_menota_mim`, `dipl_menota_ihpc_mim`, `dipl_menota_resamp_mim`, `dipl_menota_ihpc_resamp_mim`

## Entity Types

- **Person:** Person names
- **Location:** Place names

We use BIO tagging: `B-Person`, `I-Person`, `B-Location`, `I-Location`, `O`

## Data Format

All data files uses the CoNLL format:
```
Token    Label
Óláfr    B-Person
konungr  O
fór      O
til      O
Noregs   B-Location
.        O
         (blank line separates sentences)
```

## Citation

If you use this code or data, please cite:

```bibtex
@key{commingsoon}
```

## License

For complete licensing details, attributions, and citations, please see the [LICENSE](LICENSE) file. In short, we use the **GNU General Public License v3.0 (GPL-3.0)** for all the source code (Python scripts), and the **Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)** for the derived data.

**Note:** This repository does not include or redistribute the MIM-GOLD-NER dataset. Users must obtain it separately and agree to its licensing terms. Please respect the great work carried out by the authors of MIM-GOLD-NER, and do follow their license terms. We are thankful to the open source community, and hope to contribute to it ourselves with this work. 

## Acknowledgments
We are grateful for the great work carried out by the projects below, and for making it possible for us to use their data in order to conduct our academic research and develop NER models for Medieval Icelandic. 

- **Menota:** [The Menota project](https://www.menota.org/)
- **Icelandic Parsed Historical Corpus (IcePaHC)** [Wallenberg et al., 2024](http://hdl.handle.net/20.500.12537/325)
- **IceBERT base model:** [Snæbjarnarson et al. 2022](https://aclanthology.org/2022.lrec-1.477/)
- **MIM-GOLD-NER:** [Ingólfsdóttir et al. 2019](http://hdl.handle.net/20.500.12537/140)

## Contact

For questions or issues, please open a GitHub issue or contact:
[phenningsson@me.com]
