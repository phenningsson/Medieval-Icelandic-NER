# Using MIM-GOLD-NER with this repository

Some experiments in this repository use **MIM-GOLD-NER** (Modern Icelandic NER dataset) to complement the Old Icelandic training data. Due to license restrictions, we cannot redistribute MIM-GOLD-NER. Users must obtain it separately and integrate it using our provided scripts. 

### Experiments not using MIM-GOLD-NER

**Normalised Old Icelandic:**
- `norm_menota_only` - Normalised Menota
- `norm_menota_resamp`- Normalised Menota (resampled)
- `norm_menota_ihpc` - Normalised Menota + IcePaHC
- `norm_menota_ihpc_resamp` - Normalised Menota + IcePaHC (both resampled)

**Diplomatic Old Icelandic:**
- `dipl_menota` - Diplomatic Menota only
- `dipl_menota_ihpc` - Diplomatic Menota + IcePaHC
- `dipl_menota_resamp` - Diplomatic Menota (resampled)
- `dipl_menota_ihpc_resamp` - Diplomatic Menota + IcePaHC (both resampled)

### Requires MIM-GOLD-NER

**Normalised Old Icelandic:**
- `norm_menota_ihpc_mim` - Menota + IcePaHC + MIM-GOLD-NER
- `norm_menota_ihpc_resamp_mim` - Menota + IcePaHC + MIM (Menota + IcePaHC resampled)

**Diplomatic Old Icelandic:**
- `dipl_menota_mim` - Menota + MIM-GOLD-NER
- `dipl_menota_ihpc_mim` - Menota + IcePaHC + MIM
- `dipl_menota_resamp_mim` - Menota + MIM (Menota resampled)
- `diplo_menota_ihpc_mim_resamp` - Menota + IcePaHC + MIM (Menota + IcePaHC resampled)

---

## How to obtain MIM-GOLD-NER

### Step 1: Visit CLARIN.IS

Go to: https://repository.clarin.is/repository/xmlui/handle/20.500.12537/140?show=full

### Step 2: Download the corpus

1. Download the "by-source.zip" file
2. Unzip
3. Be sure to agree and adhere to the license agreement! 


## How to integrate MIM-GOLD-NER

Once you have MIM-GOLD-NER downloaded, follow these steps:

### Step 1: Filter MIM-GOLD-NER

MIM-GOLD-NER contains multiple entity types (Person, Location, Organization, Date, etc.). We only need **Person** and **Location** to match the Old Icelandic entity annotations.

```bash
python scripts/prepare_mim_data.py \
    --dir external_data/mim_gold_ner/ \
    --output-dir external_data/mim_filtered/
```

### Step 2: Create MIM datasets

Combine the filtered MIM-GOLD-NER with the Old Icelandic datasets:

```bash
python scripts/add_mim_to_experiments.py \
    --mim_train external_data/mim_filtered/mim_gold_ner_only_train_filtered.txt \
    --base_dir ner/data/
```

## Training with MIM-enhanced experiments

Train models as usual:

```bash
# Example: Train with Menota+IcePaHC+MIM (normalised)
python ner/training/train_ner.py \
    --train_file ner/data/normalised/normalised_ner_data/menota_ihpc_mim/train.txt \
    --dev_file ner/data/normalised/normalised_ner_data/dev/dev.txt \
    --test_file ner/data/normalised/normalised_ner_data/test/test.txt \
    --output_dir models/menota_ihpc_mim

# Example: Train with Menota+MIM (diplomatic)
python ner/training/train_ner.py \
    --train_file ner/data/diplomatic/diplomatic_ner_data/menota_mim/train.txt \
    --dev_file ner/data/diplomatic/diplomatic_ner_data/dev/dev.txt \
    --test_file ner/data/diplomatic/diplomatic_ner_data/test/test.txt \
    --output_dir models/diplomatic_menota_mim
```

---

## License Compliance

Be sure to agree and follow the license terms of MIM-GOLD-NER. For full license terms, see the MIM-GOLD-NER page at CLARIN.IS: https://repository.clarin.is/repository/xmlui/handle/20.500.12537/140?show=full.


### Citation

When using MIM-GOLD-NER, please cite:

```bibtex
 @misc{20.500.12537/140,
 title = {{MIM}-{GOLD}-{NER} – named entity recognition corpus (21.09)},
 author = {Ing{\'o}lfsd{\'o}ttir, Svanhv{\'{\i}}t Lilja and Gu{\dh}j{\'o}nsson, {\'A}smundur Alma and Loftsson, Hrafn},
 url = {http://hdl.handle.net/20.500.12537/140},
 note = {{CLARIN}-{IS}},
 copyright = {Icelandic Mim Gold Standard for Named Entity Recognition ({NER})},
 year = {2020} 
 }
```
---

**License Note:** This repository does not include or redistribute MIM-GOLD-NER. Users must obtain it separately and agree to its license terms. Please respect the great work carried out by the authors of MIM-GOLD-NER, and do follow their license terms. 
