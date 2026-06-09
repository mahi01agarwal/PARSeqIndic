# Indic Scene Text Recognition — IIIT-IndicSTR-Word

Implementation of PARSeq-based Scene Text Recognition (STR) for 10 Indic languages, as described in:

> **Indic Scene Text on the Roadside**  
> Ajoy Mondal, Krishna Tulsyan, C V Jawahar  
> CVIT, International Institute of Information Technology, Hyderabad  
> [Dataset & Results](https://cvit.iiit.ac.in/usodi/istr.php)

Built on top of [PARSeq](https://github.com/baudm/parseq) (Bautista & Atienza, ECCV 2022), adapted for Indic scripts with language-specific character sets, separate per-language models, and the IIIT-IndicSTR-Word dataset.

---

## Supported Languages

| Language  | Script     | Model Checkpoint                     |
|-----------|------------|--------------------------------------|
| Bengali   | Bengali    | `models/bengali/parseq_model.ckpt`   |
| Gujarati  | Gujarati   | `models/gujarati/parseq_model.ckpt`  |
| Hindi     | Devanagari | `models/hindi/parseq_model.ckpt`     |
| Kannada   | Kannada    | `models/kannada/parseq_model.ckpt`   |
| Malayalam | Malayalam  | `models/malayalam/parseq_model.ckpt` |
| Marathi   | Devanagari | `models/marathi/parseq_model.ckpt`   |
| Oriya     | Odia       | `models/oriya/parseq_model.ckpt`     |
| Punjabi   | Gurmukhi   | `models/punjabi/parseq_model.ckpt`   |
| Tamil     | Tamil      | `models/tamil/parseq_model.ckpt`     |
| Telugu    | Telugu     | `models/telugu/parseq_model.ckpt`    |

---

## Project Structure

```
STR_Code_and_Model/
│
├── CharacterSet/                        # Per-language character set definitions
│   ├── bengali.yaml
│   ├── gujarati.yaml
│   ├── hindi.yaml
│   ├── kannada.yaml
│   ├── malayalam.yaml
│   ├── marathi.yaml
│   ├── oriya.yaml
│   ├── punjabi.yaml
│   ├── tamil.yaml
│   └── telugu.yaml
│
├── code/parseq_STR/                     # Main codebase
│   │
│   ├── config/                          # Hydra base config
│   │   ├── config.yaml                  # Main config (model, data, trainer params)
│   │   ├── hydra.yaml
│   │   └── overrides.yaml
│   │
│   ├── configs/                         # Modular Hydra override configs
│   │   ├── bench.yaml
│   │   ├── main.yaml
│   │   ├── tune.yaml
│   │   ├── charset/                     # Per-language charset overrides
│   │   │   └── bengali.yaml             # ⚠️ Only Bengali added — add others for remaining languages
│   │   ├── dataset/
│   │   │   ├── real.yaml                # Use real training data
│   │   │   └── synth.yaml               # Use synthetic training data
│   │   ├── experiment/                  # Model architecture experiment configs
│   │   │   ├── parseq.yaml              # ← used for this project
│   │   │   ├── parseq-tiny.yaml
│   │   │   ├── parseq-patch16-224.yaml
│   │   │   ├── abinet.yaml
│   │   │   ├── abinet-sv.yaml
│   │   │   ├── crnn.yaml
│   │   │   ├── trba.yaml
│   │   │   ├── trbc.yaml
│   │   │   ├── tune_abinet-lm.yaml
│   │   │   └── vitstr.yaml
│   │   └── model/                       # Model hyperparameter configs
│   │       ├── parseq.yaml
│   │       ├── abinet.yaml
│   │       ├── crnn.yaml
│   │       ├── trba.yaml
│   │       └── vitstr.yaml
│   │
│   ├── data/                            # Dataset root (LMDB format)
│   │   ├── test/
│   │   │   └── bengali/                 # ⚠️ Only Bengali present — add others as needed
│   │   ├── train/
│   │   │   ├── real/
│   │   │   │   └── bengali/
│   │   │   └── synth/
│   │   │       └── bengali/
│   │   └── val/
│   │       └── bengali/
│   │
│   ├── strhub/                          # Core library
│   │   ├── data/
│   │   │   ├── aa_overrides.py
│   │   │   ├── augment.py
│   │   │   ├── dataset.py               # LMDB dataset loader
│   │   │   ├── module.py                # SceneTextDataModule (PyTorch Lightning)
│   │   │   └── utils.py
│   │   └── models/
│   │       ├── base.py
│   │       ├── modules.py
│   │       ├── utils.py
│   │       ├── parseq/                  # ← model used in this project
│   │       │   ├── system.py
│   │       │   └── modules.py
│   │       ├── abinet/
│   │       ├── crnn/
│   │       ├── trba/
│   │       └── vitstr/
│   │
│   ├── requirements/
│   │   ├── core.txt                     # Core dependencies
│   │   ├── train.txt                    # Training dependencies
│   │   ├── test.txt                     # Test/eval dependencies
│   │   ├── bench.txt                    # Benchmarking dependencies
│   │   ├── tune.txt                     # Hyperparameter tuning dependencies
│   │   └── constraints.txt
│   │
│   ├── tools/                           # Dataset conversion utilities
│   │   ├── create_lmdb_dataset.py       # ← Convert raw images to LMDB format
│   │   ├── filter_lmdb.py
│   │   ├── art_converter.py
│   │   ├── case_sensitive_str_datasets_converter.py
│   │   ├── coco_2_converter.py
│   │   ├── coco_text_converter.py
│   │   ├── lsvt_converter.py
│   │   ├── mlt19_converter.py
│   │   ├── openvino_converter.py
│   │   ├── test_abinet_lm_acc.py
│   │   └── textocr_converter.py
│   │
│   ├── test.py                          # Evaluation on LMDB test sets
│   ├── test_combined.py                 # Evaluation across multiple datasets
│   ├── test_html.py                     # HTML visual output of predictions
│   ├── train.py                         # Training script
│   ├── read.py                          # Single-image inference
│   ├── bench.py                         # Speed/FLOPs benchmarking
│   ├── tune.py                          # Hyperparameter tuning (Ray Tune)
│   ├── api.py
│   ├── html_image.py
│   ├── hubconf.py
│   ├── num_samples.py
│   └── Makefile
│
└── models/                              # Trained checkpoints (one per language)
    ├── bengali/parseq_model.ckpt
    ├── gujarati/parseq_model.ckpt
    ├── hindi/parseq_model.ckpt
    ├── kannada/parseq_model.ckpt
    ├── malayalam/parseq_model.ckpt
    ├── marathi/parseq_model.ckpt
    ├── oriya/parseq_model.ckpt
    ├── punjabi/parseq_model.ckpt
    ├── tamil/parseq_model.ckpt
    └── telugu/parseq_model.ckpt
```

> **⚠️ Note:** `configs/charset/` currently only has `bengali.yaml` and `data/` only has Bengali splits. Before running other languages, add their charset configs and data (see sections below).

---

## Installation

Requires Python >= 3.9 and PyTorch >= 1.10.

```bash
cd code/parseq_STR

# Core + evaluation dependencies
pip install -r requirements/core.txt
pip install -r requirements/test.txt

# Additionally for training
pip install -r requirements/train.txt
```

---

## Dataset

The **IIIT-IndicSTR-Word** dataset contains 250K word-level images across 10 Indic languages, extracted from roadside scene images captured with a GoPro camera.

| Split          | Images per Language | Total       |
|----------------|--------------------:|------------:|
| Train          | 17,500              | 175,000     |
| Validation     | 2,500               | 25,000      |
| Test           | 5,000               | 50,000      |
| **Total**      | **25,000**          | **250,000** |

Download from: https://cvit.iiit.ac.in/usodi/istr.php

### Data Format

All data must be in **LMDB format**. After downloading, organize as:

```
data/
├── test/<language>/
├── train/
│   ├── real/<language>/
│   └── synth/<language>/
└── val/<language>/
```

### Converting Raw Images to LMDB

If you have raw images and a labels file, convert using:

```bash
python tools/create_lmdb_dataset.py \
  --inputPath /path/to/images/ \
  --gtFile /path/to/labels.txt \
  --outputPath data/test/<language>
```

The labels file format (one entry per line):
```
image_filename.jpg transcription
```

### Adding a New Language

1. Add charset config to `configs/charset/<language>.yaml` — copy from `CharacterSet/<language>.yaml` at the repo root and wrap it in the Hydra format:
```yaml
# @package _global_
model:
  charset_train: "<paste characters here>"
  charset_test: "<paste characters here>"
```

2. Place LMDB data at `data/test/<language>/`, `data/train/real/<language>/`, etc.

---

## Inference on a Single Image

Use `read.py` to run inference on any word-level crop:

```bash
cd code/parseq_STR

python read.py ../../models/bengali/parseq_model.ckpt \
  --images /path/to/word_image.jpg
```

For multiple images:

```bash
python read.py ../../models/telugu/parseq_model.ckpt \
  --images /path/to/images/*.jpg
```

---

## Evaluation (Reproducing Paper Results)

`test.py` evaluates a checkpoint on the LMDB test set and reports **WRR (Word Recognition Rate)** and **CRR (Character Recognition Rate)**.

### Evaluate Bengali (works out of the box)

```bash
cd code/parseq_STR

python test.py ../../models/bengali/parseq_model.ckpt \
  --data_root data \
  --batch_size 512 \
  --device cpu        # use --device cuda if GPU available
```

Results are printed to terminal and also saved to `../../models/bengali/parseq_model.ckpt.log.txt`.

### Expected Output

```
Benchmark (Subset) set:
| Dataset | # samples | Accuracy | 1 - NED | Confidence | Label Length |      WER |
|:--------|----------:|---------:|--------:|-----------:|-------------:|---------:|
| bengali |      5000 |    85.34 |   92.75 |      88.xx |         4.xx |    14.66 |
```

`Accuracy` = WRR and `1 - NED` ≈ CRR as reported in the paper.

### Evaluate a Different Language

`test.py` currently has Bengali hardcoded in two places. Edit these lines before running another language:

```python
# line ~88
test_set = sorted(set(['telugu']))      # ← change language here

# line ~103
result_groups = {
    'Benchmark (Subset)': ['telugu']    # ← and here
}
```

Then run:

```bash
python test.py ../../models/telugu/parseq_model.ckpt \
  --data_root data \
  --batch_size 512 \
  --device cpu
```

### Evaluate All 10 Languages (Reproduce Table 4)

```bash
cd code/parseq_STR

for lang in bengali gujarati hindi kannada malayalam marathi oriya punjabi tamil telugu; do
  echo "=== Testing: $lang ==="
  sed -i "s/sorted(set(\[.*\]))/sorted(set(['$lang']))/" test.py
  sed -i "s/'Benchmark (Subset)': \[.*\]/'Benchmark (Subset)': ['$lang']/" test.py
  python test.py ../../models/$lang/parseq_model.ckpt \
    --data_root data \
    --batch_size 256 \
    --device cpu
done
```

### Benchmark Results (Table 4 from Paper)

| Language  | CRR (%) | WRR (%) |
|-----------|--------:|--------:|
| Bengali   | 92.75   | 85.34   |
| Gujarati  | 88.12   | 81.91   |
| Hindi     | 95.01   | 87.24   |
| Kannada   | 87.64   | 79.27   |
| Malayalam | 89.42   | 80.31   |
| Marathi   | 94.47   | 85.50   |
| Oriya     | 95.13   | 86.53   |
| Punjabi   | 91.46   | 84.27   |
| Tamil     | 95.63   | 86.35   |
| Telugu    | 92.18   | 84.94   |

---

## Training

Models are trained in two stages: pre-training on synthetic data from IndicSTR12, then fine-tuning on real data from IIIT-IndicSTR-Word.

### Stage 1 — Pre-train on Synthetic Data

```bash
cd code/parseq_STR

python train.py +experiment=parseq +charset=bengali +dataset=synth \
  trainer.accelerator=gpu trainer.devices=1
```

### Stage 2 — Fine-tune on Real Data

```bash
python train.py +experiment=parseq +charset=bengali +dataset=real \
  ckpt_path=outputs/parseq/<timestamp>/checkpoints/last.ckpt \
  trainer.accelerator=gpu trainer.devices=1
```

Change `+charset=bengali` to any supported language. Make sure the corresponding `configs/charset/<language>.yaml` and `data/train/` folders exist first.

### Key Training Configuration (`config/config.yaml`)

| Parameter        | Value    |
|------------------|----------|
| Image size       | 32 × 128 |
| Max label length | 50       |
| Batch size       | 254      |
| Learning rate    | 0.0007   |
| Encoder depth    | 12       |
| Decoder depth    | 1        |
| Patch size       | 4 × 8    |
| Embed dim        | 384      |
| Permutations (K) | 6        |
| Max epochs       | 10       |
| Gradient clip    | 20       |
| LR scheduler     | 1cycle   |
| Optimizer        | Adam     |

---

## Model Architecture

PARSeq uses a Vision Transformer (ViT) encoder and a visio-lingual decoder trained with Permutation Language Modeling (PLM):

- **Encoder:** 12-block ViT, processes 32×128 images tokenized into 4×8 patches → 128 tokens
- **Decoder:** Single transformer block with two Multi-Head Attention modules — one for context-position attention (with permutation masks), one for image-position attention
- **Training:** K=6 permutations with mirroring; 1cycle LR scheduler + Adam optimizer
- **Inference:** Left-to-right autoregressive decoding with 1 refinement iteration

---

## Character Sets

Each language has its own character set defined in `CharacterSet/<language>.yaml` (root level) and mirrored into `configs/charset/<language>.yaml` for use during training.

Character sets include:
- Native script characters (vowels, consonants, matras, conjuncts, halant)
- Digits in both native script and ASCII (0–9)
- Common punctuation and special characters
- Zero-width characters (ZWJ `\u200d`, ZWNJ `\u200c`) for correct conjunct rendering

The charset is baked into each checkpoint at training time — inference automatically uses the correct one from the checkpoint's `hparams`, so you don't need to specify it manually when running `test.py` or `read.py`.

---

## Citation

If you use this code or dataset, please cite both papers:

```bibtex
@inproceedings{mondal2024indicstr,
  title={Indic Scene Text on the Roadside},
  author={Mondal, Ajoy and Tulsyan, Krishna and Jawahar, C V},
  booktitle={...},
  year={2024}
}

@InProceedings{bautista2022parseq,
  title={Scene Text Recognition with Permuted Autoregressive Sequence Models},
  author={Bautista, Darwin and Atienza, Rowel},
  booktitle={European Conference on Computer Vision},
  pages={178--196},
  year={2022},
  publisher={Springer Nature Switzerland},
  doi={10.1007/978-3-031-19815-1_11}
}
```

---

## Acknowledgements

This work is supported by MeitY, Government of India, through the NLTM-Bhashini project.  
Codebase adapted from [baudm/parseq](https://github.com/baudm/parseq).