# myositis-ultrasound-classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

**Deep learning-based classification of inflammatory myopathies from muscle ultrasound images.**

</div>

---

## Overview

This project develops and evaluates multiple CNN-based deep learning models to automate the
diagnosis of inflammatory myopathies (muscle diseases) from **muscle ultrasound images**. It
addresses three clinically motivated classification tasks:

| Problem | Task | Classes |
|---|---|---|
| **A** | Screening | Normal vs. Affected (any disease) |
| **B** | IBM Identification | Normal vs. Inclusion Body Myositis (IBM) |
| **C** | Subtype Separation | IBM vs. Other Myositis (DM / PM) |

Problem B and C follow the evaluation protocol of **Burlina et al. (2017)**, the key reference
benchmark for this task.

---

## Results Summary

| Problem | Best Model | Test Accuracy | Val Accuracy | AUC |
|---|---|---|---|---|
| A — Normal vs. Affected | ResNet50 | **78.5%** | 80.5% | **0.87** |
| B — Normal vs. IBM | ResNet50 | **71.0%** | 72.6% | — |
| C — IBM vs. DM/PM | Xception | **72.5%** | 70.6% | **0.75** |

> **Benchmark:** Burlina et al. (2017) achieved **74.8%** on IBM vs. DM/PM (binary) with a
> larger dataset. Our results are competitive while being trained on a smaller dataset using
> modern pre-trained architectures.

---

## Dataset

The dataset consists of **3,214 muscle ultrasound images** from patients with confirmed
diagnoses, linked via an Excel spreadsheet.

| Class | Label | Count |
|---|---|---|
| Normal | N | 1,313 |
| Dermatomyositis | D | 555 |
| Polymyositis | P | 552 |
| Inclusion Body Myositis | I | 794 |

### Expected folder structure

```
myositis-ultrasound-classification/
├── processed_images/          # PNG ultrasound images (named 0.png, 1.png, ...)
├── PatientImages_MATCHED.xlsx # Labels + metadata spreadsheet
├── MSK_Problem_A_Local.ipynb  # Problem A notebook
├── MSK_Problem_B_Local.ipynb  # Problem B notebook
├── MSK_Problem_C_Local.ipynb  # Problem C notebook
├── results_A/                 # Output: metrics, plots, model (auto-created)
├── results_B/                 # Output: metrics, plots, model (auto-created)
├── results_C/                 # Output: metrics, plots, model (auto-created)
└── README.md
```

> **Note:** The `processed_images/` folder and `PatientImages_MATCHED.xlsx` are **not
> included** in the repository due to patient data privacy. You must supply your own dataset.

---

## Models Evaluated

Seven architectures were benchmarked for Problems B and C:

| Model | Params | Notes |
|---|---|---|
| ResNet50 | 25.6 M | Best for Problems A & B |
| ResNet101 | 44.7 M | Deeper ResNet variant |
| EfficientNetB0 | 5.3 M | Lightweight efficient model |
| EfficientNetB4 | 19.3 M | Larger EfficientNet variant |
| DenseNet121 | 8.1 M | Dense connections |
| InceptionV3 | 23.9 M | Multi-scale convolutions |
| Xception | 22.9 M | **Best for Problem C** |

All models use **ImageNet pre-trained weights** with selective layer unfreezing (transfer learning).

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/myositis-ultrasound-classification.git
cd myositis-ultrasound-classification
```

### 2. Create a virtual environment

```bash
python -m venv msk_env
# Windows
msk_env\Scripts\activate
# macOS / Linux
source msk_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your data

Place your files in the project root:
- `processed_images/` — folder containing PNG images
- `PatientImages_MATCHED.xlsx` — Excel file with `filename` and `Diagnosis` columns

---

## Usage

Open any notebook in Jupyter or VS Code and run all cells:

```bash
jupyter notebook MSK_Problem_A_Local.ipynb
```

### Configuration

At the top of each notebook, a **Configuration cell** allows you to adjust paths and hyperparameters:

```python
BASE_DIR    = r'.'                        # Root folder (where your data lives)
EXCEL_PATH  = os.path.join(BASE_DIR, 'PatientImages_MATCHED.xlsx')
IMAGE_DIR   = os.path.join(BASE_DIR, 'processed_images')
RESULTS_DIR = os.path.join(BASE_DIR, 'results_A')   # Output directory

LEARNING_RATE   = 1e-5
BATCH_SIZE      = 32
EPOCHS          = 50
UNFREEZE_LAYERS = 100    # Number of layers to fine-tune from the top
```

### Outputs

Each notebook automatically saves to its results folder:

| File | Description |
|---|---|
| `*_best.keras` | Best model checkpoint (by val AUC) |
| `*_log.csv` | Per-epoch training metrics |
| `*_results.png` | Training curves + confusion matrix |
| `comparison.csv` | All model results in one table |
| `comparison_table.png` | Visual comparison table |

---

## Requirements

```
tensorflow>=2.10
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
opencv-python
```

Install with:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn openpyxl opencv-python
```

---

## Project Structure

```
├── MSK_Problem_A_Local.ipynb   Problem A: ResNet50, Normal vs. Affected
├── MSK_Problem_B_Local.ipynb   Problem B: 7 models, Normal vs. IBM
├── MSK_Problem_C_Local.ipynb   Problem C: 7 models, IBM vs. DM/PM
├── create_local_notebooks.py   Script that generates the notebooks from templates
├── MSK_Project_Documentation.md  Full academic-style project report
├── MSK_Presentation_Outline.md   Slide-by-slide presentation outline
├── .gitignore
└── README.md
```

---

## Key Techniques

- **Transfer Learning** — ImageNet pre-trained weights with selective unfreezing
- **Class Weighting** — Balanced weights via `sklearn.utils.class_weight.compute_class_weight`
- **Conservative Augmentation** — Rotation ±5%, zoom ±5%, translation ±4% (preserves ultrasound texture)
- **Early Stopping** — Patience of 10 epochs monitored on validation AUC
- **ReduceLROnPlateau** — Halves learning rate if val loss stagnates for 5 epochs
- **ModelCheckpoint** — Always saves the best model based on validation AUC

---

## Reference

> Burlina, P., Billings, W., Joshi, N., & Albayda, I. (2017).
> **Automated diagnosis of myositis from muscle ultrasound: Exploring the use of machine
> learning and deep learning methods.**
> *PLOS ONE*, 12(8), e0184059.
> https://doi.org/10.1371/journal.pone.0184059

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

> ⚠️ **Medical Disclaimer:** This software is intended for research purposes only. It is not
> approved for clinical use and should not be used as a substitute for professional medical
> diagnosis.

---

## Author

**[Your Name]**
[University / Institute]
[Contact email or GitHub profile link]
