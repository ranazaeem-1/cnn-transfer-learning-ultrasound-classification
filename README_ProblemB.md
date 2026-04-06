# MSK Problem B - Multi-Model Training

## Quick Start for Google Colab

### Step 1: Upload to Drive
Upload `MSK_Problem_B_Colab.py` to your Google Drive MSK project folder.

### Step 2: Open in Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. File → Open Notebook → Google Drive → Select the file
3. Or copy-paste the code into a new notebook

### Step 3: Mount Drive
Run the first cell to mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 4: Update Path
Change `BASE_DIR` to match your Drive path:
```python
BASE_DIR = "/content/drive/MyDrive/MSK project"  # Your path here
```

### Step 5: Run Experiments

**Option A: Train ALL models** (takes several hours)
```python
all_results = []
for name in ["ResNet50", "ResNet101", "EfficientNetB0", "VGG16", "DenseNet121"]:
    result = train_single_model(name, epochs=50)
    all_results.append(result)
```

**Option B: Train ONE model**
```python
result = train_single_model("EfficientNetB0", epochs=50)
```

**Option C: Quick test** (5 epochs)
```python
result = train_single_model("EfficientNetB0", epochs=5)
```

---

## Files Created

| File | Description |
|------|-------------|
| `MSK_Problem_B_MultiModel.py` | Full modular script |
| `MSK_Problem_B_Colab.py` | Colab-optimized cell-by-cell version |

---

## Output Files (saved to `results/` folder)

- `{ModelName}_best.keras` - Best model checkpoint
- `{ModelName}_log.csv` - Training log
- `{ModelName}_results.png` - Training curves + confusion matrix
- `model_comparison.csv` - Final comparison table

---

## Models Tested

| Model | Description |
|-------|-------------|
| ResNet50 | 50-layer residual network |
| ResNet101 | 101-layer residual network (current baseline) |
| EfficientNetB0 | Efficient architecture, good for small datasets |
| VGG16 | Classic network, used in myositis literature |
| DenseNet121 | Dense connections, excellent feature reuse |

---

## Literature Reference

**Burlina et al. (2017)** - PLOS ONE
- "Automated diagnosis of myositis from muscle ultrasound"
- IBM vs DM/PM accuracy: **74.8%**
- [Link to paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0184059)
