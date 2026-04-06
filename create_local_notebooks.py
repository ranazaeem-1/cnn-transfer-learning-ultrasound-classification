import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def save_notebook(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Saved: {path}")

# ============================================================
# SHARED CODE BLOCKS
# ============================================================

IMPORTS = """
import os, json, cv2, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
warnings.filterwarnings('ignore')
print("TF version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))
""".strip()

CONFIG_A = """
# ── Paths ──────────────────────────────────────────────────
# Set BASE_DIR to the folder that contains:
#   PatientImages_MATCHED.xlsx
#   processed_images/
# (This is your local MSK project folder)
BASE_DIR    = r'.'                                 # <-- CHANGE IF NEEDED
EXCEL_PATH  = os.path.join(BASE_DIR, 'PatientImages_MATCHED.xlsx')
IMAGE_DIR   = os.path.join(BASE_DIR, 'processed_images')
RESULTS_DIR = os.path.join(BASE_DIR, 'results_A')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyper-params ────────────────────────────────────────────
IMG_SIZE       = 224
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-5
EPOCHS         = 50
UNFREEZE_LAYERS= 100
SEED           = 42
MODEL_OUT      = os.path.join(RESULTS_DIR, 'ResNet50_ProblemA_best.keras')
""".strip()

LOAD_DATA_A = """
df = pd.read_excel(EXCEL_PATH)
# Detect columns (handles multiline or unusual column names)
diag_col  = [c for c in df.columns if "diagn" in c.lower()][0]
image_col = [c for c in df.columns if c.strip().lower() == 'filename'] or \
            [c for c in df.columns if "image" in c.lower() or "file" in c.lower()]
image_col = image_col[0]
print("Diagnosis column :", diag_col)
print("Image column     :", image_col)
print("Unique diagnoses :", sorted(df[diag_col].dropna().unique()))

# Binary mapping: Normal=0, Affected=1
df['label'] = df[diag_col].apply(lambda x: 0 if str(x).strip().upper() == 'N' else 1)
df['path']  = df[image_col].apply(lambda f: os.path.join(IMAGE_DIR, str(f)))
df = df[df['path'].apply(os.path.exists)].reset_index(drop=True)
print(f"Valid images: {len(df)}")
print(df['label'].value_counts().rename({0:'Normal', 1:'Affected'}))
""".strip()

SPLIT_A = """
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=SEED)
val_df,   test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=SEED)
print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

# Save splits
train_df[['path','label']].to_csv(os.path.join(RESULTS_DIR,'train_A.csv'), index=False)
val_df[['path','label']].to_csv(os.path.join(RESULTS_DIR,'val_A.csv'),   index=False)
test_df[['path','label']].to_csv(os.path.join(RESULTS_DIR,'test_A.csv'), index=False)
""".strip()

LOAD_IMAGE_FN = """
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label
""".strip()

DATASET_A = """
augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomContrast(0.2),
    layers.RandomZoom(0.1),
], name="augmentation")

def make_dataset(dataframe, augment=False, shuffle=False):
    paths  = dataframe['path'].values
    labels = dataframe['label'].values.astype('float32')
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1024)
    if augment:
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_df, augment=True,  shuffle=True)
val_ds   = make_dataset(val_df,  augment=False, shuffle=False)
test_ds  = make_dataset(test_df, augment=False, shuffle=False)
print("Datasets ready.")
""".strip()

CLASS_WEIGHT_A = """
cw = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
class_weight = dict(enumerate(cw))
print("Class weights:", class_weight)
""".strip()

BUILD_A = """
base = keras.applications.ResNet50(weights='imagenet', include_top=False,
                                    input_shape=(IMG_SIZE, IMG_SIZE, 3))
# Freeze all, then unfreeze last UNFREEZE_LAYERS
for layer in base.layers:
    layer.trainable = False
for layer in base.layers[-UNFREEZE_LAYERS:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs, name='ResNet50_A')
model.compile(
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy',
             keras.metrics.AUC(name='auc'),
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)
model.summary()
""".strip()

TRAIN_A = """
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
    keras.callbacks.ModelCheckpoint(MODEL_OUT, monitor='val_auc', save_best_only=True, mode='max'),
    keras.callbacks.CSVLogger(os.path.join(RESULTS_DIR, 'training_log_A.csv')),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks
)
print("Training complete.")
""".strip()

PLOT_A = """
hist = history.history
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, key, title in zip(axes.flatten(),
                          ['accuracy','loss','auc','precision'],
                          ['Accuracy','Loss','AUC','Precision']):
    ax.plot(hist[key],      label='Train')
    ax.plot(hist[f'val_{key}'], label='Validation')
    ax.set_title(title); ax.legend(); ax.set_xlabel('Epoch')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_curves_A.png'), dpi=150)
plt.show()
print("Training curves saved.")
""".strip()

EVAL_A = """
test_metrics = model.evaluate(test_ds, verbose=1)
metric_names = ['loss','accuracy','auc','precision','recall']
for n, v in zip(metric_names, test_metrics):
    print(f"  Test {n}: {v:.4f}")

# Confusion matrix + ROC
y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_prob = model.predict(test_ds).ravel()
y_pred = (y_prob >= 0.5).astype(int)

from sklearn.metrics import roc_curve, auc as sk_auc
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = sk_auc(fpr, tpr)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', ax=ax1, cmap='Blues',
            xticklabels=['Normal','Affected'], yticklabels=['Normal','Affected'])
ax1.set_title('Confusion Matrix'); ax1.set_ylabel('True'); ax1.set_xlabel('Predicted')

ax2.plot(fpr, tpr, color='orange', label=f'ROC (AUC={roc_auc:.4f})')
ax2.plot([0,1],[0,1],'--', color='navy')
ax2.set_title('ROC Curve'); ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR'); ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix_roc_A.png'), dpi=150)
plt.show()

print(classification_report(y_true, y_pred, target_names=['Normal','Affected']))
""".strip()

SUMMARY_A = """
summary = {
    'Model': 'ResNet50',
    'Problem': 'A  -  Normal vs Affected',
    'Test_Accuracy':  test_metrics[1],
    'Test_AUC':       test_metrics[2],
    'Test_Precision': test_metrics[3],
    'Test_Recall':    test_metrics[4],
    'Best_Val_Accuracy': max(history.history['val_accuracy']),
    'Best_Val_AUC':      max(history.history['val_auc']),
    'Epochs_Trained': len(history.history['accuracy']),
    'Learning_Rate':  LEARNING_RATE,
    'Unfreeze_Layers': UNFREEZE_LAYERS,
}
pd.DataFrame([summary]).to_csv(os.path.join(RESULTS_DIR, 'problem_A_summary.csv'), index=False)
print("Summary saved to", RESULTS_DIR)
""".strip()

# ============================================================
# PROBLEM B  –  Normal vs IBM
# ============================================================

CONFIG_B = """
# ── Paths ──────────────────────────────────────────────────
BASE_DIR    = r'.'                                 # <-- CHANGE IF NEEDED
EXCEL_PATH  = os.path.join(BASE_DIR, 'PatientImages_MATCHED.xlsx')
IMAGE_DIR   = os.path.join(BASE_DIR, 'processed_images')
RESULTS_DIR = os.path.join(BASE_DIR, 'results_B')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyper-params ────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 32
LEARNING_RATE = 1e-5
EPOCHS        = 50
UNFREEZE_LAYERS = 80
SEED          = 42

MODELS_TO_RUN = ["ResNet50", "ResNet101", "EfficientNetB0",
                 "EfficientNetB4", "DenseNet121", "InceptionV3", "Xception"]
""".strip()

LOAD_DATA_B = """
df = pd.read_excel(EXCEL_PATH)
diag_col  = [c for c in df.columns if "diagn" in c.lower()][0]
image_col = [c for c in df.columns if c.strip().lower() == 'filename'] or \\
            [c for c in df.columns if "image" in c.lower() or "file" in c.lower()]
image_col = image_col[0]

# Problem B: Normal (N) vs IBM (I)
df_b = df[df[diag_col].isin(['N', 'I'])].copy()
df_b['label'] = df_b[diag_col].map({'N': 0, 'I': 1})
df_b['path']  = df_b[image_col].apply(lambda f: os.path.join(IMAGE_DIR, str(f)))
df_b = df_b[df_b['path'].apply(os.path.exists)].reset_index(drop=True)
print(f"Valid images: {len(df_b)}")
print(df_b['label'].value_counts().rename({0:'Normal', 1:'IBM'}))
""".strip()

SPLIT_B = """
train_df, temp_df = train_test_split(df_b, test_size=0.3, stratify=df_b['label'], random_state=SEED)
val_df,   test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=SEED)
print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
CLASS_NAMES = ['Normal', 'IBM']
""".strip()

# ── Problem C: IBM vs (DM + PM)
CONFIG_C = """
# ── Paths ──────────────────────────────────────────────────
BASE_DIR    = r'.'                                 # <-- CHANGE IF NEEDED
EXCEL_PATH  = os.path.join(BASE_DIR, 'PatientImages_MATCHED.xlsx')
IMAGE_DIR   = os.path.join(BASE_DIR, 'processed_images')
RESULTS_DIR = os.path.join(BASE_DIR, 'results_C')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyper-params ────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 32
LEARNING_RATE = 1e-5
EPOCHS        = 50
UNFREEZE_LAYERS = 80
SEED          = 42

MODELS_TO_RUN = ["ResNet50", "ResNet101", "EfficientNetB0",
                 "EfficientNetB4", "DenseNet121", "InceptionV3", "Xception"]
""".strip()

LOAD_DATA_C = """
df = pd.read_excel(EXCEL_PATH)
diag_col  = [c for c in df.columns if "diagn" in c.lower()][0]
image_col = [c for c in df.columns if c.strip().lower() == 'filename'] or \\
            [c for c in df.columns if "image" in c.lower() or "file" in c.lower()]
image_col = image_col[0]

# Problem C: IBM (I) vs other myositis types (DM=D, PM=P)
df_c = df[df[diag_col].isin(['D', 'P', 'I'])].copy()
df_c['label'] = df_c[diag_col].map({'D': 0, 'P': 0, 'I': 1})  # DM/PM=0, IBM=1
df_c['path']  = df_c[image_col].apply(lambda f: os.path.join(IMAGE_DIR, str(f)))
df_c = df_c[df_c['path'].apply(os.path.exists)].reset_index(drop=True)
print(f"Valid images: {len(df_c)}")
print(df_c['label'].value_counts().rename({0:'DM/PM (Other)', 1:'IBM'}))
""".strip()

SPLIT_C = """
train_df, temp_df = train_test_split(df_c, test_size=0.3, stratify=df_c['label'], random_state=SEED)
val_df,   test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=SEED)
print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
CLASS_NAMES = ['DM/PM', 'IBM']
""".strip()

# ── Shared dataset builder (binary, for B and C)
DATASET_BC = """
augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
    layers.RandomTranslation(0.04, 0.04),
], name="augmentation")

def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def make_dataset(dataframe, augment=False, shuffle=False):
    paths  = dataframe['path'].values
    labels = dataframe['label'].values.astype('float32')
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle: ds = ds.shuffle(1024)
    if augment:
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

cw = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
class_weight = dict(enumerate(cw))
print("Class weights:", class_weight)

train_ds = make_dataset(train_df, augment=True,  shuffle=True)
val_ds   = make_dataset(val_df,  augment=False, shuffle=False)
test_ds  = make_dataset(test_df, augment=False, shuffle=False)
print("Datasets ready.")
""".strip()

BUILD_MODEL_FN_BC = """
def get_base(name):
    kw = dict(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    return {
        'ResNet50':       keras.applications.ResNet50(**kw),
        'ResNet101':      keras.applications.ResNet101(**kw),
        'EfficientNetB0': keras.applications.EfficientNetB0(**kw),
        'EfficientNetB4': keras.applications.EfficientNetB4(**kw),
        'DenseNet121':    keras.applications.DenseNet121(**kw),
        'InceptionV3':    keras.applications.InceptionV3(**kw),
        'Xception':       keras.applications.Xception(**kw),
    }[name]

def build_binary_model(model_name):
    base = get_base(model_name)
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[-UNFREEZE_LAYERS:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(512, activation='relu')(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model   = keras.Model(inputs, outputs, name=model_name)
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    return model
""".strip()

TRAIN_ALL_BC = """
all_results = []

for model_name in MODELS_TO_RUN:
    print(f"\\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    keras.backend.clear_session()
    model = build_binary_model(model_name)

    ckpt_path = os.path.join(RESULTS_DIR, f'{model_name}_best.keras')
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_auc', patience=10,
                                       restore_best_weights=True, mode='max'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5,
                                           factor=0.5, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_auc',
                                         save_best_only=True, mode='max'),
        keras.callbacks.CSVLogger(os.path.join(RESULTS_DIR, f'{model_name}_log.csv')),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS,
        class_weight=class_weight, callbacks=callbacks, verbose=1
    )

    # ── Evaluate
    test_metrics = model.evaluate(test_ds, verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_prob  = model.predict(test_ds, verbose=0).ravel()
    y_pred  = (y_prob >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred, average='macro')

    # ── Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, title in zip(axes, ['accuracy','loss','auc'],
                                      ['Accuracy','Loss','AUC']):
        ax.plot(history.history[key],      label='Train')
        ax.plot(history.history[f'val_{key}'], label='Val')
        ax.set_title(f'{model_name} - {title}'); ax.legend()
    cm = confusion_matrix(y_true, y_pred)
    axes[2].set_visible(False)
    fig.add_axes([0.68, 0.1, 0.28, 0.8])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'{model_name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_results.png'), dpi=120)
    plt.show()

    result = {
        'model': model_name,
        'test_acc': round(test_metrics[1], 4),
        'test_auc': round(test_metrics[2], 4),
        'f1_macro': round(f1, 4),
        'best_val_acc': round(max(history.history['val_accuracy']), 4),
        'best_val_auc': round(max(history.history['val_auc']), 4),
        'epochs': len(history.history['accuracy']),
    }
    all_results.append(result)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    print(json.dumps(result, indent=2))

print("\\nAll models done!")
""".strip()

COMPARISON_BC = """
df_res = pd.DataFrame(all_results).sort_values('test_acc', ascending=False)
df_res.to_csv(os.path.join(RESULTS_DIR, 'comparison.csv'), index=False)
print(df_res.to_string(index=False))

# Plot comparison table
fig, ax = plt.subplots(figsize=(12, 3))
ax.axis('off')
tbl = ax.table(cellText=df_res.values, colLabels=df_res.columns,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
tbl.auto_set_column_width(col=list(range(len(df_res.columns))))
for j in range(len(df_res.columns)):
    tbl[0, j].set_facecolor('#2c7bb6')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
    tbl[1, j].set_facecolor('#d4efdf')
plt.title('Model Comparison', fontweight='bold', pad=12)
plt.savefig(os.path.join(RESULTS_DIR, 'comparison_table.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Comparison table saved.")
""".strip()


# ============================================================
# BUILD PROBLEM A NOTEBOOK
# ============================================================
nb_a = new_notebook()
nb_a.cells = [
    new_markdown_cell("# Problem A — Binary Classification: Normal vs Affected\n"
                      "**Local execution version** (no Google Drive / Colab required)\n\n"
                      "**Set `BASE_DIR` in the Configuration cell to your MSK project folder.**"),
    new_code_cell(IMPORTS),
    new_markdown_cell("## Configuration\nEdit `BASE_DIR` to point to your local data folder."),
    new_code_cell(CONFIG_A),
    new_markdown_cell("## Load & Label Data"),
    new_code_cell(LOAD_DATA_A),
    new_markdown_cell("## Train / Val / Test Split"),
    new_code_cell(SPLIT_A),
    new_markdown_cell("## Build tf.data Datasets"),
    new_code_cell(LOAD_IMAGE_FN),
    new_code_cell(DATASET_A),
    new_markdown_cell("## Class Weights"),
    new_code_cell(CLASS_WEIGHT_A),
    new_markdown_cell("## Build Model (ResNet50)"),
    new_code_cell(BUILD_A),
    new_markdown_cell("## Train"),
    new_code_cell(TRAIN_A),
    new_markdown_cell("## Training Curves"),
    new_code_cell(PLOT_A),
    new_markdown_cell("## Evaluate on Test Set"),
    new_code_cell(EVAL_A),
    new_markdown_cell("## Save Summary"),
    new_code_cell(SUMMARY_A),
]
save_notebook(nb_a, 'MSK_Problem_A_Local.ipynb')


# ============================================================
# BUILD PROBLEM B NOTEBOOK
# ============================================================
nb_b = new_notebook()
nb_b.cells = [
    new_markdown_cell("# Problem B — Binary Classification: Normal vs IBM\n"
                      "**Local execution version** — based on Burlina et al. (2017) group definitions.\n\n"
                      "Set `BASE_DIR` in Configuration to your local data folder."),
    new_code_cell(IMPORTS),
    new_markdown_cell("## Configuration"),
    new_code_cell(CONFIG_B),
    new_markdown_cell("## Load & Label Data\n**Classes:** N=0 (Normal), I=1 (IBM)"),
    new_code_cell(LOAD_DATA_B),
    new_markdown_cell("## Train / Val / Test Split"),
    new_code_cell(SPLIT_B),
    new_markdown_cell("## Build Datasets, Augmentation & Class Weights"),
    new_code_cell(DATASET_BC),
    new_markdown_cell("## Model Definitions"),
    new_code_cell(BUILD_MODEL_FN_BC),
    new_markdown_cell("## Train All Models & Evaluate"),
    new_code_cell(TRAIN_ALL_BC),
    new_markdown_cell("## Model Comparison Table"),
    new_code_cell(COMPARISON_BC),
]
save_notebook(nb_b, 'MSK_Problem_B_Local.ipynb')


# ============================================================
# BUILD PROBLEM C NOTEBOOK
# ============================================================
nb_c = new_notebook()
nb_c.cells = [
    new_markdown_cell("# Problem C — Binary Classification: IBM vs Other Myositis (DM / PM)\n"
                      "**Local execution version** — based on Burlina et al. (2017) group definitions.\n\n"
                      "Set `BASE_DIR` in Configuration to your local data folder."),
    new_code_cell(IMPORTS),
    new_markdown_cell("## Configuration"),
    new_code_cell(CONFIG_C),
    new_markdown_cell("## Load & Label Data\n"
                      "**Classes:** DM (D) and PM (P) → label 0 (Other Myositis), IBM (I) → label 1\n\n"
                      "> This follows the same grouping as Burlina et al. 2017."),
    new_code_cell(LOAD_DATA_C),
    new_markdown_cell("## Train / Val / Test Split"),
    new_code_cell(SPLIT_C),
    new_markdown_cell("## Build Datasets, Augmentation & Class Weights"),
    new_code_cell(DATASET_BC),
    new_markdown_cell("## Model Definitions"),
    new_code_cell(BUILD_MODEL_FN_BC),
    new_markdown_cell("## Train All Models & Evaluate"),
    new_code_cell(TRAIN_ALL_BC),
    new_markdown_cell("## Model Comparison Table"),
    new_code_cell(COMPARISON_BC),
]
save_notebook(nb_c, 'MSK_Problem_C_Local.ipynb')

print("\nAll 3 notebooks created successfully!")
