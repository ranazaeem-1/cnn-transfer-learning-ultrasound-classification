###############################################################
# ResNet101 — Problem B (D vs P vs I)
# FINAL, STABLE, COLAB-SAFE VERSION
###############################################################

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


###############################################################
# CONFIG
###############################################################

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

EXCEL_PATH = os.path.join(BASE_DIR, "PatientImages_MATCHED.xlsx")
IMG_DIR    = os.path.join(BASE_DIR, "processed_images")

FILENAME_COL = "filename"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30

UNFREEZE_LAYERS = 80           # ResNet101: safer than 100
LEARNING_RATE = 1e-5

MODEL_OUT = os.path.join(
    BASE_DIR,
    "ResNet101_ProblemB_DPI_unfreeze50_best.keras"
)

tf.random.set_seed(42)
np.random.seed(42)


###############################################################
# LOAD EXCEL
###############################################################

df = pd.read_excel(EXCEL_PATH)

if FILENAME_COL not in df.columns:
    raise ValueError(f"Missing filename column. Found: {df.columns.tolist()}")


###############################################################
# ROBUST DIAGNOSIS COLUMN DETECTION
###############################################################

def find_diagnosis_column(df):
    target = {"D", "P", "I", "N"}
    best_col = None
    best_overlap = 0
    for c in df.columns:
        vals = set(df[c].astype(str).str.strip().dropna().unique())
        overlap = len(vals & target)
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = c
    if best_col is None:
        raise ValueError("Could not find diagnosis column with D/P/I/N")
    return best_col

DIAG_COL = find_diagnosis_column(df)
print("✅ Diagnosis column:", repr(DIAG_COL))


###############################################################
# CLEAN + FILTER (Problem B)
###############################################################

df[DIAG_COL] = df[DIAG_COL].astype(str).str.strip()
df[FILENAME_COL] = df[FILENAME_COL].astype(str).str.strip()

df = df[df[DIAG_COL].isin(["D", "P", "I"])].copy()

CLASS_MAP = {"D": 0, "P": 1, "I": 2}
df["label"] = df[DIAG_COL].map(CLASS_MAP).astype("int32")

print("\nClass counts:")
print(df[DIAG_COL].value_counts())


###############################################################
# CHECK MISSING IMAGES
###############################################################

missing = [
    f for f in df[FILENAME_COL]
    if not os.path.exists(os.path.join(IMG_DIR, f))
]

print(f"\nMissing images: {len(missing)}")
if missing:
    print("First 10 missing:", missing[:10])


###############################################################
# SPLIT DATA
###############################################################

train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
)

train_df.to_csv("train_B.csv", index=False)
val_df.to_csv("val_B.csv", index=False)
test_df.to_csv("test_B.csv", index=False)


###############################################################
# CLASS WEIGHTS
###############################################################

cw = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["label"]),
    y=train_df["label"]
)

class_weight = dict(enumerate(cw))
print("\nClass weights:", class_weight)


###############################################################
# DATASET
###############################################################

def load_image(filename, label):
    filename = filename.decode()
    path = os.path.join(IMG_DIR, filename)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.stack([img]*3, axis=-1).astype("float32")
    img = preprocess_input(img)

    return img, np.int32(label)


def make_dataset(csv, shuffle):
    df = pd.read_csv(csv)
    ds = tf.data.Dataset.from_tensor_slices(
        (df[FILENAME_COL].values, df["label"].values)
    )

    def _map(x, y):
        img, lbl = tf.numpy_function(
            load_image, [x, y], [tf.float32, tf.int32]
        )
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        lbl.set_shape(())
        return img, lbl

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(800)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset("train_B.csv", True)
val_ds   = make_dataset("val_B.csv", False)
test_ds  = make_dataset("test_B.csv", False)


###############################################################
# AUGMENTATION (CONSERVATIVE)
###############################################################

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomTranslation(0.04, 0.04),
])


###############################################################
# MODEL
###############################################################

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = augmentation(inputs)

base = ResNet101(
    weights="imagenet",
    include_top=False,
    input_tensor=x
)

# Freeze all
for layer in base.layers:
    layer.trainable = False

# Unfreeze last N layers (keep BN frozen)
for layer in base.layers[-UNFREEZE_LAYERS:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

x = GlobalAveragePooling2D()(base.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(3, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTrainable base layers:",
      sum(l.trainable for l in base.layers))


###############################################################
# CALLBACKS (SAFE)
###############################################################

callbacks = [
    ModelCheckpoint(
        MODEL_OUT,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=12,
        mode="max",
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=4,
        min_lr=1e-7,
        verbose=1
    )
]


###############################################################
# TRAIN
###############################################################

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks
)


###############################################################
# TEST
###############################################################

print("\nTEST RESULTS")
model.evaluate(test_ds)

print("\n🎉 DONE")
print("Saved model:", MODEL_OUT)
