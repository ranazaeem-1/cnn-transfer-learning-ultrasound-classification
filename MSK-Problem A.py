###############################################################
#  FINAL ResNet50-100 PIPELINE FOR PROBLEM A (RECOMMENDED)
#  - Uses your best-performing normalization: /255.0
#  - Unfreezes last 100 layers (your best setting)
#  - Data augmentation + callbacks
#  - Adds: class_weight (helps imbalance) + test evaluation
#  - Adds: safe handling for missing images
###############################################################

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


###############################################################
# Config
###############################################################
EXCEL_PATH = "PatientImages_MATCHED.xlsx"
IMG_DIR = "processed_images"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

UNFREEZE_LAYERS = 100
LEARNING_RATE = 1e-5

MODEL_OUT = "ResNet50_ProblemA_unfreeze100_best.h5"

tf.random.set_seed(42)
np.random.seed(42)


###############################################################
# STEP 1 — LOAD EXCEL + CLEAN DIAGNOSIS
###############################################################
df = pd.read_excel(EXCEL_PATH)

diag_col = [c for c in df.columns if "diagn" in c.lower()][0]
df["Diagnosis_Clean"] = df[diag_col].astype(str).str.strip()

def label_A(diag):
    return 0 if diag == "N" else 1

df["label_A"] = df["Diagnosis_Clean"].apply(label_A)

if "filename" not in df.columns:
    raise ValueError("Expected a 'filename' column in the Excel file.")

df = df.dropna(subset=["filename"]).copy()


###############################################################
# STEP 2 — TRAIN / VAL / TEST SPLIT (STRATIFIED)
###############################################################
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label_A"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label_A"]
)

train_df.to_csv("train_A.csv", index=False)
val_df.to_csv("val_A.csv", index=False)
test_df.to_csv("test_A.csv", index=False)


###############################################################
# STEP 2.5 — CLASS WEIGHTS (helps imbalance)
###############################################################
classes = np.array([0, 1])
cw = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label_A"].values
)
class_weight = {0: float(cw[0]), 1: float(cw[1])}
print("Class weights:", class_weight)


###############################################################
# STEP 3 — TF DATA LOADER WITH SHAPE FIX
###############################################################
def load_image(filename, label):
    # filename arrives as bytes from tf.numpy_function
    file = filename.decode("utf-8") if isinstance(filename, (bytes, bytearray)) else str(filename)
    path = os.path.join(IMG_DIR, file)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # If missing/corrupt, return black image (avoid crash)
    if img is None:
        img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # grayscale -> 3 channels
    img = np.stack([img, img, img], axis=-1).astype("float32")

    # BEST for your setup (matches your strongest results)
    img = img / 255.0

    return img, np.int32(label)


def df_to_dataset(csv_file, batch_size=32, shuffle=True):
    df_local = pd.read_csv(csv_file)

    filenames = df_local["filename"].astype(str).values
    labels = df_local["label_A"].values.astype("int32")

    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))

    def load_and_fix_shape(x, y):
        img, lbl = tf.numpy_function(load_image, [x, y], [tf.float32, tf.int32])
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        lbl.set_shape(())
        return img, lbl

    ds = ds.map(load_and_fix_shape, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(800, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = df_to_dataset("train_A.csv", batch_size=BATCH_SIZE, shuffle=True)
val_ds   = df_to_dataset("val_A.csv",   batch_size=BATCH_SIZE, shuffle=False)
test_ds  = df_to_dataset("test_A.csv",  batch_size=BATCH_SIZE, shuffle=False)


###############################################################
# STEP 4 — DATA AUGMENTATION
###############################################################
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomZoom(0.1)
])


###############################################################
# STEP 5 — BUILD RESNET50 MODEL (UNFREEZE LAST 100 LAYERS)
###############################################################
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# IMPORTANT: leave training=None so augmentation runs only during training
x = data_augmentation(inputs)

base = ResNet50(
    weights="imagenet",
    include_top=False,
    input_tensor=x
)

# Freeze all first
for layer in base.layers:
    layer.trainable = False

# Unfreeze last N layers
for layer in base.layers[-UNFREEZE_LAYERS:]:
    layer.trainable = True

# Head
x = GlobalAveragePooling2D()(base.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
)

print(f"\nTrainable ResNet layers: {sum(l.trainable for l in base.layers)} / {len(base.layers)}")
print(f"Best model will be saved to: {MODEL_OUT}\n")


###############################################################
# STEP 6 — CALLBACKS
###############################################################
callbacks = [
    ModelCheckpoint(MODEL_OUT, monitor="val_loss", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]


###############################################################
# STEP 7 — TRAIN MODEL
###############################################################
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight
)


###############################################################
# STEP 8 — PRINT FINAL RESULTS
###############################################################
print("\n==================== TRAINING RESULTS ====================")
print("Final Training Accuracy:", history.history["accuracy"][-1])
print("Final Training Loss:", history.history["loss"][-1])
print("Final Validation Accuracy:", history.history["val_accuracy"][-1])
print("Final Validation Loss:", history.history["val_loss"][-1])
print("Final Validation AUC:", history.history["val_auc"][-1])
print("Final Validation Precision:", history.history["val_precision"][-1])
print("Final Validation Recall:", history.history["val_recall"][-1])
print("==========================================================\n")


###############################################################
# STEP 9 — TEST EVALUATION
###############################################################
test_metrics = model.evaluate(test_ds, verbose=0)
print("==================== TEST RESULTS ====================")
for name, val in zip(model.metrics_names, test_metrics):
    print(f"{name}: {val}")
print("======================================================\n")

print(f"🎉 Done. Best checkpoint saved as: {MODEL_OUT}")
