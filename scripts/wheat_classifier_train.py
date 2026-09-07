

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf # type: ignore
from tensorflow.keras import layers, models, optimizers, callbacks # type: ignore
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.losses import CategoricalCrossentropy # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pathlib import Path


tf.keras.mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(True)


IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32
EPOCHS_FROZEN   = 18
EPOCHS_FINETUNE = 25
LEARNING_RATE   = 1e-3
FINETUNE_LR     = 2e-6
WARMUP_EPOCHS   = 3
NUM_CLASSES     = 3
CLASS_NAMES     = ['good', 'average', 'bad']

DATASET_DIR       = "dataset"
RAW_DATA_DIR      = "raw_data"
MODEL_SAVE_PATH   = "wheat_classifier_mobilenetv2.keras"
PHASE1_CKPT       = "best_model_phase1.keras"
PHASE2_CKPT       = "best_model_phase2.keras"

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}



def _list_images(base_dir):
    image_paths = []
    labels = []
    for i, cls in enumerate(CLASS_NAMES):
        folder = Path(base_dir) / cls
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if f.suffix.lower() in IMAGE_EXTS:
                image_paths.append(str(f))
                labels.append(i)
    return image_paths, labels


def _decode_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    return img, tf.one_hot(label, NUM_CLASSES)


def build_datasets():
    train_paths, train_labels = _list_images(os.path.join(DATASET_DIR, "train"))
    val_paths, val_labels     = _list_images(os.path.join(DATASET_DIR, "val"))
    test_paths, test_labels   = _list_images(os.path.join(DATASET_DIR, "test"))

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(2000, reshuffle_each_iteration=True)
    train_ds = train_ds.map(_decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(_decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_ds = test_ds.map(_decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def compute_class_weights_from_raw():
    raw_counts = {}
    for cls in CLASS_NAMES:
        src = Path(RAW_DATA_DIR) / cls
        if not src.exists():
            break
        raw_counts[cls] = len([f for f in src.iterdir() if f.suffix.lower() in IMAGE_EXTS])

    if raw_counts and sum(raw_counts.values()) > 0:
        total = sum(raw_counts.values())
        weight_dict = {}
        print("\nClass weights from RAW counts:")
        for i, cls in enumerate(CLASS_NAMES):
            w = total / (NUM_CLASSES * raw_counts[cls])
            weight_dict[i] = w
            print(f"  {cls:8s}: {raw_counts[cls]:4d} images → weight {w:.3f}")
        return weight_dict

    print("\nraw_data/ not found — using uniform weights.")
    return {i: 1.0 for i in range(NUM_CLASSES)}



def build_model():
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    model = models.Model(inputs, outputs, name='WheatClassifier_MobileNetV2')
    return model, base



def warmup_cosine_schedule(epoch, _):
    if epoch < WARMUP_EPOCHS:
        return 1e-5 + (LEARNING_RATE - 1e-5) * (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS_FROZEN - WARMUP_EPOCHS, 1)
    return 1e-6 + 0.5 * (LEARNING_RATE - 1e-6) * (1 + math.cos(math.pi * progress))


def finetune_cosine_schedule(epoch, _):
    eta_min = 1e-8
    eta_max = FINETUNE_LR
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * epoch / EPOCHS_FINETUNE))



def get_phase1_callbacks():
    return [
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        callbacks.LearningRateScheduler(warmup_cosine_schedule, verbose=0),
        callbacks.ModelCheckpoint(PHASE1_CKPT, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]


def get_phase2_callbacks():
    return [
        callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        callbacks.LearningRateScheduler(finetune_cosine_schedule, verbose=0),
        callbacks.ModelCheckpoint(PHASE2_CKPT, monitor='val_loss', save_best_only=True, verbose=1)
    ]



def train(model, base_model, train_ds, val_ds, class_weight_dict):
    history_all = {}

    print("\nPHASE 1: Training head only")
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    history1 = model.fit(
        train_ds,
        epochs=EPOCHS_FROZEN,
        validation_data=val_ds,
        class_weight=class_weight_dict,
        callbacks=get_phase1_callbacks(),
        verbose=1
    )
    history_all['phase1'] = history1.history

    print(f"\nReloading best Phase 1 weights from {PHASE1_CKPT} ...")
    model.load_weights(PHASE1_CKPT)

    print("\nPHASE 2: Fine-tune top 20 layers")
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(FINETUNE_LR),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    history2 = model.fit(
        train_ds,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_ds,
        class_weight=class_weight_dict,
        callbacks=get_phase2_callbacks(),
        verbose=1
    )
    history_all['phase2'] = history2.history

    return history_all



def evaluate_model(model, test_ds):
    print("\nEVALUATION ON TEST SET")
    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    y_true = np.concatenate([y.numpy().argmax(axis=1) for _, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix — Wheat Grain Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    print("Confusion matrix saved → confusion_matrix.png")



def plot_history(history_all):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'phase1': '#2E86AB', 'phase2': '#E84855'}
    labels = {'phase1': 'Phase 1 (Head)', 'phase2': 'Phase 2 (Fine-tune)'}

    for phase, history in history_all.items():
        ep = range(1, len(history['accuracy']) + 1)
        c  = colors[phase]
        axes[0].plot(ep, history['accuracy'], color=c, label=f'{labels[phase]} Train')
        axes[0].plot(ep, history['val_accuracy'], color=c, linestyle='--', label=f'{labels[phase]} Val')
        axes[1].plot(ep, history['loss'], color=c, label=f'{labels[phase]} Train')
        axes[1].plot(ep, history['val_loss'], color=c, linestyle='--', label=f'{labels[phase]} Val')

    for ax, title, ylabel in zip(axes, ['Model Accuracy', 'Model Loss'], ['Accuracy', 'Loss']):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Wheat Grain Classifier — Training History', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.close()
    print("Training history saved → training_history.png")


if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices('GPU'))

    train_ds, val_ds, test_ds = build_datasets()
    class_weight_dict = compute_class_weights_from_raw()

    model, base_model = build_model()
    history_all = train(model, base_model, train_ds, val_ds, class_weight_dict)

    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved → {MODEL_SAVE_PATH}")

    evaluate_model(model, test_ds)
    plot_history(history_all)