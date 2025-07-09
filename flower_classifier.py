import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image_dataset_from_directory

print("✅ TensorFlow version:", tf.__version__)
print("🚀 GPU:", tf.config.list_physical_devices('GPU'))

# ========================== STEP 1: LOAD LOCAL DATASET =============================
DATA_DIR = "data"  # updated: local folder instead of Kaggle path

IMG_SIZE = 224
BATCH_SIZE = 32

train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("✅ Classes:", class_names)
NUM_CLASSES = len(class_names)

# ======================== STEP 2: PREPROCESSING ==============================
AUTOTUNE = tf.data.AUTOTUNE

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1)
])

def format_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = (train_ds
            .map(format_img, num_parallel_calls=AUTOTUNE)
            .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
            .prefetch(AUTOTUNE))

val_ds = (val_ds.map(format_img).prefetch(AUTOTUNE))

# ======================= STEP 3: BUILD THE MODEL =============================
def build_model():
    base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.Dense(256, activation='swish')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)]
    )
    return model

model = build_model()
model.summary()

# ====================== STEP 4: TRAIN THE MODEL ==============================
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\n🚀 Phase 1: Transfer Learning")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop],
    verbose=1
)

# ======================= STEP 5: FINE-TUNING FULL MODEL ======================
print("\n🔧 Phase 2: Fine-tuning full model")
model.layers[1].trainable = True
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)]
)

fine_tune = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=history.epoch[-1],
    epochs=25,
    callbacks=[early_stop],
    verbose=1
)

# ======================= STEP 6: EVALUATION ==================================
print("\n🧪 Evaluating on validation set...")
results = model.evaluate(val_ds)
test_loss, test_acc, top3_acc = results

print(f"✅ Final Val Accuracy: {test_acc:.2%}")
print(f"🏆 Final Top-3 Accuracy: {top3_acc:.2%}")

# =================== STEP 7: TRAINING CURVES =================================
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + fine_tune.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'] + fine_tune.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + fine_tune.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'] + fine_tune.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()
plt.show()

model.save("flower_model.h5")
print("✅ Model saved successfully!")
