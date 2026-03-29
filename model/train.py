import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# ── Configuration ──────────────────────────────────────────
DATASET_PATH = "dataset"
IMG_SIZE     = 224
BATCH_SIZE   = 32
EPOCHS       = 10
NUM_CLASSES  = 6

# ── Data loading & augmentation ────────────────────────────
# Training images get randomly flipped/zoomed to help the model generalise
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=15
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ── Build the model ────────────────────────────────────────
# Load MobileNetV2 pretrained on ImageNet, without its top layer
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the base model so we only train our new top layers
base_model.trainable = False

# Add our own classification head on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ── Compile ────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ── Train ──────────────────────────────────────────────────
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ── Save the model ─────────────────────────────────────────
model.save("model/classifier.h5")
print("Model saved to model/classifier.h5")

# ── Plot accuracy & loss ───────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"],     label="Train accuracy")
plt.plot(history.history["val_accuracy"], label="Val accuracy")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"],     label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("model/training_results.png")
print("Training chart saved to model/training_results.png")