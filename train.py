import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4

BASE_DIR = "dataset"

# =========================
# Image Generators
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)


def create_generators(category):
    train_path = os.path.join(BASE_DIR, category, "train")
    val_path = os.path.join(BASE_DIR, category, "val")

    train_gen = train_datagen.flow_from_directory(
        train_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_gen = val_datagen.flow_from_directory(
        val_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    return train_gen, val_gen


# =========================
# Model Builder
# =========================
def build_model(num_classes):
    base_model = InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False  # Transfer learning (safe)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================
# Train LEAF model
# =========================
print("\nTraining LEAF disease model...\n")
leaf_train, leaf_val = create_generators("leaf")
leaf_model = build_model(leaf_train.num_classes)

leaf_model.fit(
    leaf_train,
    validation_data=leaf_val,
    epochs=EPOCHS
)

leaf_model.save("tomato_leaf_disease_model.keras")
print("Leaf model saved as tomato_leaf_disease_model.keras")


# =========================
# Train FRUIT model
# =========================
print("\nTraining FRUIT disease model...\n")
fruit_train, fruit_val = create_generators("fruit")
fruit_model = build_model(fruit_train.num_classes)

fruit_model.fit(
    fruit_train,
    validation_data=fruit_val,
    epochs=EPOCHS
)

fruit_model.save("tomato_fruit_disease_model.keras")
print("Fruit model saved as tomato_fruit_disease_model.keras")
