import shutil
import subprocess
from pathlib import Path

try:
    import tensorflow as tf
except ImportError as err:
    subprocess.check_call(['pip', 'install', 'tensorflow'])
    subprocess.check_call(['pip', 'install', 'Pillow'])
    import tensorflow as tf

from PIL import Image, ImageOps, UnidentifiedImageError

from CaptureConfig import MODEL_IMAGE_SIZE


SOURCE_DIR = Path("images")
TRAIN_DIR = SOURCE_DIR / "train"
TEST_DIR = SOURCE_DIR / "test"
MODEL_PATH = "tensorflow_nn.h5"

# This order must match Dinosaur.CLASSES = ["JUMP", "DUCK", "RIGHT"].
CLASSES = ["up", "down", "right"]

TRAIN_RATIO = 0.8
BATCH_SIZE = 32
EPOCHS = 25
SEED = 42


def class_dir(class_name):
    return SOURCE_DIR / class_name


def prepare_output_dirs():
    for split_dir in (TRAIN_DIR, TEST_DIR):
        if split_dir.exists():
            shutil.rmtree(split_dir)

        for class_name in CLASSES:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)


def sanitize_image(source_path, destination_path):
    with Image.open(source_path) as image:
        image = ImageOps.grayscale(image)
        image = image.resize((MODEL_IMAGE_SIZE[1], MODEL_IMAGE_SIZE[0]), Image.Resampling.LANCZOS)
        image.save(destination_path)


def build_dataset():
    prepare_output_dirs()
    dataset_counts = {class_name: {"train": 0, "test": 0, "skipped": 0} for class_name in CLASSES}

    for class_name in CLASSES:
        files = sorted(class_dir(class_name).glob("*.png"))
        shuffled_indices = tf.random.shuffle(tf.range(len(files)), seed=SEED).numpy()
        files = [files[index] for index in shuffled_indices]
        train_count = int(len(files) * TRAIN_RATIO)

        for index, source_path in enumerate(files):
            split = "train" if index < train_count else "test"
            destination_dir = TRAIN_DIR if split == "train" else TEST_DIR
            destination_path = destination_dir / class_name / source_path.name

            try:
                sanitize_image(source_path, destination_path)
                dataset_counts[class_name][split] += 1
            except (OSError, UnidentifiedImageError):
                dataset_counts[class_name]["skipped"] += 1

    return dataset_counts


def make_generator(directory, shuffle):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    return datagen.flow_from_directory(
        directory,
        target_size=MODEL_IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="grayscale",
        classes=CLASSES,
        shuffle=shuffle,
        seed=SEED,
    )


def compute_class_weights(train_generator):
    counts = {}
    for class_name, class_index in train_generator.class_indices.items():
        counts[class_index] = int((train_generator.classes == class_index).sum())

    total = sum(counts.values())
    return {
        class_index: total / (len(CLASSES) * count)
        for class_index, count in counts.items()
        if count > 0
    }


def build_model():
    input_shape = MODEL_IMAGE_SIZE + (1,)
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])


def main():
    dataset_counts = build_dataset()

    train_generator = make_generator(TRAIN_DIR, shuffle=True)
    validation_generator = make_generator(TEST_DIR, shuffle=False)

    print("Dataset preparado:", dataset_counts)
    print("Orden de clases:", train_generator.class_indices)

    model = build_model()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        class_weight=compute_class_weights(train_generator),
        callbacks=callbacks,
    )

    loss, accuracy = model.evaluate(validation_generator, verbose=0)
    print(f"Accuracy de validacion: {accuracy:.4f}")
    model.save(MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
