import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from patchify import patchify
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from vit import ViT


# Hyperparameters

hp = {}
hp["image_size"] = 200
hp["channel_num"] = 3
hp["patch_size"] = 25
hp["num_patches"] = (hp["image_size"]**2) // (hp["patch_size"]**2)
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"]*hp["patch_size"]*hp["channel_num"])

hp["batch_size"] = 16
hp["lr"] = 1e-4
hp["num_epochs"] = 50
hp["classes_num"] = 5
hp["class_names"] = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

hp["num_layers"] = 12
hp["hidden_dim"] = 768
hp["mlp_dim"] = 3072
hp["num_heads"] = 12
hp["dropout_rate"] = 0.1

# Functions

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.1):
    images = shuffle(glob(os.path.join(path, "*", "*.jpg")))
    split_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)

    return train_x, valid_x, test_x

def process_image_label(path):
    # reading images
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (hp["image_size"], hp["image_size"]))
    # image = image/255.0


    # preprocessing into patches
    patch_shape = (hp["patch_size"], hp["patch_size"], hp["channel_num"])
    patches = patchify(image, patch_shape, hp["patch_size"])
    
    # formatting input
    patches = np.reshape(patches, (64, 25, 25, 3))
    # for i in range(64):
    #     filename = f"{i}.png"
    #     filepath = os.path.join("files", filename)
    #     cv2.imwrite(filepath, patches[i])
    
    patches = np.reshape(patches, hp["flat_patches_shape"])
    patches = patches.astype(np.float32)

    # labels
    #print(path)
    class_name = path.split("\\")[-2]
    class_index = hp["class_names"].index(class_name)
    class_index = np.array(class_index, dtype=np.int32)

    return patches, class_index


def parse(path):
    patches, labels = tf.numpy_function(process_image_label, [path], [tf.float32, tf.int32])
    # one hot encoding
    labels = tf.one_hot(labels, hp["classes_num"])

    patches.set_shape(hp["flat_patches_shape"])
    labels.set_shape(hp["classes_num"])

    return patches, labels

def tf_dataset(images, batch=32):
    ds  = tf.data.Dataset.from_tensor_slices((images))
    ds = ds.map(parse).batch(batch).prefetch(8)
    return ds


# driver code 

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    # Directory for storing files
    dataset_path = "D:/git_repos/ViT_imageclassification/flower_photos"
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "log.csv")

    # Loading dataset
    train_x, valid_x, test_x = load_data(dataset_path)
    print(f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

    train_ds = tf_dataset(train_x, batch=hp["batch_size"])
    valid_ds = tf_dataset(valid_x, batch=hp["batch_size"])

    # MODEL - Vision Transformer
    model = ViT(hp)
    model.compile(
        loss = "categorical_crossentropy",
        optimizer = tf.keras.optimizers.Adam(hp["lr"], clipvalue=1.0),
        metrics=["acc"]
    )

    callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-10, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=False),
    ]

    model.fit(
        train_ds,
        epochs=hp["num_epochs"],
        validation_data=valid_ds,
        callbacks=callbacks
    )