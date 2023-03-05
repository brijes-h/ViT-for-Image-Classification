import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from patchify import patchify
import tensorflow as tf


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
hp["class_names"] = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

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
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (hp["image_size"], hp["image_size"]))
    image = image/255.0
    print(image.shape)

    # preprocessing into patches
    patch_shape = (hp["patch_size"], hp["patch_size"], hp["channel_num"])
    patches = patchify(image, patch_shape, hp["patch_size"])
    print(patches.shape)



if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    # Directory for storing files
    dataset_path = "flower_photos"
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "log.csv")

    # Loading dataset
    train_x, valid_x, test_x = load_data(dataset_path)
    print(f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

    process_image_label(train_x[0])