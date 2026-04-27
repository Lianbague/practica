import os
from PIL import Image
import numpy as np


def load_tinyimagenet_subset(root, split="train", max_classes=5):
    X = []
    Y = []

    train_dir = os.path.join(root, "train")
    #class_folders = sorted(os.listdir(train_dir))[:max_classes]
    class_folders = [
        d for d in sorted(os.listdir(train_dir))
        if os.path.isdir(os.path.join(train_dir, d))
    ][:max_classes]

    class_to_idx = {cls: i for i, cls in enumerate(class_folders)}

    if split == "train":
        for cls in class_folders:
            img_dir = os.path.join(train_dir, cls, "images")

            for img_name in os.listdir(img_dir):
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(img_dir, img_name)

                img = Image.open(img_path).convert("RGB")
                img = img.resize((64, 64))

                X.append(np.array(img))
                Y.append(class_to_idx[cls])

    elif split == "val":
        val_dir = os.path.join(root, "val")
        img_dir = os.path.join(val_dir, "images")
        annotation_file = os.path.join(val_dir, "val_annotations.txt")

        val_map = {}
        with open(annotation_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                val_map[parts[0]] = parts[1]

        for img_name, cls in val_map.items():
            if cls not in class_to_idx:
                continue

            img_path = os.path.join(img_dir, img_name)

            img = Image.open(img_path).convert("RGB")
            img = img.resize((64, 64))

            X.append(np.array(img))
            Y.append(class_to_idx[cls])

    return np.array(X), np.array(Y)


import h5py
import numpy as np
import tensorflow as tf

# -------------------------------
# LOADERS
# -------------------------------

def load_from_hdf5(path):
    with h5py.File(path, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
    return X, Y


def load_from_npz(path):
    data = np.load(path)
    return data['X'], data['Y']


def parse_tfrecord(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'channels': tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(example, feature_description)

    img = tf.io.decode_raw(example['image'], tf.uint8)
    h = example['height']
    w = example['width']
    c = example['channels']

    img = tf.reshape(img, (h, w, c))
    label = example['label']

    return img, label


def load_from_tfrecord(path):
    raw_dataset = tf.data.TFRecordDataset(path)
    parsed = raw_dataset.map(parse_tfrecord)

    X, Y = [], []

    for img, label in parsed:
        X.append(img.numpy())
        Y.append(label.numpy())

    return np.array(X), np.array(Y)