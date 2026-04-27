import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import random

# -------------------------------
# GENERIC DATASET CLASS
# -------------------------------
class ImageDataset:
    def __init__(self, X, Y, img_shape):
        self.X = X
        self.Y = Y
        self.img_shape = img_shape

    def normalize(self):
        self.X = self.X.astype(np.float32) / 255.0


# -------------------------------
# DATASET CONVERTER
# -------------------------------
class DatasetConverter:

    @staticmethod
    def to_hdf5(dataset, path):
        with h5py.File(path, 'w') as f:
            f.create_dataset('X', data=dataset.X)
            f.create_dataset('Y', data=dataset.Y)
        print(f"Saved HDF5 to {path}")

    @staticmethod
    def to_npz(dataset, path):
        np.savez(path, X=dataset.X, Y=dataset.Y)
        print(f"Saved NPZ to {path}")

    @staticmethod
    def to_tfrecord(dataset, path):
        with tf.io.TFRecordWriter(path) as writer:
            for i in range(len(dataset.X)):
                img = dataset.X[i]

                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=[img.tobytes()])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[int(dataset.Y[i])])),
                    'height': tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[img.shape[0]])),
                    'width': tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[img.shape[1]])),
                    'channels': tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[img.shape[2]])),
                }

                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

        print(f"Saved TFRecord to {path}")


# -------------------------------
# DATASET VALIDATOR
# -------------------------------
class DatasetValidator:

    @staticmethod
    def compare_datasets(ds1, ds2, num_samples=10, visualize=False):
        assert len(ds1.X) == len(ds2.X), "❌ Different dataset sizes!"
        print("✔ Same number of samples")

        label_mismatches = np.sum(ds1.Y != ds2.Y)
        if label_mismatches == 0:
            print("✔ Labels are IDENTICAL")
        else:
            print(f"❌ Label mismatches: {label_mismatches}")

        indices = random.sample(range(len(ds1.X)), num_samples)
        differences = []

        for idx in indices:
            img1 = ds1.X[idx]
            img2 = ds2.X[idx]

            mse = np.mean((img1 - img2) ** 2)
            differences.append(mse)

            print(f"Sample {idx} → MSE: {mse:.6f}")

            if visualize:
                DatasetValidator.visualize_comparison(
                    img1, img2, idx, ds1.img_shape)

        print(f"\nAverage MSE: {np.mean(differences):.6f}")

        if np.mean(differences) < 1e-6 and label_mismatches == 0:
            print("✅ Datasets are PERFECTLY identical")
        else:
            print("⚠️ Differences detected")

    @staticmethod
    def visualize_comparison(img1, img2, idx, shape):
        img1 = img1.reshape(shape)
        img2 = img2.reshape(shape)
        diff = np.abs(img1 - img2)

        plt.figure(figsize=(10, 3))

        plt.subplot(1, 3, 1)
        plt.title(f"Original {idx}")
        plt.imshow(img1)

        plt.subplot(1, 3, 2)
        plt.title(f"Converted {idx}")
        plt.imshow(img2)

        plt.subplot(1, 3, 3)
        plt.title("Difference")
        plt.imshow(diff)

        plt.show()