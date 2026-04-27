import numpy as np
import h5py
import tensorflow as tf
import random
import matplotlib.pyplot as plt

class MNISTDataset:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def normalize(self):
        self.X = self.X / 255.0


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
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(
                        value=[dataset.X[i].astype(np.uint8).tobytes()])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(
                        value=[int(dataset.Y[i])]))
                }
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
        print(f"Saved TFRecord to {path}")


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
                DatasetValidator.visualize_comparison(img1, img2, idx)

        print(f"\nAverage MSE: {np.mean(differences):.6f}")

        if np.mean(differences) < 1e-6 and label_mismatches == 0:
            print("✅ Datasets are PERFECTLY identical")
        else:
            print("⚠️ Differences detected")

    @staticmethod
    def visualize_comparison(img1, img2, idx):
        img1 = img1.reshape(28, 28)
        img2 = img2.reshape(28, 28)
        diff = np.abs(img1 - img2)

        plt.figure(figsize=(10, 3))

        plt.subplot(1, 3, 1)
        plt.title(f"Original {idx}")
        plt.imshow(img1, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title(f"Converted {idx}")
        plt.imshow(img2, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Difference")
        plt.imshow(diff, cmap='hot')

        plt.show()
