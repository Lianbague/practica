import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from MNIST_dataset_converter import DatasetValidator
from MNIST_dataset_converter import MNISTDataset
from MNIST_dataset_converter import DatasetConverter


def load_from_imagefolder(root_path):
    """
    Traverses folders 0-9, loads images, and flattens them.
    Returns: X (m, 784), Y (m,)
    """
    images = []
    labels = []
    
    print(f"Loading data from {root_path}...")
    
    for label in range(10):
        folder_path = os.path.join(root_path, str(label))
        if not os.path.exists(folder_path):
            continue
            
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(folder_path, filename)
                try:
                    # Open, convert to grayscale, and ensure 28x28 size
                    img = Image.open(img_path).convert('L')
                    img = img.resize((28, 28))
                    
                    # Flatten to 784 pixels and append
                    images.append(np.array(img).flatten())
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    
    return np.array(images), np.array(labels)

def load_from_hdf5(file_path):
    import h5py
    with h5py.File(file_path, 'r') as f:
        X = f['X'][:]
        Y = f['Y'][:]
    return X, Y

def load_from_tfrecord(file_path):
    import tensorflow as tf
    raw_dataset = tf.data.TFRecordDataset(file_path)
    
    # Define a parsing function (this depends on how you saved the TFRecord)
    def _parse_function(proto):
        # Define your feature description here based on how you saved the data
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        parsed_features = tf.io.parse_single_example(proto, feature_description)
        image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
        label = parsed_features['label']
        return image, label
    
    parsed_dataset = raw_dataset.map(_parse_function)
    
    images = []
    labels = []
    for image, label in parsed_dataset:
        images.append(image.numpy())
        labels.append(label.numpy())
    
    return np.array(images), np.array(labels)

def load_from_npz(file_path):
    data = np.load(file_path)
    X = data['X']
    Y = data['Y']
    return X, Y


# ------------- Neural Network Functions -------------

def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    # Stabilized softmax to prevent overflow
    exp_Z = np.exp(Z - np.max(Z, axis=0))
    return exp_Z / np.sum(exp_Z, axis=0)

def forward_prop(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def deriv_ReLU(Z):
    return Z > 0

def back_prop(z1, a1, z2, a2, w2, Y, X):
    m = Y.size
    OneHot_Y = one_hot(Y)
    dZ2 = a2 - OneHot_Y
    dW2 = 1/m * dZ2.dot(a1.T)
    db2 = 1/m * np.reshape(np.sum(dZ2, axis=1), (10, 1))
    
    dZ1 = w2.T.dot(dZ2) * deriv_ReLU(z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.reshape(np.sum(dZ1, axis=1), (10, 1))
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(a2):
    return np.argmax(a2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X)
        dW1, db1, dW2, db2 = back_prop(z1, a1, z2, a2, w2, Y, X)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print(f"Iteration: {i} | Accuracy: {get_accuracy(get_predictions(a2), Y):.4f}")
    return w1, b1, w2, b2



# ------------- Neural Network Class -------------

class NeuralNetworkMNIST:
    def __init__(self, train_dir, test_dir, format="imagefolder"):
        self.format = format
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.w1, self.b1, self.w2, self.b2 = None, None, None, None # Parameters will be initialized during training
    
    def load_data(self):
        if self.format == "imagefolder":
            self.X_train_raw, self.Y_train_raw = load_from_imagefolder(self.train_dir)
            self.X_test_raw, self.Y_test_raw = load_from_imagefolder(self.test_dir)
        elif self.format == "hdf5":
            self.X_train_raw, self.Y_train_raw = load_from_hdf5(self.train_dir)
            self.X_test_raw, self.Y_test_raw = load_from_hdf5(self.test_dir)
        elif self.format == "tfrecord":
            self.X_train_raw, self.Y_train_raw = load_from_tfrecord(self.train_dir)
            self.X_test_raw, self.Y_test_raw = load_from_tfrecord(self.test_dir)
        elif self.format == "npz":
            self.X_train_raw, self.Y_train_raw = load_from_npz(self.train_dir)
            self.X_test_raw, self.Y_test_raw = load_from_npz(self.test_dir)
        else:
            raise ValueError(f"Unsupported format: {self.format}")
        
    
    def preprocess_data(self):
        # Shuffle training data
        shuffler = np.random.permutation(len(self.X_train_raw))
        self.X_train_raw = self.X_train_raw[shuffler]
        self.Y_train_raw = self.Y_train_raw[shuffler]

        # Transpose and normalize: (784, m)
        self.X_train = self.X_train_raw.T / 255.
        self.Y_train = self.Y_train_raw

    def train(self, iterations=500, alpha=0.1):
        self.w1, self.b1, self.w2, self.b2 = gradient_descent(self.X_train, self.Y_train, iterations, alpha)
    
    def evaluate_on_test_set(self):
        X_test = self.X_test_raw.T / 255.
        Y_test = self.Y_test_raw
        _, _, _, a2_test = forward_prop(self.w1, self.b1, self.w2, self.b2, X_test)
        test_predictions = get_predictions(a2_test)
        test_accuracy = get_accuracy(test_predictions, Y_test)
        print(f"Test Accuracy: {test_accuracy:.4f}")

# ------------- Main Execution -------------
if __name__ == "__main__":
    train_dir = "mnist_imagefolder/train"
    test_dir = "mnist_imagefolder/test"
    
    nn_mnist = NeuralNetworkMNIST(train_dir, test_dir, format="imagefolder")
    nn_mnist.load_data()
    nn_mnist.preprocess_data()
    nn_mnist.train(iterations=500, alpha=0.1)
    nn_mnist.evaluate_on_test_set()



