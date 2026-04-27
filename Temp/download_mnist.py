import os
from torchvision import datasets
from PIL import Image

# Descargar MNIST
train_dataset = datasets.MNIST(root="./data", train=True, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, download=True)

def save_dataset(dataset, split):
    base_path = f"mnist_imagefolder/{split}"

    for i, (img, label) in enumerate(dataset):
        # Crear carpeta de la clase si no existe
        class_path = os.path.join(base_path, str(label))
        os.makedirs(class_path, exist_ok=True)

        # Guardar imagen
        img_path = os.path.join(class_path, f"{i}.png")
        img.save(img_path)

# Guardar train y test
save_dataset(train_dataset, "train")
save_dataset(test_dataset, "test")

print("✅ MNIST convertido a ImageFolder")