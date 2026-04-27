import os
import random
from torchvision import datasets

# Descargar MNIST
full_train = datasets.MNIST(root="./data", train=True, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, download=True)

# Mezclar índices
indices = list(range(len(full_train)))
random.shuffle(indices)

# Split 80/20
split = int(0.8 * len(indices))
train_idx = indices[:split]
val_idx = indices[split:]

def save_subset(dataset, indices, split_name):
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        path = f"mnist_imagefolder_with_validation/{split_name}/{label}"
        os.makedirs(path, exist_ok=True)
        img.save(f"{path}/{i}.png")

# Guardar datasets
save_subset(full_train, train_idx, "train")
save_subset(full_train, val_idx, "val")
save_subset(test_dataset, range(len(test_dataset)), "test")

print("✅ Train / Val / Test creados")