# Collapse in Self-Supervised Learning

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/7NE0NH-PfkA" frameborder="0" allowfullscreen></iframe>
</div>


## Acknowledgment
Thanks to the authors for making their code available. I borrowed some code from [DINOv2](https://github.com/facebookresearch/dinov2) and [I-JEPA](https://github.com/facebookresearch/ijepa/tree/main) repositories.

## References

```bibtex
@Article{chen2020simsiam,
  author  = {Xinlei Chen and Kaiming He},
  title   = {Exploring Simple Siamese Representation Learning},
  journal = {arXiv preprint arXiv:2011.10566},
  year    = {2020},
}
```

```bibtex
@article{Jing2021UnderstandingDC,
  title   = {Understanding Dimensional Collapse in Contrastive Self-supervised Learning},
  author  = {Li Jing and Pascal Vincent and Yann LeCun and Yuandong Tian},
  journal = {arXiv preprint arXiv:2110.09348},
  year    = {2021}
}
```

```bibtex
@inproceedings{wang2022asym,
  title     = {On the Importance of Asymmetry for Siamese Representation Learning},
  author    = {Xiao Wang and Haoqi Fan and Yuandong Tian and Daisuke Kihara and Xinlei Chen},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```

## Importing and augmenting CIFAR-10

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import types
```

```python
Config = {
    'NUM_CLASSES': 10,
    'BATCH_SIZE': 128,
    'EPOCHS': 10,
    'LR': 0.001, #3e-4,
    'IMG_SIZE': 32,
    'DATA_MEANS': np.array([0.4914, 0.4822, 0.4465]), # mean of the CIFAR dataset, used for normalization
    'DATA_STD': np.array([0.2023, 0.1994, 0.2010]),   # standard deviation of the CIFAR dataset, used for normalization
    'CROP_SCALES': (0.8, 1.0),
    'SEED': 42,
    'PATCH_SIZE': 4,
    'IN_CHANNELS': 3,
    'EMBED_DIM': 256,
    'DEPTH': 6,
    'NUM_HEADS': 4,
    'MLP_DIM': 1024,
    'DROPOUT': 0.1,
    'HEAD_MLP_DIM': 2048,
    'HEAD_DIM': 128,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Convert to SimpleNamespace
config = types.SimpleNamespace(**Config)

class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

Here, I'm passing light augmentations to the teacher dataloader and stronger augmentations to student dataloader. I'm also setting `shuffle=False` on trainloaders just to compare the same image for teacher and student networks.

```python
# Transform 1: Light augmentations
transform_teacher = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(Config['DATA_MEANS'], Config['DATA_STD'])
])

# Transform 2: Stronger augmentations
transform_student = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=Config['CROP_SCALES']),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(Config['DATA_MEANS'], Config['DATA_STD']),
    transforms.RandomErasing(p=1, scale=(0.4, 0.5), ratio=(0.3, 3.3), value=0)
])

# Create datasets
trainset_teacher = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_teacher)

trainset_student = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_student)

testset_teacher = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_teacher)

testset_student = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_student)

# Create DataLoaders
trainloader_teacher = torch.utils.data.DataLoader(
    trainset_teacher, batch_size=Config['BATCH_SIZE'], shuffle=False, num_workers=2)

trainloader_student = torch.utils.data.DataLoader(
    trainset_student, batch_size=Config['BATCH_SIZE'], shuffle=False, num_workers=2)

testloader_teacher = torch.utils.data.DataLoader(
    testset_teacher, batch_size=Config['BATCH_SIZE'], shuffle=False, num_workers=2)

testloader_student = torch.utils.data.DataLoader(
    testset_student, batch_size=Config['BATCH_SIZE'], shuffle=False, num_workers=2)
```

```python
# Unnormalize function
def unnormalize(img, mean, std):
    img = img.numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

# Function to show images from a trainloader
def show_augmented_images(trainloader, mean, std, title):
    images, labels = next(iter(trainloader))  # one batch
    indices = list(range(0, 10))

    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(indices):
        img = unnormalize(images[idx], np.array(mean), np.array(std))
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.title(class_names[labels[idx].item()])
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

show_augmented_images(trainloader_teacher, Config['DATA_MEANS'], Config['DATA_STD'], title="Augmentations from trainloader_teacher")
show_augmented_images(trainloader_student, Config['DATA_MEANS'], Config['DATA_STD'], title="Augmentations from trainloader_student")
```

Here is how the augmentations look like:

![image](https://github.com/user-attachments/assets/8bc695e2-7d6d-4c41-8d39-db9109405fb5)

---

![image](https://github.com/user-attachments/assets/21b05ca8-4b04-4f5a-8b51-352b4dc2e4f2)



<script>
  document.addEventListener("DOMContentLoaded", function() {
    renderMathInElement(document.body, {
      delimiters: [
        {left: '$$', right: '$$', display: true}, // Display math (e.g., equations on their own line)
        {left: '$', right: '$', display: false},  // Inline math (e.g., within a sentence)
        {left: '\\(', right: '\\)', display: false}, // Another way to write inline math
        {left: '\\[', right: '\\]', display: true}   // Another way to write display math
      ]
    });
  });
</script>
