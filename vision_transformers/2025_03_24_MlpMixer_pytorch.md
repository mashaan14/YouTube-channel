# MLP-Mixer in pytroch

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>


## Acknowledgment:

All code snippets in this blog were created by prompting Gemini in Google Colab. Thanks to the developers who made their code available.

## Loading CIFAR-10

> **prompt:** `load cifar10 from torchvision`

```python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Load CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

---

> **prompt:** `display images in the first batch from testloader`

```python
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(testloader)
# Correct the outdated syntax
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
```

![Screenshot 2025-02-23 at 10 37 38 AM](https://github.com/user-attachments/assets/1815114a-a923-496e-97ec-b90672a41757)

## MLP-Mixer class

> **prompt:** `write the mlp-mixer class in pytorch`

```python
import torch
import torch.nn as nn

class MLPMixerLayer(nn.Module):
    def __init__(self, in_features, token_dim, channel_dim):
        super().__init__()
        # The input of these layers should be of shape [batch_size, num_patches, in_features]
        self.token_mixing = nn.Sequential(
            nn.Linear(in_features, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, in_features)
        )
        self.channel_mixing = nn.Sequential(
            nn.Linear(in_features, channel_dim),
            nn.GELU(),
            nn.Linear(channel_dim, in_features)
        )
        # layer norm is applied on the last dimension.
        self.layer_norm1 = nn.LayerNorm(in_features)
        self.layer_norm2 = nn.LayerNorm(in_features)


    def forward(self, x):
        # Token mixing
        residual = x
        # input here should be [batch_size, num_patches, in_features]
        x = x.permute(0,2,1) # shape [batch_size, in_features, num_patches]
        x = self.layer_norm1(x)
        x = x.permute(0,2,1) # shape [batch_size, num_patches, in_features]
        x = self.token_mixing(x)
        x = x + residual

        # Channel Mixing
        residual = x
        x = self.layer_norm2(x)
        x = self.channel_mixing(x)
        x = x + residual

        return x


class MLPMixer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, token_dim, channel_dim):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        self.to_patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

        # in_features here is the dim, since the output of the Conv2d layer has a last dimension of dim
        self.mixer_blocks = nn.Sequential(*[MLPMixerLayer(dim, token_dim, channel_dim) for _ in range(depth)])

        self.layer_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
          nn.Linear(dim, num_classes)
        )


    def forward(self, x):
        patches = self.to_patch_embedding(x)
        batch_size, num_channels, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        # the shape should be batch_size, num_patches, dim
        patches = patches.view(batch_size, self.num_patches, -1)

        x = self.mixer_blocks(patches)
        x = self.layer_norm(x)
        x = torch.mean(x, dim=1)  # Global average pooling
        x = self.mlp_head(x)

        return x
```

## Model initialization

> **prompt:** `initialize the model using MLPMixer class`

```python
image_size = 32
patch_size = 4
num_classes = 10
dim = 64
depth = 2
token_dim = 32
channel_dim = 256

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MLPMixer(image_size, patch_size, num_classes, dim, depth, token_dim, channel_dim).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Model initialized.")
```

## Training loop

> **prompt:** `write the the training loop, use 10 epochs`

```python
num_epochs = 10

for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
```

```console
[1,  2000] loss: 1.904
[1,  4000] loss: 1.652
[1,  6000] loss: 1.519
[1,  8000] loss: 1.442
[1, 10000] loss: 1.377
[1, 12000] loss: 1.342
[2,  2000] loss: 1.261
[2,  4000] loss: 1.263
[2,  6000] loss: 1.221
[2,  8000] loss: 1.181
[2, 10000] loss: 1.167
[2, 12000] loss: 1.174
[3,  2000] loss: 1.118
[3,  4000] loss: 1.097
[3,  6000] loss: 1.094
[3,  8000] loss: 1.078
[3, 10000] loss: 1.073
[3, 12000] loss: 1.059
[4,  2000] loss: 1.036
[4,  4000] loss: 1.016
[4,  6000] loss: 1.008
[4,  8000] loss: 1.008
[4, 10000] loss: 1.024
[4, 12000] loss: 1.008
[5,  2000] loss: 0.951
[5,  4000] loss: 0.978
[5,  6000] loss: 0.967
[5,  8000] loss: 0.957
[5, 10000] loss: 0.968
[5, 12000] loss: 0.966
[6,  2000] loss: 0.939
[6,  4000] loss: 0.924
[6,  6000] loss: 0.923
[6,  8000] loss: 0.912
[6, 10000] loss: 0.931
[6, 12000] loss: 0.920
[7,  2000] loss: 0.887
[7,  4000] loss: 0.877
[7,  6000] loss: 0.896
[7,  8000] loss: 0.889
[7, 10000] loss: 0.910
[7, 12000] loss: 0.883
[8,  2000] loss: 0.828
[8,  4000] loss: 0.845
[8,  6000] loss: 0.861
[8,  8000] loss: 0.864
[8, 10000] loss: 0.843
[8, 12000] loss: 0.890
[9,  2000] loss: 0.811
[9,  4000] loss: 0.829
[9,  6000] loss: 0.832
[9,  8000] loss: 0.839
[9, 10000] loss: 0.843
[9, 12000] loss: 0.828
[10,  2000] loss: 0.777
[10,  4000] loss: 0.795
[10,  6000] loss: 0.815
[10,  8000] loss: 0.825
[10, 10000] loss: 0.810
[10, 12000] loss: 0.803
Finished Training
```

## Testing

> **prompt:** `pull a random batch from testloader and test it using the trained model`

```python
# Get a random batch from the testloader
dataiter = iter(testloader)
images, labels = next(dataiter)
images = images.to(device)
labels = labels.to(device)

# Test the model
outputs = model(images)

# Get predicted labels
_, predicted = torch.max(outputs, 1)

# Print predictions and actual labels
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))
print('Actual: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
```

```console
Predicted:  cat   ship  ship  plane
Actual:  cat   ship  ship  plane
```

---

> **prompt:** `test all batches in testloader and create a confusion matrix`

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Initialize the confusion matrix
confusion_mat = np.zeros((10, 10), dtype=int)

# Iterate over all batches in the testloader
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Update the confusion matrix
        for i in range(len(labels)):
            confusion_mat[labels[i], predicted[i]] += 1

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = confusion_mat.max() / 2.
for i, j in np.ndindex(confusion_mat.shape):
    plt.text(j, i, confusion_mat[i, j],
             horizontalalignment="center",
             color="white" if confusion_mat[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
```

![image](https://github.com/user-attachments/assets/c614e4ab-07fa-4c38-bc4f-0a08560e6ba8)

---

> **prompt:** `print the test accuracy for testloader`

```python
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
```

```console
Accuracy of the network on the 10000 test images: 67 %
```

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
