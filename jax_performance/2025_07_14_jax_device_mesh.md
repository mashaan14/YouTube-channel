# Parallel Vision Transformer using JAX Device Mesh

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/6WK7R1HBPOc" frameborder="0" allowfullscreen></iframe>
</div>

## Contents

* [Acknowledgment](#acknowledgment)
* [References](#references)
* [Imports and configuration](#imports-and-configuration)
* [Preparing CIFAR-10](#preparing-cifar-10)
* [Create a device mesh](#create-a-device-mesh)
* [`VisionTransformer` class](#visiontransformer-class)
  * [ViT with a device mesh](#vit-with-a-device-mesh)
  * [ViT without a device mesh](#vit-without-a-device-mesh) 
* [Initializing the model](#initializing-the-model)
* [Visualize parallelism with `shard_map`](#visualize-parallelism-with-create_device_mesh)
* [Training loop with `jax.profiler`](#training-loop-with-jaxprofiler)
* [Visualize batches](#visualize-batches)
* [Epoch run time](#epoch-run-time)
* [Peak memory allocation](#peak-memory-allocation)
* [Installing TensorBoard](#installing-tensorboard)

## Acknowledgment
These resources were helpful in preparing this post:
  - [Train a miniGPT language model with JAX](https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html)
  - [Scale up on multiple devices](https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html)

## References
```bibtex
@software{jax2018github,
  author  = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title   = {JAX: composable transformations of Python+NumPy programs},
  url     = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year    = {2018},
}
```

## Imports and configuration
We need `ml_collections` to prepare the configs and `grain` for dataloaders.
```bash
pip install ml_collections grain
```

```python
import jax
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding # For data and model parallelism (explained in more detail later)
from jax.experimental import mesh_utils
import jax.profiler


from flax import nnx
import optax
import grain.python as grain


import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from ml_collections import ConfigDict
import time
```

```python
data_config = ConfigDict(dict(
    batch_size=128,             
    img_size=32,                
    seed=12,                    
    crop_scales=(0.9, 1.0),     
    crop_ratio=(0.9, 1.1),  
    data_means=(0.4914, 0.4822, 0.4465),
    data_std=(0.2023, 0.1994, 0.2010)
))

model_config = ConfigDict(dict(
    num_epochs=1,              
    learning_rate=1e-3,         
    patch_size=4,               
    num_classes=10,             
    embed_dim=384,              
    mlp_dim=1536,               
    num_heads=8,                
    num_layers=6,               
    dropout_rate=0.1,           
))
```

## Preparing CIFAR-10

We need `torchvision` to import CIFAR-10 and perform random flipping and cropping. We also need the images to be in numpy arrays to be accepted by jax.

```python
def image_to_numpy(img):
  img = np.array(img, dtype=np.float32)
  img = (img / 255. - np.array(data_config.data_means)) / np.array(data_config.data_std)

  return img
```

```python
def numpy_collate(batch):
  if isinstance(batch[0], np.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple, list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    return np.array(batch)
```

```python
# images in the test set will only be converted into numpy arrays
test_transform = image_to_numpy
# images in the train set will be randomly flipped, cropped, and then converted to numpy arrays
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((data_config.img_size), scale=data_config.crop_scales, ratio=data_config.crop_ratio),
    image_to_numpy
])

train_dataset = torchvision.datasets.CIFAR10('data', train=True, transform=train_transform, download=True)
val_dataset = torchvision.datasets.CIFAR10('data', train=True, transform=test_transform, download=True)
train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
_, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])
test_set = torchvision.datasets.CIFAR10('data', train=False, transform=test_transform, download=True)
```

```python
train_sampler = grain.IndexSampler(
    len(train_dataset),  
    shuffle=True,            
    seed=data_config.seed,               
    shard_options=grain.NoSharding(),  # No sharding here, because it can be done inside the training loop
    num_epochs=model_config.num_epochs,            
)

val_sampler = grain.IndexSampler(
    len(val_dataset),  
    shuffle=False,         
    seed=data_config.seed,             
    shard_options=grain.NoSharding(),  
    num_epochs=model_config.num_epochs,          
)


train_loader = grain.DataLoader(
    data_source=train_dataset,
    sampler=train_sampler,                 
    operations=[
        grain.Batch(data_config.batch_size, drop_remainder=True),
    ]
)

# Test (validation) dataset `grain.DataLoader`.
val_loader = grain.DataLoader(
    data_source=val_dataset,
    sampler=val_sampler,                   
    operations=[
        grain.Batch(2*data_config.batch_size),
    ]
)
```

```python
num_images_to_plot = 10
images_plotted = 0
cols = 5
rows = (num_images_to_plot + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
axes = axes.flatten()

for i, batch in enumerate(train_loader):
    images = batch[0]
    labels = batch[1]

    for j in range(images.shape[0]):
        if images_plotted >= num_images_to_plot:
            break

        ax = axes[images_plotted]
        img_to_plot = images[j]
        if isinstance(img_to_plot, jax.Array):
            img_to_plot = np.array(img_to_plot)

        img_to_plot = img_to_plot * np.array(data_config.data_std) + np.array(data_config.data_means)
        ax.imshow(img_to_plot)
        ax.set_title(f"Label: {labels[j]}")
        ax.axis('off')
        images_plotted += 1

    if images_plotted >= num_images_to_plot:
        break

for k in range(images_plotted, len(axes)):
    fig.delaxes(axes[k])

plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/397818f2-9c25-40f2-a653-3c4ac1fad921)

## Create a device mesh

![drawings-02 002](https://github.com/user-attachments/assets/36a44167-d54d-4b87-918c-8cc3efcff8c9)

We need to create a device mesh before creating the vision transformer class, so we can specify sharding options in vision transformer layers.

```python
# Create a `Mesh` object representing TPU device arrangement.
mesh = Mesh(mesh_utils.create_device_mesh((4, 2)), ('batch', 'model'))
# mesh = Mesh(mesh_utils.create_device_mesh((8, 1)), ('batch', 'model'))
```

## `VisionTransformer` class

**⚠️ CAUTION: ⚠️**

_**I did not measure the loss or accuracy on CIFAR-10, so I do not know how this code performs in terms of loss optimization. The sole purpose of this class is to monitor memory usage and runtime for a single epoch.**_

### ViT with a device mesh

```python
class PatchEmbedding(nnx.Module):
    def __init__(self, img_size: int, patch_size: int, embed_dim: int, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        fh, fw = patch_size, patch_size # Filter height/width are your patch_size

        self.conv_proj = nnx.Conv(
            in_features=3,
            out_features=embed_dim,         
            kernel_size=(fh, fw),        
            strides=(fh, fw),            
            padding='VALID',             
            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
            dtype=dtype,                 
            rngs=rngs,                   
        )

    def __call__(self, x: jax.Array) -> jax.Array:

        x = self.conv_proj(x)
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])
        return x

class MLPBlock(nnx.Module): 
            def __init__(self, embed_dim: int, mlp_dim: int, dropout_rate: float, *, rngs: nnx.Rngs, dtype: jnp.dtype, mesh: 'jax.sharding.Mesh'):
                self.fc1 = nnx.Linear(
                    in_features=embed_dim,
                    out_features=mlp_dim,
                    kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
                    bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
                    rngs=rngs,
                    dtype=dtype
                )
                self.gelu = nnx.gelu
                self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
                self.fc2 = nnx.Linear(
                    in_features=mlp_dim,
                    out_features=embed_dim,
                    kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
                    bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
                    rngs=rngs,
                    dtype=dtype
                )
                self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

            def __call__(self, x: jax.Array) -> jax.Array:
                x = self.fc1(x)
                x = self.gelu(x)
                x = self.dropout1(x)
                x = self.fc2(x)
                x = self.dropout2(x)
                return x

class EncoderBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.0, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.norm1 = nnx.LayerNorm(epsilon=1e-6,
                                    num_features=embed_dim,
                                    scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
                                    bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P(None, 'model'))),
                                    rngs=rngs)
        self.attn = nnx.MultiHeadAttention(num_heads=num_heads,
                                            in_features=embed_dim,
                                            kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
                                            bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
                                            decode=False,
                                            rngs=rngs)
        self.norm2 = nnx.LayerNorm(epsilon=1e-6,
                                    num_features=embed_dim,
                                    scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
                                    bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P(None, 'model'))),
                                    rngs=rngs)

        self.mlp = MLPBlock(embed_dim, mlp_dim, dropout_rate, rngs=rngs, dtype=dtype, mesh=mesh)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nnx.Module):
    def __init__(self, img_size, patch_size, num_classes, embed_dim, num_layers, num_heads, mlp_dim, dropout_rate=0.0, *, rngs: nnx.Rngs = nnx.Rngs(0), dtype: jnp.dtype = jnp.float32):
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim, rngs=rngs, dtype=dtype)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))

        self.encoder_blocks = [
            EncoderBlock(embed_dim, num_heads, mlp_dim, dropout_rate, rngs=rngs, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.norm = nnx.LayerNorm(epsilon=1e-6,
                                    num_features=embed_dim,
                                    scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
                                    bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P(None, 'model'))),
                                    rngs=rngs)
        self.head = nnx.Linear(in_features=embed_dim,
                                out_features=num_classes,
                                kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
                                bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
                                rngs=rngs,
                                dtype=dtype)
        self.dtype = dtype # Store the global dtype for the model

    def __call__(self, x):
        x = x.astype(self.dtype)
        x = self.patch_embed(x)
        batch_size = x.shape[0]
        cls_tokens = jnp.tile(self.cls_token.value, (batch_size, 1, 1))
        x = jnp.concatenate((cls_tokens, x), axis=1)
        for block in self.encoder_blocks:
            x = block(x)

        cls_output = self.norm(x[:, 0])
        return self.head(cls_output)
```

### ViT without a device mesh

It is the same as the class above but with all `nnx.with_partitioning` removed.

```python
class PatchEmbedding(nnx.Module):
    def __init__(self, img_size: int, patch_size: int, embed_dim: int, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        fh, fw = patch_size, patch_size # Filter height/width are your patch_size

        self.conv_proj = nnx.Conv(
            in_features=3,
            out_features=embed_dim,          
            kernel_size=(fh, fw),        
            strides=(fh, fw),            
            padding='VALID',             
            dtype=dtype,                 
            rngs=rngs,                   
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv_proj(x)
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])
        return x

class MLPBlock(nnx.Module): 
            def __init__(self, embed_dim: int, mlp_dim: int, dropout_rate: float, *, rngs: nnx.Rngs, dtype: jnp.dtype):
                self.fc1 = nnx.Linear(
                    in_features=embed_dim,
                    out_features=mlp_dim,
                    rngs=rngs,
                    dtype=dtype
                )
                self.gelu = nnx.gelu
                self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
                self.fc2 = nnx.Linear(
                    in_features=mlp_dim,
                    out_features=embed_dim,
                    rngs=rngs,
                    dtype=dtype
                )
                self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

            def __call__(self, x: jax.Array) -> jax.Array:
                x = self.fc1(x)
                x = self.gelu(x)
                x = self.dropout1(x)
                x = self.fc2(x)
                x = self.dropout2(x)
                return x

class EncoderBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.0, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.norm1 = nnx.LayerNorm(epsilon=1e-6,
                                    num_features=embed_dim,
                                    rngs=rngs)
        self.attn = nnx.MultiHeadAttention(num_heads=num_heads,
                                            in_features=embed_dim,
                                            decode=False,
                                            rngs=rngs)
        self.norm2 = nnx.LayerNorm(epsilon=1e-6,
                                    num_features=embed_dim,
                                    rngs=rngs)

        self.mlp = MLPBlock(embed_dim, mlp_dim, dropout_rate, rngs=rngs, dtype=dtype)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nnx.Module):
    def __init__(self, img_size, patch_size, num_classes, embed_dim, num_layers, num_heads, mlp_dim, dropout_rate=0.0, *, rngs: nnx.Rngs = nnx.Rngs(0), dtype: jnp.dtype = jnp.float32):
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim, rngs=rngs, dtype=dtype)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)), dtype=dtype)
        self.pos_embed = nnx.Param(jax.random.normal(rngs.params(), (1, num_patches + 1, embed_dim)), dtype=dtype)

        self.encoder_blocks = [
            EncoderBlock(embed_dim, num_heads, mlp_dim, dropout_rate, rngs=rngs, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.norm = nnx.LayerNorm(epsilon=1e-6,
                                    num_features=embed_dim,
                                    rngs=rngs)
        self.head = nnx.Linear(in_features=embed_dim,
                                out_features=num_classes,
                                rngs=rngs,
                                dtype=dtype)
        self.dtype = dtype

    def __call__(self, x):
        x = x.astype(self.dtype)
        x = self.patch_embed(x)
        batch_size = x.shape[0]
        cls_tokens = jnp.tile(self.cls_token.value, (batch_size, 1, 1))
        x = jnp.concatenate((cls_tokens, x), axis=1)
        x = x + self.pos_embed.value

        for block in self.encoder_blocks:
            x = block(x)

        cls_output = self.norm(x[:, 0])
        return self.head(cls_output)
```

## Initializing the model

```python
def create_model(rngs):
    return VisionTransformer(data_config.img_size, model_config.patch_size, model_config.num_classes, model_config.embed_dim, model_config.num_layers, model_config.num_heads, model_config.mlp_dim, model_config.dropout_rate, rngs=rngs, dtype=jnp.float32)

model = create_model(rngs=nnx.Rngs(0))

model = VisionTransformer(data_config.img_size, 
                             model_config.patch_size, 
                             model_config.num_classes, 
                             model_config.embed_dim, 
                             model_config.num_layers, 
                             model_config.num_heads, 
                             model_config.mlp_dim, 
                             model_config.dropout_rate,
                             rngs=rngs,
                             dtype=jnp.float32)
```

```python
def loss_fn(model, images, labels):
    logits = model(images)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
    return loss, logits

@nnx.jit
def train_step(model: VisionTransformer, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, images, labels):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, images, labels)
    metrics.update(loss=loss, logits=logits, lables=labels)
    optimizer.update(grads)
```

```python
model = create_model(rngs=nnx.Rngs(0))
optimizer = nnx.Optimizer(model, optax.adam(1e-3))
metrics = nnx.MultiMetric(
  loss=nnx.metrics.Average('loss'),
)
rng = jax.random.PRNGKey(0)

metrics_history = {
  'train_loss': [],
}
```

## Visualize parallelism with `create_device_mesh`

```python
print(f'model.encoder_blocks[0].mlp.fc2.kernel.shape: {model.encoder_blocks[0].mlp.fc2.kernel.shape}')
print(model.encoder_blocks[0].mlp.fc2.kernel.sharding)
jax.debug.visualize_array_sharding(model.encoder_blocks[0].mlp.fc2.kernel)
print('---------------------------')
print(f'model.encoder_blocks[0].mlp.fc2.bias.shape: {model.encoder_blocks[0].mlp.fc2.bias.shape}')
print(model.encoder_blocks[0].mlp.fc2.bias.sharding)
jax.debug.visualize_array_sharding(model.encoder_blocks[0].mlp.fc2.bias)
```

![Screenshot 2025-06-22 at 11 49 57 AM](https://github.com/user-attachments/assets/a7888d01-02a5-456d-a3ad-9018db576dff)


```python
print(f'labels.shape: {labels.shape}')
print(labels.sharding)
jax.debug.visualize_array_sharding(labels)
```

* Mesh4×2
  
![Screenshot 2025-06-22 at 11 50 33 AM](https://github.com/user-attachments/assets/3203c9fa-0ed5-4fb5-a9fb-6813de06c033)

* Mesh 8×1
  
![Screenshot 2025-06-22 at 11 52 16 AM](https://github.com/user-attachments/assets/126e0567-5d21-4897-a256-1ce0542cb66f)


## Training loop with `jax.profiler`

```python
num_steps_per_epoch = len(train_dataset) // data_config.batch_size

log_dir = "./jax_profile_logs"
jax.profiler.start_trace(log_dir)
for epoch in range(model_config.num_epochs):
    step = 0
    epoch_start_time = time.time()
    for batch in train_loader:
        start_time = time.time()
        if step >= num_steps_per_epoch:
            break  # Skip extra steps beyond the intended epoch              

        # with a device mesh
        images = jax.device_put(batch[0], NamedSharding(mesh, P('batch', None)))
        labels = jax.device_put(batch[1], NamedSharding(mesh, P('batch')))

        # without a device mesh
        # images = batch[0]
        # labels = batch[1]

        train_step(model, optimizer, metrics, images, labels)

        if (step + 1) % 20 == 0:
          for metric, value in metrics.compute().items():
              metrics_history[f'train_{metric}'].append(value)
          metrics.reset()

          elapsed_time = time.time() - start_time
          print(f"Step {step + 1}, Loss: {metrics_history['train_loss'][-1]}, Elapsed Time: {elapsed_time:.2f} seconds")

        step += 1

    epoch_elapsed_time = time.time() - epoch_start_time
    print(f"Epoch {epoch + 1} completed in {epoch_elapsed_time:.2f} seconds")

jax.profiler.stop_trace()
```

## Visualize batches

![drawings-02 001](https://github.com/user-attachments/assets/d77bca39-e2cf-4f3b-9c5d-96775cd4ec04)

## Epoch run time
![peak_memory_allocation](https://github.com/user-attachments/assets/8bb41cf7-869c-4887-a142-13956ff286c3)

## Peak memory allocation
![epoch_run_time](https://github.com/user-attachments/assets/8a0b5e02-e4d3-4226-be73-9d06b6dd22da)

## Installing TensorBoard

I had issues visualizing TensorBoard reports in Kaggle notebooks and colab. Also, I was afraid of losing TensorBoard reports when the runtime restarts, which usually happens unexpectedly in colab. So, I had to install TensorBoard loacally and store all reports in one folder and visualize them. Here are the commands to install TensorBoard with a dedicated conda environment:

```bash
conda create -n tf_env
conda activate tf_env
conda install tensorboard
pip install -U tensorboard-plugin-profile
```

```bash
tensorboard --version
```

```console
2.19.0
```

After running `jax.profiler.start_trace(log_dir)` `jax.profiler.stop_trace()`, TensorBoard creates two files `.trace.json.gz` and `.xplane.pb`. For example if `log_dir = "./jax_profile_logs"`, the directory structure will be:

```
jax_profile_logs/
└── plugins/
    └── profile/
        └── <timestamp>/
            ├── .trace.json.gz
            ├── .xplane.pb
```

I downloaded these two files and place them in a similar directory structure in my Downloads folder. I ran 8 experiments, so I got 8 folders in total:

```
/
└── Users/
    └── mashaanalshammari/
        └── Downloads/
            └── plugins/
                └── profile/
                    ├── mesh_4_2_batch_128/
                    │   ├── .trace.json.gz
                    │   └── .xplane.pb
                    ├── mesh_4_2_batch_1024/
                    │   ├── .trace.json.gz
                    │   └── .xplane.pb
                    ├── mesh_4_2_batch_4096/
                    │   ├── .trace.json.gz
                    │   └── .xplane.pb
                    ├── mesh_8_1_batch_128/
                    │   ├── .trace.json.gz
                    │   └── .xplane.pb
                    ├── mesh_8_1_batch_1024/
                    │   ├── .trace.json.gz
                    │   └── .xplane.pb
                    ├── mesh_8_1_batch_4096/
                    │   ├── .trace.json.gz
                    │   └── .xplane.pb
                    ├── mesh_none_batch_128/
                    │   ├── .trace.json.gz
                    │   └── .xplane.pb
                    └── mesh_none_batch_1024/
                        ├── .trace.json.gz
                        └── .xplane.pb
```

Then run this command from the terminal to open TensorBoard:

```bash
tensorboard --logdir=/Users/mashaanalshammari/Downloads/
```

Here is a screenshot of the memory viewer, you can pick a profiler file from the dropdown menu.

![Screenshot 2025-06-21 at 12 44 35 AM](https://github.com/user-attachments/assets/03e2e861-ddc0-46e3-9121-cec6d1cc65d2)




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
