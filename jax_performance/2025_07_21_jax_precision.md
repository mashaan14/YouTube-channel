# Training with Different JAX Precisions 

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/znxzsPzNK6c" frameborder="0" allowfullscreen></iframe>
</div>

## Contents

* [Acknowledgment](#acknowledgment)
* [Imports and configuration](#imports-and-configuration)
* [Preparing CIFAR-10](#preparing-cifar-10)
* [Create a device mesh](#create-a-device-mesh)
* [`VisionTransformer` class](#visiontransformer-class)
* [Testing different values for `param_dtype`](#testing-different-values-for-param_dtype)
* [Testing different values for `dtype`](#testing-different-values-for-dtype)
* [`jax.lax.Precision`](#jaxlaxprecision)
* [`jax.default_matmul_precision`](#jaxdefault_matmul_precision)
* [Why I don’t see a memory impact from precision alone?](#why-i-dont-see-a-memory-impact-from-precision-alone)


## Acknowledgment
These resources were helpful in preparing this post:
  - [class jax.lax.Precision](https://docs.jax.dev/en/latest/jax.lax.html#jax.lax.Precision)
  - [Part 1.1: Training Larger Models on a Single GPU](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/scaling/JAX/single_gpu_techniques.html)

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
    dtype=jnp.bfloat16,          
    param_dtype=jnp.float32,    
    rngs=nnx.Rngs(0)            
))
```

![drawings-02 001](https://github.com/user-attachments/assets/ba8dda40-de6d-47e7-94f1-38795d1ed3b3)


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

## Create a device mesh

Here I'm creating a device mesh to distribute the data across 8 devices. I'm not using model parallelism.

```python
# Create a `Mesh` object representing TPU device arrangement.
mesh = Mesh(mesh_utils.create_device_mesh((8, 1)), ('batch', 'model'))
```

```python
print(nnx.tabulate(model, jnp.ones((1, 32, 32, 3))))
```

## `VisionTransformer` class

**⚠️ CAUTION: ⚠️**

_**I did not measure the loss or accuracy on CIFAR-10, so I do not know how this code performs in terms of loss optimization. The sole purpose of this class is to monitor memory usage as we change the data type.**_

**⚠️ CAUTION: ⚠️**

_**I commented out all calls of `nnx.with_partitioning` because it won't work with `nnx.tabulate`, which I'm going to call later. For data parallelism uncomment `nnx.with_partitioning`**_

```python
class PatchEmbedding(nnx.Module):
    def __init__(self, img_size, patch_size, embed_dim, dtype, param_dtype, rngs: nnx.Rngs(0)):
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        fh, fw = patch_size, patch_size # Filter height/width are your patch_size

        self.conv_proj = nnx.Conv(
            in_features=3,
            out_features=embed_dim,          
            kernel_size=(fh, fw),        
            strides=(fh, fw),            
            padding='VALID',             
            # kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
            # bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
            dtype=dtype,                 
            param_dtype=param_dtype,           
            rngs=rngs,                   
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv_proj(x)
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])
        return x

class EncoderBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate, dtype, param_dtype, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(epsilon=1e-6,
                                    num_features=embed_dim,
                                    # scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
                                    # bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P(None, 'model'))),
                                    dtype=dtype,
                                    param_dtype=param_dtype,
                                    rngs=rngs)
        self.attn = nnx.MultiHeadAttention(num_heads=num_heads,
                                            in_features=embed_dim,
                                            # kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
                                            # bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
                                            dtype=dtype,
                                            param_dtype=param_dtype,
                                            rngs=rngs)
        self.norm2 = nnx.LayerNorm(epsilon=1e-6,
                                    num_features=embed_dim,
                                    # scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
                                    # bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P(None, 'model'))),
                                    dtype=dtype,
                                    param_dtype=param_dtype,
                                    rngs=rngs)

        self.linear1 = nnx.Linear(
                    in_features=embed_dim,
                    out_features=mlp_dim,
                    # kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
                    # bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs
                )
        self.dropout1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.linear2 = nnx.Linear(
                    in_features=mlp_dim,
                    out_features=embed_dim,
                    # kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
                    # bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs
                )
        self.dropout2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x), decode=False)

        mlp_out = self.norm2(x)
        mlp_out = self.linear1(mlp_out)
        mlp_out = nnx.gelu(mlp_out)
        mlp_out = self.dropout1(mlp_out)
        mlp_out = self.linear2(mlp_out)
        mlp_out = self.dropout2(mlp_out)
        x = x + mlp_out

        return x

class VisionTransformer(nnx.Module):
    def __init__(self, img_size, patch_size, num_classes, embed_dim, num_layers, num_heads, mlp_dim, dropout_rate, dtype, param_dtype, rngs: nnx.Rngs = nnx.Rngs(0)):
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim, dtype, param_dtype, rngs=rngs)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)), dtype=dtype)
        self.pos_embed = nnx.Param(jax.random.normal(rngs.params(), (1, num_patches + 1, embed_dim)), dtype=dtype)

        self.encoder_blocks = [
            EncoderBlock(embed_dim, num_heads, mlp_dim, dropout_rate, dtype, param_dtype, rngs=rngs)
            for _ in range(num_layers)
        ]
        self.norm = nnx.LayerNorm(epsilon=1e-6,
                                    num_features=embed_dim,
                                    # scale_init=nnx.with_partitioning(nnx.initializers.ones_init(), NamedSharding(mesh, P(None, 'model'))),
                                    # bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P(None, 'model'))),
                                    dtype=dtype,
                                    param_dtype=param_dtype,
                                    rngs=rngs)
        self.head = nnx.Linear(in_features=embed_dim,
                                out_features=num_classes,
                                # kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
                                # bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
                                dtype=dtype,
                                param_dtype=param_dtype,
                                rngs=rngs)
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

## Testing different values for `param_dtype`

Now let's change our config dict to run `VisionTransformer` class for each of the following choices:

| `param_dtype`       |
|----------------|
| `jnp.float32`   |
| `jnp.float16`   |
| `jnp.bfloat16`   |


Now let's run `nnx.tabulate` to check the parameters size. Here is an example ouutput of `nnx.tabulate`:

```console
VisionTransformer Summary                                             
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ path               ┃ type               ┃ inputs            ┃ outputs            ┃ Param             ┃ RngState ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│                    │ VisionTransformer  │ float32[1,32,32,… │ bfloat16[1,10]     │ cls_token:        │ 2 (12 B) │
│                    │                    │                   │                    │   dtype: type     │          │
│                    │                    │                   │                    │   value:          │          │
│                    │                    │                   │                    │ float32[1,1,384]  │          │
│                    │                    │                   │                    │ pos_embed:        │          │
│                    │                    │                   │                    │   dtype: type     │          │
│                    │                    │                   │                    │   value:          │          │
│                    │                    │                   │                    │ float32[1,65,384] │          │
│                    │                    │                   │                    │                   │          │
│                    │                    │                   │                    │ 10,695,562 (42.8  │          │
│                    │                    │                   │                    │ MB)               │          │
├────────────────────┼────────────────────┼───────────────────┼────────────────────┼───────────────────┼──────────┤
│ patch_embed        │ PatchEmbedding     │ bfloat16[1,32,32… │ bfloat16[1,64,384] │ 18,816 (75.3 KB)  │          │
├────────────────────┼────────────────────┼───────────────────┼────────────────────┼───────────────────┼──────────┤
│ patch_embed/conv_… │ Conv               │ bfloat16[1,32,32… │ bfloat16[1,8,8,38… │ bias:             │          │
│                    │                    │                   │                    │ float32[384]      │          │
│                    │                    │                   │                    │ kernel:           │          │
│                    │                    │                   │                    │ float32[4,4,3,38… │          │
│                    │                    │                   │                    │                   │          │
│                    │                    │                   │                    │ 18,816 (75.3 KB)  │          │
├────────────────────┼────────────────────┼───────────────────┼────────────────────┼───────────────────┼──────────┤
│ encoder_blocks/0   │ EncoderBlock       │ float32[1,65,384] │ float32[1,65,384]  │ 1,774,464 (7.1    │ 2 (12 B) │
│                    │                    │                   │                    │ MB)               │          │
├────────────────────┼────────────────────┼───────────────────┼────────────────────┼───────────────────┼──────────┤
│ encoder_blocks/0/… │ LayerNorm          │ float32[1,65,384] │ bfloat16[1,65,384] │ bias:             │          │
│                    │                    │                   │                    │ float32[384]      │          │
│                    │                    │                   │                    │ scale:            │          │
│                    │                    │                   │                    │ float32[384]      │          │
│                    │                    │                   │                    │                   │          │
│                    │                    │                   │                    │ 768 (3.1 KB)      │          │
├────────────────────┼────────────────────┼───────────────────┼────────────────────┼───────────────────┼──────────┤
│ encoder_blocks/0/… │ MultiHeadAttention │ -                 │ bfloat16[1,65,384] │ 591,360 (2.4 MB)  │          │
│                    │                    │ bfloat16[1,65,38… │                    │                   │          │
│                    │                    │ decode: false     │                    │                   │          │
├────────────────────┼────────────────────┼───────────────────┼────────────────────┼───────────────────┼──────────┤
```

This is a comparison of parameters size based on `param_dtype`:

![param_dtype_size](https://github.com/user-attachments/assets/d640e484-7316-4161-8143-b7ffac1d35d1)

According to this plot, it is an easy decision to go with `param_dtype=jnp.bfloat16`. But if we go with this choice we will run into this error:

```console
XlaRuntimeError: UNIMPLEMENTED: Dot algorithm ALG_DOT_F16_F16_F32 is not supported.
```

## Testing different values for `dtype`

Since we're restricted to use `param_dtype=jnp.float32`, let's play with `dtype` and test four different choices:

| `batch_size`       | `dtype`       |
|----------------|----------------|
| `128`   | `jnp.float32`   |
| `128`   | `jnp.bfloat16`   |
| `4096`   | `jnp.float32`   |
| `4096`   | `jnp.bfloat16`   |


![peak_memory_allocation](https://github.com/user-attachments/assets/4a500f6c-5149-4015-8015-884ca68258bd)

I was expecting the difference with larger batch size, but they were so close with `batch_size=128`. After digging into (HLO Op stats) in TensorBoard and serching for this text `16,32,32,3`, which represents the input size divided into 8 devices `128/8=16`, I found a copy operation that converts the input tensor to `bfloat16`: 

![Screenshot 2025-06-29 at 3 49 12 PM](https://github.com/user-attachments/assets/245b6570-ddd5-40b6-bf6d-5ec141be896c)

But that copy operation was not performed with `batch_size=4096`:

![Screenshot 2025-06-29 at 3 50 54 PM](https://github.com/user-attachments/assets/b86d9691-7912-454a-bc37-44db2467683c)

Honestly, I don't know what causes this copy operation, but it is the reason why these two precisions perform the same with `batch_size=128`.

## `jax.lax.Precision`

[`jax.lax.Precision`](https://docs.jax.dev/en/latest/jax.lax.html#jax.lax.Precision) has three options that control the precision: `DEFAULT`, `HIGH`, and `HIGHEST`. Here is an example on how to use it with a linear layer:

```python
self.head = nnx.Linear(in_features=embed_dim,
                        out_features=num_classes,
                        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), NamedSharding(mesh, P(None, 'model'))),
                        bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), NamedSharding(mesh, P('model'))),
                        dtype=dtype,
                        # param_dtype=param_dtype,
                        precision=jax.lax.Precision.HIGHEST,
                        rngs=rngs)
```

In my experiments all three options have the same impact. This might indicate that XLA was using a certain precision and `jax.lax.Precision` could not override it.

![precision_time](https://github.com/user-attachments/assets/de4ee2ef-04bf-4ee1-bd49-ea31d52fd39a)

![precision_memory](https://github.com/user-attachments/assets/5bd8ea28-2161-451a-85b5-285195ffd17f)

## `jax.default_matmul_precision`

I warpped the training loop in a `with jax.default_matmul_precision('float32')`. But testing two choices 'float32' and 'bfloat16', the performance was the same:

![matmul_dtype_time](https://github.com/user-attachments/assets/7b2ff1ec-1b48-44b0-a47a-6f251f5320fc)

![matmul_dtype_memory](https://github.com/user-attachments/assets/d378cb13-169b-497b-a144-a6f7f0fb1133)


## Why I don’t see a memory impact from precision alone?

* I could be doing something wrong (that's always a possibility 😄). I did my best to minimize the chance of making an error. I turned off the parallelism and run on a single device, I also passed the options directly to layers instead of passing them to the model.
* XLA has the final say on precision and it has its own heuristics to control that. As a user you control how many devices you want to run and you also control the batch size.

