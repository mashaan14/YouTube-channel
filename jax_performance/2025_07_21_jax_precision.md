# Parallel Vision Transformer using JAX Device Mesh

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/G6c6zk0RhRM" frameborder="0" allowfullscreen></iframe>
</div>

## Contents

* [Acknowledgment](#acknowledgment)
* [Imports and configuration](#imports-and-configuration)
* [Preparing CIFAR-10](#preparing-cifar-10)
* [Create a device mesh](#create-a-device-mesh)
* [`VisionTransformer` class](#visiontransformer-class)
* [Testing different values for `param_dtype`](#testing-different-values-for-param_dtype)


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

* choice #1
```python
model_config = ConfigDict(dict(
    ...
    param_dtype=jnp.float32,
    ...
))
```

* choice #2
```python
model_config = ConfigDict(dict(
    ...
    param_dtype=jnp.float16,
    ...
))
```

* choice #3
```python
model_config = ConfigDict(dict(
    ...
    param_dtype=jnp.bfloat16,
    ...
))
```

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

![param_dtype_size](https://github.com/user-attachments/assets/d45e6f84-3c5b-444e-94a0-850ecf29bf53)

According to this plot, it is an easy decision to go with `param_dtype=jnp.bfloat16`. But if we go with this choice we will run into this error:

```console
XlaRuntimeError: UNIMPLEMENTED: Dot algorithm ALG_DOT_F16_F16_F32 is not supported.
```



---

```python
dtype=jnp.float32,          # Data type
```

```console
Total Parameters: 10,695,564 (42.8 MB) 
```

---

```python
dtype=jnp.bfloat16,          # Data type
```

```console
Total Parameters: 10,695,564 (42.8 MB) 
```

---

```python
dtype=jnp.float16,          # Data type
```

```console
Total Parameters: 10,695,564 (42.8 MB)
```

## bfloat16

```console
Step 20, Loss: 3.958667516708374, Elapsed Time: 0.11 seconds
Step 40, Loss: 2.3608651161193848, Elapsed Time: 0.08 seconds
Step 60, Loss: 2.1720101833343506, Elapsed Time: 0.10 seconds
Step 80, Loss: 2.0653440952301025, Elapsed Time: 0.05 seconds
Step 100, Loss: 1.9823248386383057, Elapsed Time: 0.07 seconds
Step 120, Loss: 1.9301824569702148, Elapsed Time: 0.05 seconds
Step 140, Loss: 1.9318405389785767, Elapsed Time: 0.09 seconds
Step 160, Loss: 1.8989524841308594, Elapsed Time: 0.09 seconds
Step 180, Loss: 1.9052046537399292, Elapsed Time: 0.09 seconds
Step 200, Loss: 1.8983386754989624, Elapsed Time: 0.08 seconds
Step 220, Loss: 1.8348573446273804, Elapsed Time: 0.09 seconds
Step 240, Loss: 1.8322759866714478, Elapsed Time: 0.07 seconds
Step 260, Loss: 1.8048137426376343, Elapsed Time: 0.08 seconds
Step 280, Loss: 1.809252142906189, Elapsed Time: 0.08 seconds
Step 300, Loss: 1.796458125114441, Elapsed Time: 0.09 seconds
Step 320, Loss: 1.7766698598861694, Elapsed Time: 0.10 seconds
Step 340, Loss: 1.745721697807312, Elapsed Time: 0.08 seconds
Step 360, Loss: 1.7272859811782837, Elapsed Time: 0.07 seconds
Step 380, Loss: 1.713021159172058, Elapsed Time: 0.10 seconds
Epoch 1 completed in 157.41 seconds
```
