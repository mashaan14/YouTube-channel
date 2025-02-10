# JAX Speed Test


<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/1SQFVYVSuyE" frameborder="0" allowfullscreen></iframe>
</div>



## Acknowledgment:
- [Just In Time Compilation with JAX](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html#main-content)
- [🔪 JAX - The Sharp Bits 🔪](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions)
- I borrowed the CNN architecture from [FLAX examples](https://github.com/google/flax/blob/main/examples/mnist/train.py).
- I borrowed the cifar10 preprocessing code from this github [repository](https://github.com/satojkovic/vit-jax-flax/tree/main) by satojkovic.

## References:
```bibtex
@software{jax2018github,
 author  = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
 title  = {JAX: composable transformations of Python+NumPy programs},
 url    = {http://github.com/google/jax},
 version = {0.3.13},
 year   = {2018},
}
```

## Import libraries

```python
# Standard libraries
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from torch.utils import data

# Plotting libraries
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython.display import Javascript  # Restrict height of output cell.

# jax
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import flax
from flax import linen as nn
from flax.training import train_state
import optax

#torchvision
import torchvision
import torchvision.transforms as transforms

# scikit-learn
from sklearn.datasets import (make_blobs, make_circles)
from sklearn.model_selection import train_test_split
```

```python
plt.style.use('dark_background')
plot_colors = cm.tab10.colors
timing_list = []
```

## Hyper-parameters

```python
IMAGE_SIZE = 32
BATCH_SIZE = 128
DATA_MEANS = np.array([0.49139968, 0.48215841, 0.44653091])
DATA_STD = np.array([0.24703223, 0.24348513, 0.26158784])
CROP_SCALES = (0.8, 1.0)
CROP_RATIO = (0.9, 1.1)
SEED = 42
```

## Toy Dataset
Here we create a two dimensional toy dataset using `data.Dataset` class. Creating a `Dataset` instance will help us creating a `Dataloader` for training and testing.

```python
class ToyDataset(data.Dataset):

  def __init__(self, size, seed):
    super().__init__()
    self.size = size
    self.np_rng = np.random.RandomState(seed=seed)
    self.make_nested_classes()

  def make_nested_classes(self):
    X, y = make_blobs(n_samples=int(self.size*0.2), n_features=2, centers=2, cluster_std=1.9, random_state=SEED)
    X1, y1 = make_circles(n_samples=(int(self.size*0.6), int(self.size*0.2)), noise=0.05, factor=0.1, random_state=SEED)
    # increase the radius
    X1 = X1*3
    # move along the x-axis
    X1[:,0] = X1[:,0]+2.5
    # move along the y-axis
    X1[:,1] = X1[:,1]-7

    X = np.concatenate((X, X1), axis=0)
    y = np.concatenate((y, y1), axis=0)

    self.data = X
    self.label = y

  def __len__(self):
    return self.size

  def __getitem__(self, idx):
    data_point = self.data[idx]
    data_label = self.label[idx]
    return data_point, data_label
```

```python
dataset = ToyDataset(size=10000, seed=SEED)
dataset
```

```console
<__main__.ToyDataset at 0x7d27b41450f0>
```

```python
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(dataset.data[:,0], dataset.data[:,1], marker='o', color=np.array(plot_colors)[dataset.label])

ax.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,
            labelbottom=False,labeltop=False,labelleft=False,labelright=False);
ax.set(xlabel=None, ylabel=None)
plt.show()
```

![image](https://github.com/user-attachments/assets/f5b0cae1-9388-449d-85a6-250ebcf0fc4e)

## Train/Test splits
We split the dataset to 80% for training and 20% for testing using `data.random_split`. Then we package these splits in dataloaders. We specified `collate_fn=numpy_collate` to create numpy batches instead of torch tensor batches, which is the default option.

```python
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(SEED))
```

```python
# A helper function that converts batches into numpy arrays instead of the default option which is torch tensors
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
train_data_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=numpy_collate)
test_data_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=numpy_collate)
```

## MLP
The `MLPClassifier` creates a neural net instance where the hidden layers are specified by the user. It applies `relu` function to the hidden layer output and applies `log_softmax` to the output. We initialize the `MLPClassifier` to one hidden layer with ten neurons.

```python
class MLPClassifier(nn.Module):
    hidden_layers: int
    hidden_dim: int
    n_classes: int

    @nn.compact
    def __call__(self, x):
        for layer in range(self.hidden_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.n_classes)(x)
        x = nn.log_softmax(x)
        return x
```

```python
model = MLPClassifier(hidden_layers=1, hidden_dim=10, n_classes=2)
print(model)
```

```console
MLPClassifier(
    # attributes
    hidden_layers = 1
    hidden_dim = 10
    n_classes = 2
)
```

![image](https://github.com/user-attachments/assets/69b3d4ec-403d-4058-b5b9-5821fcc3870b)

We set the optimizer to adam using `optax` library. Then we initialized the model using random parameters. For the loss function, we used cross entropy to evaluate the model predictions.

```python
optimizer = optax.adam(learning_rate=0.01)

rng, inp_rng, init_rng = jax.random.split(jax.random.PRNGKey(SEED), 3)
params = model.init(jax.random.PRNGKey(SEED),
                    jax.random.normal(inp_rng, (BATCH_SIZE, dataset.data.shape[1])))

model_state = train_state.TrainState.create(apply_fn=model.apply,
                                            params=params,
                                            tx=optimizer)
```

## MLP Training (JIT mode)

![image](https://github.com/user-attachments/assets/6e1def27-f264-4cb7-b535-21d0650ec8cc)

```python
@jax.jit
def apply_model(state, images, labels):
  """Computes gradients, loss and accuracy for a single batch."""

  def loss_fn(params):
    logits = state.apply_fn(params, images)
    one_hot = jax.nn.one_hot(labels, logits.shape[1])
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy

@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)

def train_epoch(state, data_loader):
  """Train for a single epoch."""

  epoch_loss = []
  epoch_accuracy = []

  for batch in data_loader:
    batch_images, batch_labels = batch
    grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
    state = update_model(state, grads)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy

def train_model(state, train_data_loader, num_epochs):
  # Training loop
  for epoch in range(num_epochs):
    state, train_loss, train_accuracy = train_epoch(state, train_data_loader)
    print(f'epoch: {epoch:03d}, train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.4f}')
  return state
```

```python
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))
func_time = %timeit -o train_model(model_state, train_data_loader, num_epochs=1)
```

```console
epoch: 000, train loss: 0.5810, train accuracy: 0.7098
epoch: 000, train loss: 0.5810, train accuracy: 0.7098
epoch: 000, train loss: 0.5810, train accuracy: 0.7098
epoch: 000, train loss: 0.5810, train accuracy: 0.7098
epoch: 000, train loss: 0.5810, train accuracy: 0.7098
epoch: 000, train loss: 0.5810, train accuracy: 0.7098
epoch: 000, train loss: 0.5810, train accuracy: 0.7098
epoch: 000, train loss: 0.5810, train accuracy: 0.7098
65.7 ms ± 1.73 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
timing_list.append(['MLP', 'jit', np.mean(func_time.timings), np.std(func_time.timings)])
```

## MLP Training (no JIT mode)

```python
def apply_model_no_jit(state, images, labels):
  """Computes gradients, loss and accuracy for a single batch."""

  def loss_fn_no_jit(params):
    logits = state.apply_fn(params, images)
    one_hot = jax.nn.one_hot(labels, logits.shape[1])
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn_no_jit, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy

def update_model_no_jit(state, grads):
  return state.apply_gradients(grads=grads)

def train_epoch_no_jit(state, data_loader):
  """Train for a single epoch."""

  epoch_loss = []
  epoch_accuracy = []

  for batch in data_loader:
    batch_images, batch_labels = batch
    grads, loss, accuracy = apply_model_no_jit(state, batch_images, batch_labels)
    state = update_model_no_jit(state, grads)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy

def train_model_no_jit(state, train_data_loader, num_epochs):
  # Training loop
  for epoch in range(num_epochs):
    state, train_loss, train_accuracy = train_epoch_no_jit(state, train_data_loader)
    print(f'epoch: {epoch:03d}, train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.4f}')
  return state
```

```python
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))
func_time = %timeit -o trained_model_state = train_model_no_jit(model_state, train_data_loader, num_epochs=1)
```

```console
epoch: 000, train loss: 0.5810, train accuracy: 0.7098
epoch: 000, train loss: 0.5810, train accuracy: 0.7098
epoch: 000, train loss: 0.5810, train accuracy: 0.7098
epoch: 000, train loss: 0.5810, train accuracy: 0.7098
epoch: 000, train loss: 0.5810, train accuracy: 0.7098
epoch: 000, train loss: 0.5810, train accuracy: 0.7098
epoch: 000, train loss: 0.5810, train accuracy: 0.7098
epoch: 000, train loss: 0.5810, train accuracy: 0.7098
2.36 s ± 24.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
timing_list.append(['MLP', 'no jit', np.mean(func_time.timings), np.std(func_time.timings)])
```

## Load CIFAR10 Dataset

```python
# A helper function that normalizes the images between the values specified by the hyper-parameters.
def image_to_numpy(img):
  img = np.array(img, dtype=np.float32)
  img = (img / 255. - DATA_MEANS) / DATA_STD
  return img
```

```python
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# images in the test set will only be converted into numpy arrays
test_transform = image_to_numpy
# images in the train set will be randomly flipped, cropped, and then converted to numpy arrays
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=CROP_SCALES, ratio=CROP_RATIO),
    image_to_numpy
])

# Validation set should not use train_transform.
train_dataset = torchvision.datasets.CIFAR10('data', train=True, transform=train_transform, download=True)
val_dataset = torchvision.datasets.CIFAR10('data', train=True, transform=test_transform, download=True)
train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000], generator=torch.Generator().manual_seed(SEED))
_, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(SEED))
test_set = torchvision.datasets.CIFAR10('data', train=False, transform=test_transform, download=True)

train_data_loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2, persistent_workers=True, collate_fn=numpy_collate,
)
val_data_loader = torch.utils.data.DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=2, persistent_workers=True, collate_fn=numpy_collate,
)
test_data_loader = torch.utils.data.DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=2, persistent_workers=True, collate_fn=numpy_collate,
)
```

```console
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to data/cifar-10-python.tar.gz
100%|██████████| 170498071/170498071 [00:03<00:00, 53575476.16it/s]
Extracting data/cifar-10-python.tar.gz to data
Files already downloaded and verified
Files already downloaded and verified
```

## CNN

```python
class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x
```

```python
model = CNN()

optimizer = optax.adam(learning_rate=1e-4)

rng, inp_rng, init_rng = jax.random.split(jax.random.PRNGKey(SEED), 3)
params = model.init(jax.random.PRNGKey(SEED),
                    jax.random.normal(inp_rng, (BATCH_SIZE, 32, 32, 3)))

model_state = train_state.TrainState.create(apply_fn=model.apply,
                                            params=params,
                                            tx=optimizer)
```

## CNN Training (JIT mode)

```python
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))
func_time = %timeit -o train_model(model_state, train_data_loader, num_epochs=1)
```

```console
epoch: 000, train loss: 1.6836, train accuracy: 0.4074
epoch: 000, train loss: 1.6942, train accuracy: 0.3986
epoch: 000, train loss: 1.6877, train accuracy: 0.4070
epoch: 000, train loss: 1.6871, train accuracy: 0.4008
epoch: 000, train loss: 1.6854, train accuracy: 0.4038
epoch: 000, train loss: 1.6895, train accuracy: 0.4048
epoch: 000, train loss: 1.6913, train accuracy: 0.4031
epoch: 000, train loss: 1.6914, train accuracy: 0.4038
7.45 s ± 90.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
timing_list.append(['CNN', 'jit', np.mean(func_time.timings), np.std(func_time.timings)])
```

## CNN Training (no JIT mode)

```python
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))
func_time = %timeit -o trained_model_state = train_model_no_jit(model_state, train_data_loader, num_epochs=1)
```

```console
epoch: 000, train loss: 1.6863, train accuracy: 0.4041
epoch: 000, train loss: 1.6924, train accuracy: 0.4013
epoch: 000, train loss: 1.6858, train accuracy: 0.4047
epoch: 000, train loss: 1.6865, train accuracy: 0.4044
epoch: 000, train loss: 1.6862, train accuracy: 0.4032
epoch: 000, train loss: 1.6871, train accuracy: 0.4038
epoch: 000, train loss: 1.6895, train accuracy: 0.4024
epoch: 000, train loss: 1.6866, train accuracy: 0.4046
29 s ± 401 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

```python
timing_list.append(['CNN', 'no jit', np.mean(func_time.timings), np.std(func_time.timings)])
```

## Plotting the running time

```python
df = pd.DataFrame(timing_list, columns=('NN type', 'jit Mode', 'mean run time (seconds)', 'std run time (seconds)'))
sns.catplot(df, x='mean run time (seconds)', hue='jit Mode', col='NN type',
            kind='bar', sharex=False, palette="muted")
plt.show()
```

![image](https://github.com/user-attachments/assets/ab65cf92-236d-41ea-ad14-c70021d21b38)


## Printing cache size

This code prints thr model cache size. Notice that the cache size changed twice

* from 0 to 1 in the first batch, because the code just got translated.
* from 1 to 2 in the last batch, because the last batch has different dimensions so the code got translated again.

Other than that, the cache size stayed the same, which means the code was jitted.

```python
print(apply_model._cache_size())

for batch in train_data_loader:
  batch_images, batch_labels = batch
  grads, loss, accuracy = apply_model(model_state, batch_images, batch_labels)
  print(apply_model._cache_size())
```

```console
0
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
1
2
```

## Printing jaxpr translation

```python
for batch in train_data_loader:
  batch_images, batch_labels = batch
  grads, loss, accuracy = apply_model(model_state, batch_images, batch_labels)
  print(jax.make_jaxpr(apply_model)(model_state, batch_images, batch_labels))
  break
```

```console
{ lambda ; a:i32[] b:f32[10] c:f32[2,10] d:f32[2] e:f32[10,2] f:i32[] g:f32[10] h:f32[2,10]
    i:f32[2] j:f32[10,2] k:f32[10] l:f32[2,10] m:f32[2] n:f32[10,2] o:f32[128,2]
    p:i32[128]. let
    q:f32[10] r:f32[2,10] s:f32[2] t:f32[10,2] u:f32[] v:f32[] = pjit[
      name=apply_model
      jaxpr={ lambda ; w:i32[] x:f32[10] y:f32[2,10] z:f32[2] ba:f32[10,2] bb:i32[]
          bc:f32[10] bd:f32[2,10] be:f32[2] bf:f32[10,2] bg:f32[10] bh:f32[2,10]
          bi:f32[2] bj:f32[10,2] bk:f32[128,2] bl:i32[128]. let
          bm:f32[128,10] = dot_general[dimension_numbers=(([1], [0]), ([], []))] bk
            y
          bn:f32[1,10] = reshape[dimensions=None new_sizes=(1, 10)] x
          bo:f32[128,10] = add bm bn
          bp:f32[128,10] = custom_jvp_call[
            call_jaxpr={ lambda ; bq:f32[128,10]. let
                br:f32[128,10] = pjit[
                  name=relu
                  jaxpr={ lambda ; bs:f32[128,10]. let
                      bt:f32[128,10] = max bs 0.0
                    in (bt,) }
                ] bq
              in (br,) }
            jvp_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x7ac6181d0af0>
            num_consts=0
            symbolic_zeros=False
          ] bo
          bu:bool[128,10] = gt bo 0.0
          _:f32[128,10] = broadcast_in_dim[
            broadcast_dimensions=()
            shape=(128, 10)
          ] 0.0
          bv:f32[128,2] = dot_general[dimension_numbers=(([1], [0]), ([], []))] bp
            ba
          bw:f32[1,2] = reshape[dimensions=None new_sizes=(1, 2)] z
          bx:f32[128,2] = add bv bw
          by:f32[128,2] bz:f32[128,2] ca:f32[128,1] = pjit[
            name=log_softmax
            jaxpr={ lambda ; cb:f32[128,2]. let
                cc:f32[128] = reduce_max[axes=(1,)] cb
                cd:f32[128,1] = reshape[dimensions=None new_sizes=(128, 1)] cc
                ce:bool[128,2] = eq cb cd
                cf:f32[128,2] = convert_element_type[
                  new_dtype=float32
                  weak_type=False
                ] ce
                _:f32[128] = reduce_sum[axes=(1,)] cf
                cg:f32[128,1] = broadcast_in_dim[
                  broadcast_dimensions=(0,)
                  shape=(128, 1)
                ] cc
                ch:f32[128,1] = stop_gradient cg
                ci:f32[128,2] = sub cb ch
                cj:f32[128,2] = exp ci
                ck:f32[128] = reduce_sum[axes=(1,)] cj
                cl:f32[128,1] = broadcast_in_dim[
                  broadcast_dimensions=(0,)
                  shape=(128, 1)
                ] ck
                cm:f32[128,1] = log cl
                cn:f32[128,2] = sub ci cm
              in (cn, cj, cl) }
          ] bx
          co:f32[128,2] = pjit[
            name=_one_hot
            jaxpr={ lambda ; cp:i32[128]. let
                cq:i32[128,1] = broadcast_in_dim[
                  broadcast_dimensions=(0,)
                  shape=(128, 1)
                ] cp
                cr:i32[1,2] = iota[dimension=1 dtype=int32 shape=(1, 2)] 
                cs:bool[128,2] = eq cq cr
                ct:f32[128,2] = convert_element_type[
                  new_dtype=float32
                  weak_type=False
                ] cs
              in (ct,) }
          ] bl
          cu:f32[128,2] cv:f32[128,2] cw:f32[128,1] = pjit[
            name=log_softmax
            jaxpr={ lambda ; cx:f32[128,2]. let
                cy:f32[128] = reduce_max[axes=(1,)] cx
                cz:f32[128,1] = reshape[dimensions=None new_sizes=(128, 1)] cy
                da:bool[128,2] = eq cx cz
                db:f32[128,2] = convert_element_type[
                  new_dtype=float32
                  weak_type=False
                ] da
                _:f32[128] = reduce_sum[axes=(1,)] db
                dc:f32[128,1] = broadcast_in_dim[
                  broadcast_dimensions=(0,)
                  shape=(128, 1)
                ] cy
                dd:f32[128,1] = stop_gradient dc
                de:f32[128,2] = sub cx dd
                df:f32[128,2] = exp de
                dg:f32[128] = reduce_sum[axes=(1,)] df
                dh:f32[128,1] = broadcast_in_dim[
                  broadcast_dimensions=(0,)
                  shape=(128, 1)
                ] dg
                di:f32[128,1] = log dh
                dj:f32[128,2] = sub de di
              in (dj, df, dh) }
          ] by
          dk:f32[128,2] = mul co cu
          dl:f32[128] = reduce_sum[axes=(1,)] dk
          dm:f32[128] = neg dl
          dn:f32[] = reduce_sum[axes=(0,)] dm
          do:f32[] = div dn 128.0
          dp:f32[] = div 1.0 128.0
          dq:f32[128] = broadcast_in_dim[broadcast_dimensions=() shape=(128,)] dp
          dr:f32[128] = neg dq
          ds:f32[128,2] = broadcast_in_dim[
            broadcast_dimensions=(0,)
            shape=(128, 2)
          ] dr
          dt:f32[128,2] = mul co ds
          du:f32[128,2] = pjit[
            name=log_softmax
            jaxpr={ lambda ; dv:f32[128,2] dw:f32[128,1] dx:f32[128,2]. let
                dy:f32[128,2] = neg dx
                dz:f32[128] = reduce_sum[axes=(1,)] dy
                ea:f32[128,1] = reshape[dimensions=None new_sizes=(128, 1)] dz
                eb:f32[128,1] = div ea dw
                ec:f32[128] = reduce_sum[axes=(1,)] eb
                ed:f32[128,2] = broadcast_in_dim[
                  broadcast_dimensions=(0,)
                  shape=(128, 2)
                ] ec
                ee:f32[128,2] = mul ed dv
                ef:f32[128,2] = add_any dx ee
              in (ef,) }
          ] cv cw dt
          eg:f32[128,2] = pjit[
            name=log_softmax
            jaxpr={ lambda ; eh:f32[128,2] ei:f32[128,1] ej:f32[128,2]. let
                ek:f32[128,2] = neg ej
                el:f32[128] = reduce_sum[axes=(1,)] ek
                em:f32[128,1] = reshape[dimensions=None new_sizes=(128, 1)] el
                en:f32[128,1] = div em ei
                eo:f32[128] = reduce_sum[axes=(1,)] en
                ep:f32[128,2] = broadcast_in_dim[
                  broadcast_dimensions=(0,)
                  shape=(128, 2)
                ] eo
                eq:f32[128,2] = mul ep eh
                er:f32[128,2] = add_any ej eq
              in (er,) }
          ] bz ca du
          es:f32[2] = reduce_sum[axes=(0,)] eg
          et:f32[1,2] = reshape[dimensions=None new_sizes=(1, 2)] es
          eu:f32[2] = reshape[dimensions=None new_sizes=(2,)] et
          ev:f32[2,10] = dot_general[dimension_numbers=(([0], [0]), ([], []))] eg
            bp
          ew:f32[10,2] = transpose[permutation=(1, 0)] ev
          ex:f32[128,10] = dot_general[dimension_numbers=(([1], [1]), ([], []))] eg
            ba
          ey:f32[128,10] = broadcast_in_dim[
            broadcast_dimensions=()
            shape=(128, 10)
          ] 0.0
          ez:bool[128,10] = eq bu True
          fa:f32[128,10] = select_n ez ey ex
          fb:f32[10] = reduce_sum[axes=(0,)] fa
          fc:f32[1,10] = reshape[dimensions=None new_sizes=(1, 10)] fb
          fd:f32[10] = reshape[dimensions=None new_sizes=(10,)] fc
          fe:f32[10,2] = dot_general[dimension_numbers=(([0], [0]), ([], []))] fa
            bk
          ff:f32[2,10] = transpose[permutation=(1, 0)] fe
          fg:i32[128] = argmax[axes=(1,) index_dtype=int32] by
          fh:bool[128] = eq fg bl
          fi:i32[128] = convert_element_type[new_dtype=int32 weak_type=False] fh
          fj:f32[128] = convert_element_type[new_dtype=float32 weak_type=False] fi
          fk:f32[] = reduce_sum[axes=(0,)] fj
          fl:f32[] = div fk 128.0
        in (fd, ff, eu, ew, do, fl) }
    ] a b c d e f g h i j k l m n o p
  in (q, r, s, t, u, v) }
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
