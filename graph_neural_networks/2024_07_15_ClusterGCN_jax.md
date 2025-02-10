# Cluster-GCN Sampler in JAX

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/8mknbxIIf94" frameborder="0" allowfullscreen></iframe>
</div>


## Acknowledgment:
I borrowed some code from [Introduction to Graph Neural Nets with JAX/jraph](https://colab.research.google.com/github/deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb#scrollTo=1n1kCuqtkvfm) and [pytorch-geometric tutorials](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html).

## References:
```bibtex
@inproceedings{clustergcn,
  title   = {Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks},
  author  = { Wei-Lin Chiang and Xuanqing Liu and Si Si and Yang Li and Samy Bengio and Cho-Jui Hsieh},
  booktitle = {ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year    = {2019},
  url     = {https://arxiv.org/pdf/1905.07953.pdf},
}
```

```bibtex
@software{jraph2020github,
  author  = {Jonathan Godwin* and Thomas Keck* and Peter Battaglia and Victor Bapst and Thomas Kipf and Yujia Li and Kimberly Stachenfeld and Petar Veli\v{c}kovi\'{c} and Alvaro Sanchez-Gonzalez},
  title   = {Jraph: A library for graph neural networks in jax.},
  url     = {http://github.com/deepmind/jraph},
  version = {0.0.1.dev},
  year    = {2020},
}
```

## Graph sampling using scipy sparse matrices

![image](https://github.com/user-attachments/assets/d7a1d0fb-318d-47bc-af84-68eac90cdb88)

![image](https://github.com/user-attachments/assets/27ba9276-ec1b-402d-bdd9-fc1615c84318)

![image](https://github.com/user-attachments/assets/01abcb16-90ea-484f-892d-2f28e7abde6f)

![image](https://github.com/user-attachments/assets/74292e2e-dddd-4241-a49a-d23012a9edf2)

![image](https://github.com/user-attachments/assets/25df9bc5-08fb-4990-a6da-2f34ce4c8622)

## Install METIS

```python
# credit to this answer on stackoverflow
# https://stackoverflow.com/questions/73860660/metis-installation-on-google-colab

import requests
import tarfile

# Download and extract the file from the official website
url = "http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz"
response = requests.get(url, stream=True)
file = tarfile.open(fileobj=response.raw, mode="r|gz")
file.extractall(path=".")

# steps to install metis
%cd metis-5.1.0
!make config shared=1 prefix=~/.local/
!make install
!cp ~/.local/lib/libmetis.so /usr/lib/libmetis.so
!export METIS_DLL=/usr/lib/libmetis.so
!pip3 install metis-python

import metispy as metis
```

## Import libraries

```bash
# install pytorch_geometric
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git

# install jraph
!pip install git+https://github.com/deepmind/jraph.git
```

```python
# Standard libraries
import numpy as np
import scipy as sp
import seaborn as sns
import pandas as pd
import random
import itertools
from typing import Any, NamedTuple, Iterable, Mapping, Union, Optional

# Plotting libraries
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx

# jax
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import flax
import flax.linen as nn
from flax.training import train_state
import optax
import pickle

# jraph
import jraph
from jraph._src import models as jraph_models

# PyTorch geometric
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
```

```python
random_seed = 42
random.seed(random_seed)
key = jax.random.PRNGKey(random_seed)
torch_geometric.seed_everything(random_seed)
plt.style.use('dark_background')
colors = cm.tab10.colors
accuracy_list = []
```

## Create A Toy Graph

```python
edges = jax.random.randint(key, (150,2), 0, 50)
features = jax.random.uniform(key, shape=(50,2))
```

```python
adj = sp.sparse.csr_matrix((np.ones((edges.shape[0]), dtype=np.float32), (edges[:, 0], edges[:, 1])), shape=(features.shape[0], features.shape[0]))
adj += adj.transpose()
adj_lil = adj.tolil()

adj_lists = [[] for _ in range(features.shape[0])]
print(adj_lists)
for i in range(features.shape[0]):
  rows = adj_lil[i].rows[0]
  # self-edge needs to be removed for valid format of METIS
  if i in rows:
    rows.remove(i)
  adj_lists[i] = rows

print(adj_lists)
```

```console
[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
[[2, 20, 21, 22, 26, 46], [6, 7, 14, 15, 16, 31, 48], [0, 13, 14, 20, 23, 24, 33, 35], [24, 32, 34, 40, 45], [5, 9, 14, 15, 32, 33, 41, 46, 49], [4, 21, 27, 37, 39], [1, 7, 11, 15, 16, 24, 28, 30, 37, 42], [1, 6, 22, 40, 42], [18, 22, 25, 29, 46], [4, 11, 14, 44], [32], [6, 9, 21, 26, 33, 45], [20, 22, 33], [2, 41, 42, 46], [1, 2, 4, 9, 21, 24, 29, 30, 38, 45], [1, 4, 6, 19, 36, 37, 40, 44], [1, 6, 19, 32, 36, 37], [20, 21, 22, 24, 31, 36, 49], [8, 37], [15, 16, 26, 33, 38], [0, 2, 12, 17, 26, 36], [0, 5, 11, 14, 17, 25, 28, 43, 49], [0, 7, 8, 12, 17, 28, 40, 48, 49], [2, 30, 34, 39, 42], [2, 3, 6, 14, 17, 34], [8, 21, 29], [0, 11, 19, 20, 34], [5, 41, 43], [6, 21, 22, 36], [8, 14, 25], [6, 14, 23, 38, 40, 44], [1, 17, 33, 34, 36, 39, 42, 48], [3, 4, 10, 16, 36], [2, 4, 11, 12, 19, 31, 36], [3, 23, 24, 26, 31, 36], [2, 36, 40, 44], [15, 16, 17, 20, 28, 31, 32, 33, 34, 35, 42, 45], [5, 6, 15, 16, 18, 41], [14, 19, 30, 44], [5, 23, 31, 40, 44], [3, 7, 15, 22, 30, 35, 39, 43], [4, 13, 27, 37], [6, 7, 13, 23, 31, 36], [21, 27, 40], [9, 15, 30, 35, 38, 39, 47], [3, 11, 14, 36], [0, 4, 8, 13], [44], [1, 22, 31, 49], [4, 17, 21, 22, 48]
```

```python
_, groups = metis.part_graph(adj_lists, 5, seed=random_seed)
nodes_colors = np.array(colors)[groups]

G = nx.from_scipy_sparse_array(adj, create_using=nx.DiGraph)
plt.figure(figsize=(8,8))
plt.axis('off')
nx.draw_networkx_nodes(G, pos=features, node_size=200, node_color=nodes_colors)
nx.draw_networkx_edges(G, pos=features, edge_color="grey", alpha=0.7)
plt.show()
```

![image](https://github.com/user-attachments/assets/658c3b72-e24c-4160-b2f5-6592838954bb)

## `partition_graph` Function

```python
GraphsInfo = NamedTuple('GraphsInfo', [('global_idx', int), ('train_mask', bool), ('val_mask', bool), ('test_mask', bool), ('y', int), ('one_hot_labels', int)])
```

```python
def partition_graph(edges, features, num_parts, num_parts_per_subgraph,
                    graph_train_mask, graph_val_mask, graph_test_mask,
                    graph_labels, one_hot_labels,):
  """partition a graph by METIS."""

  graphs_list     = []
  graphinfo_list  = []

  num_all_nodes = features.shape[0]
  adj = sp.sparse.csr_matrix((np.ones((edges.shape[0]), dtype=np.float32), (edges[:, 0], edges[:, 1])), shape=(num_all_nodes, num_all_nodes))
  adj += adj.transpose()
  adj_lil = adj.tolil()

  adj_lists = [[] for _ in range(num_all_nodes)]
  for i in range(num_all_nodes):
    rows = adj_lil[i].rows[0]
    # self-edge needs to be removed for valid format of METIS
    if i in rows:
      rows.remove(i)
    adj_lists[i] = rows


  _, groups = metis.part_graph(adj_lists, num_parts, seed=random_seed)

  part_row   = []
  part_col   = []
  part_data  = []
  parts = [[] for _ in range(num_parts)]

  for nd_idx in range(num_all_nodes):
    gp_idx = groups[nd_idx]
    parts[gp_idx].append(nd_idx)
    for nb_idx in adj[nd_idx].indices:
      if groups[nb_idx] == gp_idx:
        part_data.append(1)
        part_row.append(nd_idx)
        part_col.append(nb_idx)

  part_data.append(0)
  part_row.append(num_all_nodes - 1)
  part_col.append(num_all_nodes - 1)
  part_adj = sp.sparse.coo_matrix((part_data, (part_row, part_col))).tocsr()

  global_idx        = []
  features_batches  = []
  adj_batches       = []
  if graph_train_mask is not None:
    graph_train_mask_batches  = []
    graph_val_mask_batches    = []
    graph_test_mask_batches   = []
    graph_labels_batches      = []
    one_hot_labels_batches    = []

  for i in range(0, len(parts), num_parts_per_subgraph):
    if len(parts) - i < num_parts_per_subgraph:
      continue
    parts_merged = list(itertools.chain(*parts[i:i + num_parts_per_subgraph]))
    global_idx.append(parts_merged)
    features_batches.append(features[parts_merged,:])
    adj_batches.append(part_adj[parts_merged, :][:, parts_merged])

    if graph_train_mask is not None:
      graph_train_mask_batches.append(graph_train_mask[jnp.asarray(parts_merged, dtype=jnp.int32)])
      graph_val_mask_batches.append(graph_val_mask[jnp.asarray(parts_merged, dtype=jnp.int32)])
      graph_test_mask_batches.append(graph_test_mask[jnp.asarray(parts_merged, dtype=jnp.int32)])
      graph_labels_batches.append(graph_labels[jnp.asarray(parts_merged, dtype=jnp.int32)])
      one_hot_labels_batches.append(one_hot_labels[jnp.asarray(parts_merged, dtype=jnp.int32),:])

  for i in range(len(adj_batches)):
    graph = jraph.GraphsTuple(
        n_node=jnp.asarray([features_batches[i].shape[0]]),
        n_edge=jnp.asarray([adj_batches[i].nnz]),
        nodes=features_batches[i],
        edges=None,
        globals=None,
        senders=jnp.asarray(adj_batches[i].tocoo().row),
        receivers=jnp.asarray(adj_batches[i].tocoo().col))
    graphs_list.append(graph)

    if graph_train_mask is not None:
      graphinfo = GraphsInfo(global_idx=global_idx[i],
                             train_mask=graph_train_mask_batches[i],
                             val_mask=graph_val_mask_batches[i],
                             test_mask=graph_test_mask_batches[i],
                             y=graph_labels_batches[i],
                             one_hot_labels=one_hot_labels_batches[i])
      graphinfo_list.append(graphinfo)
    else:
      graphinfo = GraphsInfo(global_idx=global_idx[i],
                             train_mask=None,
                             val_mask=None,
                             test_mask=None,
                             y=None,
                             one_hot_labels=None)
      graphinfo_list.append(graphinfo)


  print(f'Number of nodes:          {num_all_nodes}')
  print(f'Number of edges:          {adj.nnz}')
  print(f'Number of edges part:     {part_adj.nnz}')
  print(f'Number of partitions:     {len(parts)}')
  print(f'Number of batches:        {len(adj_batches)}')

  return graphs_list, graphinfo_list
```

## Test `partition_graph` function with `num_parts=5` and `num_parts_per_subgraph=2`

```python
graphs_list, _ = partition_graph(edges=edges, features=features,
                              num_parts=5, num_parts_per_subgraph=2,
                              graph_train_mask=None, graph_val_mask=None,
                              graph_test_mask=None, graph_labels=None,
                              one_hot_labels=None,)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout()
axs = axs.flatten()

for i, graph in enumerate(graphs_list):

  A_sample = sp.sparse.csr_array((jnp.ones((graph.senders.shape[0],1)).squeeze(),
                          (graph.senders, graph.receivers)),
                          shape=(graph.nodes.shape[0], graph.nodes.shape[0]))
  G_sample = nx.from_scipy_sparse_array(A_sample, create_using=nx.DiGraph)
  node_pos=graph.nodes
  nx.draw_networkx_nodes(G_sample, pos=node_pos, node_size=100, node_color='#1f77b4', ax = axs[i])
  nx.draw_networkx_edges(G_sample, pos=node_pos, edge_color="white", alpha=0.7, ax=axs[i])
  axs[i].scatter(features[:,0], features[:,1], marker='o', s=100, alpha=0.3, c='white', linewidth=0)

plt.show()
```

```console
Number of nodes:          50
Number of edges:          280
Number of edges part:     151
Number of partitions:     5
Number of batches:        2
```

![image](https://github.com/user-attachments/assets/5156beeb-ae04-43b1-a11c-d35958c7d7db)

## Test `partition_graph` function with `num_parts=10` and `num_parts_per_subgraph=2`

```python
graphs_list, _ = partition_graph(edges=edges, features=features,
                              num_parts=10, num_parts_per_subgraph=2,
                              graph_train_mask=None, graph_val_mask=None,
                              graph_test_mask=None, graph_labels=None,
                              one_hot_labels=None,)

fig, axs = plt.subplots(1, 5, figsize=(15, 3))
fig.tight_layout()
axs = axs.flatten()

for i, graph in enumerate(graphs_list):

  A_sample = sp.sparse.csr_array((jnp.ones((graph.senders.shape[0],1)).squeeze(),
                          (graph.senders, graph.receivers)),
                          shape=(graph.nodes.shape[0], graph.nodes.shape[0]))
  G_sample = nx.from_scipy_sparse_array(A_sample, create_using=nx.DiGraph)
  node_pos=graph.nodes
  nx.draw_networkx_nodes(G_sample, pos=node_pos, node_size=100, node_color='#1f77b4', ax = axs[i])
  nx.draw_networkx_edges(G_sample, pos=node_pos, edge_color="white", alpha=0.7, ax=axs[i])
  axs[i].scatter(features[:,0], features[:,1], marker='o', s=100, alpha=0.3, c='white', linewidth=0)

plt.show()
```

```console
Number of nodes:          50
Number of edges:          280
Number of edges part:     87
Number of partitions:     10
Number of batches:        5
```

![image](https://github.com/user-attachments/assets/b0147344-90ed-4b03-9d9c-ba59ec64103d)

## Import Cora Dataset

```python
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
num_features = dataset.num_features
num_classes = dataset.num_classes
data_Cora = dataset[0]  # Get the first graph object.
data_Cora
```

```console
Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.x
Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.tx
Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.allx
Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.y
Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.ty
Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.ally
Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.graph
Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.test.index
Processing...
Done!
Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
```

```python
# some statistics about the graph.
print(f'Number of nodes:          {data_Cora.num_nodes}')
print(f'Number of edges:          {data_Cora.num_edges}')
print(f'Average node degree:      {data_Cora.num_edges / data_Cora.num_nodes:.2f}')
print(f'Number of training nodes: {data_Cora.train_mask.sum()}')
print(f'Training node label rate: {int(data_Cora.train_mask.sum()) / data_Cora.num_nodes:.3f}')
print(f'Has isolated nodes:       {data_Cora.has_isolated_nodes()}')
print(f'Has self-loops:           {data_Cora.has_self_loops()}')
print(f'Is undirected:            {data_Cora.is_undirected()}')
```

```console
Number of nodes:          2708
Number of edges:          10556
Average node degree:      3.90
Number of training nodes: 140
Training node label rate: 0.052
Has isolated nodes:       False
Has self-loops:           False
Is undirected:            True
```

```python
graph = jraph.GraphsTuple(
      n_node=jnp.asarray([data_Cora.x.shape[0]]),
      n_edge=jnp.asarray([data_Cora.edge_index.shape[1]]),
      nodes=jnp.asarray(data_Cora.x),
      # No edge features.
      edges=None,
      globals=None,
      senders=jnp.asarray([data_Cora.edge_index[0,:]]).squeeze(),
      receivers=jnp.asarray([data_Cora.edge_index[1,:]]).squeeze())

graph_train_mask = jnp.asarray([data_Cora.train_mask]).squeeze()
graph_val_mask = jnp.asarray([data_Cora.val_mask]).squeeze()
graph_test_mask = jnp.asarray([data_Cora.test_mask]).squeeze()
graph_labels = jnp.asarray([data_Cora.y]).squeeze()
one_hot_labels = jax.nn.one_hot(graph_labels, len(jnp.unique(graph_labels)))
```

```python
print(f'Number of nodes: {graph.n_node[0]}')
print(f'Number of edges: {graph.n_edge[0]}')
print(f'Feature matrix data type: {graph.nodes.dtype}')
print(f'senders list data type:   {graph.senders.dtype}')
print(f'receivers list data type: {graph.receivers.dtype}')
print(f'Labels matrix data type:  {graph_labels.dtype}')
```

```console
Number of nodes: 2708
Number of edges: 10556
Feature matrix data type: float32
senders list data type:   int32
receivers list data type: int32
Labels matrix data type:  int32
```

## GCN Layers from Jraph

```python
# Functions must be passed to jraph GNNs, but pytype does not recognise
# linen Modules as callables to here we wrap in a function.
def make_embed_fn(latent_size):
  def embed(inputs):
    return nn.Dense(latent_size)(inputs)
  return embed

def _attention_logit_fn(sender_attr: jnp.ndarray,
                        receiver_attr: jnp.ndarray,
                        edges: jnp.ndarray) -> jnp.ndarray:
  del edges
  x = jnp.concatenate((sender_attr, receiver_attr), axis=1)
  return nn.Dense(1)(x)
```

```python
class GCN(nn.Module):
  """Defines a GAT network using FLAX

  Args:
    graph: GraphsTuple the network processes.

  Returns:
    output graph with updated node values.
  """
  gcn1_output_dim: int
  output_dim: int

  @nn.compact
  def __call__(self, x):
    gcn1 = jraph.GraphConvolution(update_node_fn=lambda n: jax.nn.relu(make_embed_fn(self.gcn1_output_dim)(n)),
                          add_self_edges=True)
    gcn2 = jraph.GraphConvolution(update_node_fn=nn.Dense(self.output_dim))
    return gcn2(gcn1(x))
```

```python
model = GCN(8, len(jnp.unique(graph_labels)))
model
```

```console
GCN(
    # attributes
    gcn1_output_dim = 8
    output_dim = 7
)
```

## Optimizer and Loss
We set the optimizer to adam using `optax` library. Then we initialized the model using random parameters.

```python
optimizer = optax.adam(learning_rate=0.01)

rng, inp_rng, init_rng = jax.random.split(jax.random.PRNGKey(random_seed), 3)
params = model.init(jax.random.PRNGKey(random_seed),graph)

model_state = train_state.TrainState.create(apply_fn=model.apply,
                                            params=params,
                                            tx=optimizer)
```

```python
def compute_loss(state, params, graph, labels, one_hot_labels, mask):
  """Computes loss."""
  pred_graph = state.apply_fn(params, graph)
  preds = jax.nn.log_softmax(pred_graph.nodes)
  loss = optax.softmax_cross_entropy(preds, one_hot_labels)
  loss_mask = jnp.sum(jnp.where(mask, loss, 0)) / jnp.sum(mask)

  pred_labels = jnp.argmax(preds, axis=1)
  acc = (pred_labels == labels)
  acc_mask = jnp.sum(jnp.where(mask, acc, 0)) / jnp.sum(mask)
  return loss_mask, acc_mask
```

## Training on a full batch

```python
@jax.jit  # Jit the function for efficiency
def train_step(state, graph, graph_labels, one_hot_labels, train_mask):
  # Gradient function
  grad_fn = jax.value_and_grad(compute_loss,  # Function to calculate the loss
                                argnums=1,  # Parameters are second argument of the function
                                has_aux=True  # Function has additional outputs, here accuracy
                              )
  # Determine gradients for current model, parameters and batch
  (loss, acc), grads = grad_fn(state, state.params, graph, graph_labels, one_hot_labels, train_mask)
  # Perform parameter update with gradients and optimizer
  state = state.apply_gradients(grads=grads)
  # Return state and any other value we might want
  return state, loss, acc
```

```python
def train_model(state, graph, graph_labels, one_hot_labels, train_mask, val_mask, num_epochs):
  # Training loop
  for epoch in range(num_epochs):
    state, loss, acc = train_step(state, graph, graph_labels, one_hot_labels, train_mask)
    print(f'step: {epoch:03d}, train loss: {loss:.4f}, train acc: {acc:.4f}')
  return state, acc
```

```python
trained_model_state, train_acc = train_model(model_state, graph, graph_labels, one_hot_labels, graph_train_mask, graph_val_mask, num_epochs=200)
accuracy_list.append(['full batch', 'train', float(train_acc)])
```

```console
step: 000, train loss: 1.9460, train acc: 0.1357
step: 001, train loss: 1.9413, train acc: 0.2429
step: 002, train loss: 1.9360, train acc: 0.2786
step: 003, train loss: 1.9289, train acc: 0.2786
step: 004, train loss: 1.9208, train acc: 0.2286
step: 005, train loss: 1.9126, train acc: 0.1786
step: 006, train loss: 1.9045, train acc: 0.1857
step: 007, train loss: 1.8962, train acc: 0.2286
step: 008, train loss: 1.8875, train acc: 0.2786
step: 009, train loss: 1.8783, train acc: 0.2929
step: 010, train loss: 1.8685, train acc: 0.3071
...
...
...
step: 191, train loss: 0.0702, train acc: 1.0000
step: 192, train loss: 0.0693, train acc: 1.0000
step: 193, train loss: 0.0683, train acc: 1.0000
step: 194, train loss: 0.0674, train acc: 1.0000
step: 195, train loss: 0.0665, train acc: 1.0000
step: 196, train loss: 0.0656, train acc: 1.0000
step: 197, train loss: 0.0647, train acc: 1.0000
step: 198, train loss: 0.0639, train acc: 1.0000
step: 199, train loss: 0.0630, train acc: 1.0000
```

```python
test_loss, test_acc = compute_loss(trained_model_state, trained_model_state.params, graph, graph_labels, one_hot_labels, graph_test_mask)
print(f'test loss: {test_loss:.4f}, test acc: {test_acc:.4f}')
accuracy_list.append(['full batch', 'test', float(test_acc)])
```

```console
test loss: 0.7673, test acc: 0.7660
```

## Creating mini batches

```python
graphs_list, graphinfo_list = partition_graph(edges=jnp.concatenate((jnp.expand_dims(graph.senders,0), jnp.expand_dims(graph.receivers,0)), axis=0).T,
                                              features=graph.nodes, num_parts=32,
                                              num_parts_per_subgraph=8, graph_train_mask=graph_train_mask,
                                              graph_val_mask=graph_val_mask, graph_test_mask=graph_test_mask,
                                              graph_labels=graph_labels, one_hot_labels=one_hot_labels)
```

```console
Number of nodes:          2708
Number of edges:          10556
Number of edges part:     8549
Number of partitions:     32
Number of batches:        4
```

```python
for i, subgraph in enumerate(graphs_list):
  print(f'Subgraph: {i:02d}, feature matrix: {subgraph.nodes.shape}, senders: {subgraph.senders.shape}, receivers: {subgraph.receivers.shape}')
```

```console
Subgraph: 00, feature matrix: (675, 1433), senders: (2165,), receivers: (2165,)
Subgraph: 01, feature matrix: (680, 1433), senders: (2282,), receivers: (2282,)
Subgraph: 02, feature matrix: (680, 1433), senders: (2202,), receivers: (2202,)
Subgraph: 03, feature matrix: (673, 1433), senders: (1900,), receivers: (1900,)
```

## Make sure everything works as expected

```python
def test_graph_partition(graph, graphinfo_list, batch_id, test_node_local_id):
  test_node_global_id = graphinfo_list[batch_id].global_idx[test_node_local_id]
  print(f'batch id: {batch_id}, test node local id: {test_node_local_id}, test node global id: {test_node_global_id}')
  print('test node in global edge index:')
  print(graph.senders[(graph.senders == test_node_global_id)])
  print(graph.receivers[(graph.senders == test_node_global_id)])
  local_edge_index = jnp.vstack((graphs_list[batch_id].senders, graphs_list[batch_id].receivers))
  print('test node in local edge index:')
  print(local_edge_index[:, (local_edge_index[0, :] == test_node_local_id)])
  local_edge_index_receivers = local_edge_index[1, (local_edge_index[0, :] == test_node_local_id)]
  print('receivers in global edge index:')
  for i in local_edge_index_receivers:
    print(graphinfo_list[batch_id].global_idx[i], end=' ')

  print()
  print('test node global features:')
  print(graph.nodes[test_node_global_id,:10])
  print('test node local features:')
  print(graphs_list[batch_id].nodes[test_node_local_id,:10])

  print(f'test node globals: label= {graph_labels[test_node_global_id]}, train= {graph_train_mask[test_node_global_id]}')
  print(f'test node locals:  label= {graphinfo_list[batch_id].y[test_node_local_id]}, train= {graphinfo_list[batch_id].train_mask[test_node_local_id]} ')

test_graph_partition(graph, graphinfo_list, batch_id=0, test_node_local_id=0)
print()
test_graph_partition(graph, graphinfo_list, batch_id=1, test_node_local_id=300)
print()
test_graph_partition(graph, graphinfo_list, batch_id=2, test_node_local_id=200)
print()
test_graph_partition(graph, graphinfo_list, batch_id=3, test_node_local_id=400)
```

```console
batch id: 0, test node local id: 0, test node global id: 8
test node in global edge index:
[8 8 8]
[ 269  281 1996]
test node in local edge index:
[[ 0  0  0]
 [ 3  4 54]]
receivers in global edge index:
269 281 1996 
test node global features:
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
test node local features:
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
test node globals: label= 3, train= True
test node locals:  label= 3, train= True 

batch id: 1, test node local id: 300, test node global id: 1547
test node in global edge index:
[1547]
[2169]
test node in local edge index:
[[300]
 [321]]
receivers in global edge index:
2169 
test node global features:
[0.         0.         0.04761905 0.         0.         0.
 0.         0.         0.         0.        ]
test node local features:
[0.         0.         0.04761905 0.         0.         0.
 0.         0.         0.         0.        ]
test node globals: label= 1, train= False
test node locals:  label= 1, train= False 

batch id: 2, test node local id: 200, test node global id: 890
test node in global edge index:
[890 890]
[1269 1314]
test node in local edge index:
[[200 200]
 [207 210]]
receivers in global edge index:
1269 1314 
test node global features:
[0.         0.         0.         0.         0.05882353 0.
 0.         0.         0.         0.        ]
test node local features:
[0.         0.         0.         0.         0.05882353 0.
 0.         0.         0.         0.        ]
test node globals: label= 5, train= False
test node locals:  label= 5, train= False 

batch id: 3, test node local id: 400, test node global id: 2259
test node in global edge index:
[2259 2259 2259 2259]
[1419 1436 2335 2337]
test node in local edge index:
[[400 400 400 400]
 [373 376 402 404]]
receivers in global edge index:
1419 1436 2335 2337 
test node global features:
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
test node local features:
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
test node globals: label= 4, train= False
test node locals:  label= 4, train= False 
```

## Training on mini batches

```python
def train_model_batch(state, graphs_list, graphinfo_list, num_epochs):
  # Training loop
  for epoch in range(num_epochs):
    for i, (graph, graphinfo) in enumerate(zip(graphs_list, graphinfo_list)):
      state, loss, acc = train_step(state, graph, graphinfo.y, graphinfo.one_hot_labels, graphinfo.train_mask)

    print(f'step: {epoch:03d}, train loss: {loss:.4f}, train acc: {acc:.4f}')
  return state, acc
```

```python
trained_model_state, train_acc = train_model_batch(model_state, graphs_list, graphinfo_list, num_epochs=200)
accuracy_list.append(['mini batch', 'train', float(train_acc)])
```

```console
step: 000, train loss: 1.9539, train acc: 0.0732
step: 001, train loss: 1.9542, train acc: 0.0732
step: 002, train loss: 1.9502, train acc: 0.0732
step: 003, train loss: 1.9444, train acc: 0.0732
step: 004, train loss: 1.9365, train acc: 0.0732
step: 005, train loss: 1.9260, train acc: 0.1220
step: 006, train loss: 1.9135, train acc: 0.1220
step: 007, train loss: 1.8995, train acc: 0.1463
step: 008, train loss: 1.8838, train acc: 0.1463
step: 009, train loss: 1.8659, train acc: 0.2195
step: 010, train loss: 1.8453, train acc: 0.3659
...
...
...
step: 191, train loss: 0.0071, train acc: 1.0000
step: 192, train loss: 0.0071, train acc: 1.0000
step: 193, train loss: 0.0070, train acc: 1.0000
step: 194, train loss: 0.0069, train acc: 1.0000
step: 195, train loss: 0.0068, train acc: 1.0000
step: 196, train loss: 0.0067, train acc: 1.0000
step: 197, train loss: 0.0067, train acc: 1.0000
step: 198, train loss: 0.0066, train acc: 1.0000
step: 199, train loss: 0.0065, train acc: 1.0000
```

```python
test_loss, test_acc = compute_loss(trained_model_state, trained_model_state.params, graph, graph_labels, one_hot_labels, graph_test_mask)
print(f'test loss: {test_loss:.4f}, test acc: {test_acc:.4f}')
accuracy_list.append(['mini batch', 'test', float(test_acc)])
```

```console
test loss: 0.7264, test acc: 0.7850
```

## Plotting the results

```python
df = pd.DataFrame(accuracy_list, columns=('Method', 'Split', 'Accuracy'))
sns.barplot(df,x='Method', y='Accuracy', hue='Split', palette="muted")
plt.show()
```

![image](https://github.com/user-attachments/assets/b2a12a2e-7eef-43ba-ba41-301a5727cac2)







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
