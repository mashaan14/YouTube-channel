# A Visual Guide to GNN Sampling using PyTorch Geometric

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/y0poBC8xN1k" frameborder="0" allowfullscreen></iframe>
</div>


## Acknowledgment:
I borrowed some code from [pytorch-geometric tutorials](https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html)

## References:
```bibtex
@inproceedings{Fey/Lenssen/2019,
  title={Fast Graph Representation Learning with {PyTorch Geometric}},
  author={Fey, Matthias and Lenssen, Jan E.},
  booktitle={ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year={2019},
}
```

## Install Dependencies

```python
# install pytorch_geometric
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
```

```console
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
  Building wheel for torch-geometric (pyproject.toml) ... done
```

If you're using a CPU, check torch version:

`!python -c "import torch; print(torch.__version__)"`

the command to install the dpendencies:

`!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.1+cpu.html`

---

If you're using a GPU, check torch version:

`!python -c "import torch; print(torch.__version__)"`

and CUDA version:

`!python -c "import torch; print(torch.version.cuda)"`

the command to install the dpendencies:

`!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.1+cu121.html`

```bash
!python -c "import torch; print(torch.__version__)"
```

```console
2.4.1+cu121
```

```bash
!python -c "import torch; print(torch.version.cuda)"
```

```console
12.1
```

```bash
!pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.1+cpu.html
```

```console
Looking in links: https://data.pyg.org/whl/torch-2.2.1+cpu.html
Collecting pyg_lib
  Downloading https://data.pyg.org/whl/torch-2.2.0%2Bcpu/pyg_lib-0.4.0%2Bpt22cpu-cp310-cp310-linux_x86_64.whl (1.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.3/1.3 MB 10.2 MB/s eta 0:00:00
Collecting torch_scatter
  Downloading https://data.pyg.org/whl/torch-2.2.0%2Bcpu/torch_scatter-2.1.2%2Bpt22cpu-cp310-cp310-linux_x86_64.whl (508 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 508.1/508.1 kB 40.6 MB/s eta 0:00:00
Collecting torch_sparse
  Downloading https://data.pyg.org/whl/torch-2.2.0%2Bcpu/torch_sparse-0.6.18%2Bpt22cpu-cp310-cp310-linux_x86_64.whl (1.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 65.4 MB/s eta 0:00:00
Collecting torch_cluster
  Downloading https://data.pyg.org/whl/torch-2.2.0%2Bcpu/torch_cluster-1.6.3%2Bpt22cpu-cp310-cp310-linux_x86_64.whl (770 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 770.0/770.0 kB 26.8 MB/s eta 0:00:00
Collecting torch_spline_conv
  Downloading https://data.pyg.org/whl/torch-2.2.0%2Bcpu/torch_spline_conv-1.2.2%2Bpt22cpu-cp310-cp310-linux_x86_64.whl (213 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 213.9/213.9 kB 17.0 MB/s eta 0:00:00
Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from torch_sparse) (1.13.1)
Requirement already satisfied: numpy<2.3,>=1.22.4 in /usr/local/lib/python3.10/dist-packages (from scipy->torch_sparse) (1.26.4)
Installing collected packages: torch_spline_conv, torch_scatter, pyg_lib, torch_sparse, torch_cluster
Successfully installed pyg_lib-0.4.0+pt22cpu torch_cluster-1.6.3+pt22cpu torch_scatter-2.1.2+pt22cpu torch_sparse-0.6.18+pt22cpu torch_spline_conv-1.2.2+pt22cpu
```

## Import libraries

```python
# Standard libraries
import numpy as np
from scipy import sparse
import seaborn as sns
import pandas as pd

# Plotting libraries
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm
from IPython.display import Javascript  # Restrict height of output cell.

# PyTorch
import torch
import torch.nn.functional as F
from torch.nn import Linear

# import pyg_lib
# import torch_sparse

# PyTorch geometric
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import StochasticBlockModelDataset
from torch_geometric.utils import to_networkx
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader
from torch_geometric.loader import GraphSAINTNodeSampler, GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from torch_geometric import seed_everything
```

```python
random_seed = 42
torch.manual_seed(1234567)
seed_everything(42)
plt.style.use('dark_background')
```

```python
num_nodes_per_class = 500
num_nodes = [num_nodes_per_class] * 3
edge_probs = [[0.1, 0.05, 0.02],
              [0.05, 0.1, 0.02],
              [0.02, 0.02, 0.1]]
dataset = StochasticBlockModelDataset('/content', num_nodes, edge_probs, num_channels=10)
```

```python
colors = cm.tab10.colors
y_colors = np.array(colors)[dataset[0].y.numpy()]
```

```python
G = to_networkx(dataset[0], to_undirected=True)
node_pos=nx.spring_layout(G, seed=0)
plt.figure(figsize=(8,8))
plt.axis('off')
nx.draw_networkx_nodes(G,
                pos=node_pos,
                node_size=200,
                node_color=y_colors,
                alpha=0.9
                )
nx.draw_networkx_edges(G,
                pos=node_pos,
                edge_color="grey",
                alpha=0.2
                )
plt.show()
```

![image](https://github.com/user-attachments/assets/61ddd997-f8b0-44ab-95b5-d21f2b307ecc)

![image](https://github.com/user-attachments/assets/13bb392f-ecfd-4e1f-828a-267ccc5d4000)

## `NeighborLoader`

`NeighborLoader` produces subgraphs $G_s$ sampled from the original graph $G$. The number of subgraphs is determined by:
- batch size: the number of seed nodes (first nodes in the batch)
- the number of nodes in $G$

$\text{number of subgraphs} = \frac{\text{number of nodes}}{\text{batch size (the number of seed nodes)}}$

```python
BATCH_SIZE = 128
loader_neighbor_128 = NeighborLoader(dataset[0], num_neighbors=[10, 10], batch_size=BATCH_SIZE)
print(f'number of nodes / batch size:         {dataset[0].x.shape[0]} / {BATCH_SIZE} = {dataset[0].x.shape[0]/BATCH_SIZE}')
```

```console
number of nodes / batch size:         1500 / 128 = 11.71875
```

This `NeighborLoader` has 12 subgraphs.

```python
for i, s in enumerate(loader_neighbor_128):
  print(f'Subgraph: {i:02d}, feature matrix: {s.x.shape}, edges list: {s.edge_index.shape}')
```

```console
Subgraph: 00, feature matrix: torch.Size([1484, 10]), edges list: torch.Size([2, 8300])
Subgraph: 01, feature matrix: torch.Size([1490, 10]), edges list: torch.Size([2, 8350])
Subgraph: 02, feature matrix: torch.Size([1483, 10]), edges list: torch.Size([2, 8240])
Subgraph: 03, feature matrix: torch.Size([1483, 10]), edges list: torch.Size([2, 8430])
Subgraph: 04, feature matrix: torch.Size([1480, 10]), edges list: torch.Size([2, 8140])
Subgraph: 05, feature matrix: torch.Size([1489, 10]), edges list: torch.Size([2, 8210])
Subgraph: 06, feature matrix: torch.Size([1483, 10]), edges list: torch.Size([2, 8180])
Subgraph: 07, feature matrix: torch.Size([1495, 10]), edges list: torch.Size([2, 8760])
Subgraph: 08, feature matrix: torch.Size([1478, 10]), edges list: torch.Size([2, 7420])
Subgraph: 09, feature matrix: torch.Size([1478, 10]), edges list: torch.Size([2, 7500])
Subgraph: 10, feature matrix: torch.Size([1485, 10]), edges list: torch.Size([2, 7560])
Subgraph: 11, feature matrix: torch.Size([1459, 10]), edges list: torch.Size([2, 6280])
```

Let's plot the first 3 subgraphs. Nodes in white indicate that these nodes were not sampled in this subgraph.

```python
fig, axs = plt.subplots(1, 3, figsize=(21, 7))
axs = axs.flatten()
for i in range(3):
  s = next(iter(loader_neighbor_128))
  # create an array to color all nodes in white
  sampled_graph_color = np.ones_like(y_colors)
  # use label colors for the nodes in this subgraph
  # and keep everything else in white
  sampled_graph_color[s.n_id] = y_colors[s.n_id]

  axs[i].axis('off')
  axs[i].set_title(f'Subgraph: {i:02d}')
  nx.draw_networkx_nodes(G,
                  pos=node_pos,
                  node_size=200,
                  node_color=sampled_graph_color,
                  alpha=0.6,
                  ax = axs[i]
                  )

plt.show()
```

![image](https://github.com/user-attachments/assets/ef70c56d-9dcc-4396-a58c-13078cf98cef)

### Balancing the tradeoff between batch size and number of subgraphs:
Setting the batch size to a small number increases the number of subgraphs. But each subgraph would have a smaller number of nodes because we are sampling from a smaller subset of seed nodes. Selecting a large number for the batch size leads to a smaller number of subgraphs each of which has more nodes because we are sampling from a larger subset of seed nodes.

```python
BATCH_SIZE = 32
loader_neighbor_32 = NeighborLoader(dataset[0], num_neighbors=[10, 10], batch_size=BATCH_SIZE)
print(f'number of nodes / batch size:         {dataset[0].x.shape[0]} / {BATCH_SIZE} = {dataset[0].x.shape[0]/BATCH_SIZE}')
```

```console
number of nodes / batch size:         1500 / 32 = 46.875
```

```python
fig, axs = plt.subplots(1, 3, figsize=(21, 7))
axs = axs.flatten()
for i in range(3):
  s = next(iter(loader_neighbor_128))
  # create an array to color all nodes in white
  sampled_graph_color = np.ones_like(y_colors)
  # use label colors for the nodes in this subgraph
  # and keep everything else in white
  sampled_graph_color[s.n_id] = y_colors[s.n_id]

  axs[i].axis('off')
  axs[i].set_title(f'Subgraph: {i:02d}')
  nx.draw_networkx_nodes(G,
                  pos=node_pos,
                  node_size=200,
                  node_color=sampled_graph_color,
                  alpha=0.6,
                  ax = axs[i]
                  )

plt.suptitle('first 3 subgraphs with batch_size=128')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(21, 7))
axs = axs.flatten()
for i in range(3):
  s = next(iter(loader_neighbor_32))
  # create an array to color all nodes in white
  sampled_graph_color = np.ones_like(y_colors)
  # use label colors for the nodes in this subgraph
  # and keep everything else in white
  sampled_graph_color[s.n_id] = y_colors[s.n_id]

  axs[i].axis('off')
  axs[i].set_title(f'Subgraph: {i:02d}')
  nx.draw_networkx_nodes(G,
                  pos=node_pos,
                  node_size=200,
                  node_color=sampled_graph_color,
                  alpha=0.6,
                  ax = axs[i]
                  )

plt.suptitle('first 3 subgraphs with batch_size=32')
plt.show()
```

![image](https://github.com/user-attachments/assets/9ceb6f04-ed38-40bb-ba69-e3aa8a342e1e)

![image](https://github.com/user-attachments/assets/dbd97f36-274c-4459-afcc-622d9e6d305b)


## `ClusterLoader`

In `ClusterLoader` the graph is partitioned using $METIS$ algorithm into the number of partitioned specified in the parameter `num_parts`. Then, these partitions are grouped into batches where each batch contains partitions specified in `batch_size` parameter. The number of subgraphs is determined by:
- the number of partitions: `num_parts` parameter
- the batch size: `batch_size` parameter

$\text{number of subgraphs} = \frac{\text{total number of partitions}}{\text{batch size (number of partitions in each batch)}}$

```python
NUM_PARTS = 128
BATCH_SIZE = 32
cluster_data = ClusterData(dataset[0], num_parts=NUM_PARTS)
loader_cluster_128_32 = ClusterLoader(cluster_data, batch_size=32, shuffle=True)
```

```console
Computing METIS partitioning...
Done!
```

```python
print(f'number of partitions / batch size:         {NUM_PARTS} / {BATCH_SIZE} = {NUM_PARTS/BATCH_SIZE}')
```

```console
number of nodes / batch size:         128 / 32 = 4.0
```

This `ClusterLoader` has 4 subgraphs. Note that these subgraphs are independent from each other. There is no overlap between these subgraphs like in `NeighborLoader`.

```python
for i, s in enumerate(loader_cluster_128_32):
  print(f'Subgraph: {i:02d}, feature matrix: {s.x.shape}, edges list: {s.edge_index.shape}')
```

```console
Subgraph: 00, feature matrix: torch.Size([376, 10]), edges list: torch.Size([2, 8542])
Subgraph: 01, feature matrix: torch.Size([372, 10]), edges list: torch.Size([2, 8376])
Subgraph: 02, feature matrix: torch.Size([379, 10]), edges list: torch.Size([2, 8744])
Subgraph: 03, feature matrix: torch.Size([373, 10]), edges list: torch.Size([2, 8534])
```

```python
fig, axs = plt.subplots(1, 3, figsize=(21, 7))
axs = axs.flatten()
for i in range(3):
  s = next(iter(loader_cluster_128_32))

  G = to_networkx(s, to_undirected=True)
  node_pos=nx.spring_layout(G, seed=0)

  axs[i].axis('off')
  axs[i].set_title(f'Subgraph: {i:02d}')
  nx.draw_networkx_nodes(G,
                  pos=node_pos,
                  node_size=200,
                  node_color=np.array(colors)[s.y.numpy()],
                  alpha=0.6,
                  ax = axs[i]
                  )

plt.show()
```

![image](https://github.com/user-attachments/assets/473abbc2-b5a4-409a-9b0e-5687bb267c53)

### Balancing the tradeoff between the number of partitions and the batch size:
Setting the number of partitions to a smaller number leads to larger clusters with more nodes. On the other hand, setting the batch size to a large number increases the number of nodes in a subgraph. But these nodes are not necessarily in the same cluster, because the batch size represents a group of clusters merged into one subgraph.

```python
NUM_PARTS = 256
BATCH_SIZE = 32
cluster_data = ClusterData(dataset[0], num_parts=NUM_PARTS)
loader_cluster_256_32 = ClusterLoader(cluster_data, batch_size=32, shuffle=True)
```

```console
Computing METIS partitioning...
Done!
```

```python
fig, axs = plt.subplots(1, 3, figsize=(21, 7))
axs = axs.flatten()
for i in range(3):
  s = next(iter(loader_cluster_128_32))

  G = to_networkx(s, to_undirected=True)
  node_pos=nx.spring_layout(G, seed=0)

  axs[i].axis('off')
  axs[i].set_title(f'Subgraph: {i:02d}')
  nx.draw_networkx_nodes(G,
                  pos=node_pos,
                  node_size=200,
                  node_color=np.array(colors)[s.y.numpy()],
                  alpha=0.6,
                  ax = axs[i]
                  )

plt.suptitle('first 3 subgraphs with num_parts=128 and batch_size=32')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(21, 7))
axs = axs.flatten()
for i in range(3):
  s = next(iter(loader_cluster_256_32))

  G = to_networkx(s, to_undirected=True)
  node_pos=nx.spring_layout(G, seed=0)

  axs[i].axis('off')
  axs[i].set_title(f'Subgraph: {i:02d}')
  nx.draw_networkx_nodes(G,
                  pos=node_pos,
                  node_size=200,
                  node_color=np.array(colors)[s.y.numpy()],
                  alpha=0.6,
                  ax = axs[i]
                  )

plt.suptitle('first 3 subgraphs with num_parts=256 and batch_size=32')
plt.show()
```

![image](https://github.com/user-attachments/assets/be6f7217-963f-44c8-8dad-7b90bed373a3)

![image](https://github.com/user-attachments/assets/43b66f3d-7c17-49d0-8aa2-8128f1710a6f)

![image](https://github.com/user-attachments/assets/2c718e56-4beb-47c5-8dc5-603c958d53c5)

## `GraphSAINTSampler`

`GraphSAINTSampler` samples a number of subgraphs based on the number specified in the parameter `num_steps`. The `GraphSAINT` paper presented three methods of sampling:
- Random node sample
- Random edge sampler
- Random walk based samplers

number of subgraphs = num_steps parameter

```python
loader_SAINT_256_node = GraphSAINTNodeSampler(dataset[0], batch_size=256, num_steps=4)
loader_SAINT_256_edge = GraphSAINTEdgeSampler(dataset[0], batch_size=256, num_steps=4)
loader_SAINT_256_RW = GraphSAINTRandomWalkSampler(dataset[0], batch_size=256, walk_length=2, num_steps=4)
```

```python
for i, s in enumerate(loader_SAINT_256_node):
  print(f'Subgraph node sampler: {i:02d}, feature matrix: {s.x.shape}, edges list: {s.edge_index.shape}')
for i, s in enumerate(loader_SAINT_256_edge):
  print(f'Subgraph edge sampler: {i:02d}, feature matrix: {s.x.shape}, edges list: {s.edge_index.shape}')
for i, s in enumerate(loader_SAINT_256_RW):
  print(f'Subgraph RW sampler:   {i:02d}, feature matrix: {s.x.shape}, edges list: {s.edge_index.shape}')
```

```console
Subgraph node sampler: 00, feature matrix: torch.Size([235, 10]), edges list: torch.Size([2, 2988])
Subgraph node sampler: 01, feature matrix: torch.Size([229, 10]), edges list: torch.Size([2, 2938])
Subgraph node sampler: 02, feature matrix: torch.Size([239, 10]), edges list: torch.Size([2, 3086])
Subgraph node sampler: 03, feature matrix: torch.Size([238, 10]), edges list: torch.Size([2, 3214])
Subgraph edge sampler: 00, feature matrix: torch.Size([429, 10]), edges list: torch.Size([2, 10582])
Subgraph edge sampler: 01, feature matrix: torch.Size([433, 10]), edges list: torch.Size([2, 11134])
Subgraph edge sampler: 02, feature matrix: torch.Size([428, 10]), edges list: torch.Size([2, 10548])
Subgraph edge sampler: 03, feature matrix: torch.Size([446, 10]), edges list: torch.Size([2, 11376])
Subgraph RW sampler:   00, feature matrix: torch.Size([602, 10]), edges list: torch.Size([2, 19842])
Subgraph RW sampler:   01, feature matrix: torch.Size([598, 10]), edges list: torch.Size([2, 19226])
Subgraph RW sampler:   02, feature matrix: torch.Size([610, 10]), edges list: torch.Size([2, 21084])
Subgraph RW sampler:   03, feature matrix: torch.Size([598, 10]), edges list: torch.Size([2, 19862])
```

```python
fig, axs = plt.subplots(1, 4, figsize=(24, 6))
axs = axs.flatten()
for i in range(4):
  s = next(iter(loader_SAINT_256_node))

  G = to_networkx(s, to_undirected=True)
  node_pos=nx.spring_layout(G, seed=0)

  axs[i].axis('off')
  axs[i].set_title(f'Subgraph: {i:02d}')
  nx.draw_networkx_nodes(G,
                  pos=node_pos,
                  node_size=200,
                  node_color=np.array(colors)[s.y.numpy()],
                  alpha=0.6,
                  ax = axs[i]
                  )
plt.suptitle('Subgraphs using GraphSAINT Node Sampler')
plt.show()

fig, axs = plt.subplots(1, 4, figsize=(24, 6))
axs = axs.flatten()
for i in range(4):
  s = next(iter(loader_SAINT_256_edge))

  G = to_networkx(s, to_undirected=True)
  node_pos=nx.spring_layout(G, seed=0)

  axs[i].axis('off')
  axs[i].set_title(f'Subgraph: {i:02d}')
  nx.draw_networkx_nodes(G,
                  pos=node_pos,
                  node_size=200,
                  node_color=np.array(colors)[s.y.numpy()],
                  alpha=0.6,
                  ax = axs[i]
                  )
plt.suptitle('Subgraphs using GraphSAINT Edge Sampler')
plt.show()

fig, axs = plt.subplots(1, 4, figsize=(24, 6))
axs = axs.flatten()
for i in range(4):
  s = next(iter(loader_SAINT_256_RW))

  G = to_networkx(s, to_undirected=True)
  node_pos=nx.spring_layout(G, seed=0)

  axs[i].axis('off')
  axs[i].set_title(f'Subgraph: {i:02d}')
  nx.draw_networkx_nodes(G,
                  pos=node_pos,
                  node_size=200,
                  node_color=np.array(colors)[s.y.numpy()],
                  alpha=0.6,
                  ax = axs[i]
                  )
plt.suptitle('Subgraphs using GraphSAINT random walk Sampler')
plt.show()
```
![image](https://github.com/user-attachments/assets/dc0f8e9f-48a1-4640-aa66-432b69c91f82)

![image](https://github.com/user-attachments/assets/926b7c67-13aa-489b-a44b-5431e07eed5d)

![image](https://github.com/user-attachments/assets/1a916cbb-5cde-42d0-83d9-80a8bfe81208)



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
