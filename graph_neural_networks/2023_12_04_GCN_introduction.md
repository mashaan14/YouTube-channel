# Graph Convolutional Networks (GCNs)

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/G6c6zk0RhRM" frameborder="0" allowfullscreen></iframe>
</div>



## Acknowledgment:
I borrowed some code from these resources:
  - [https://github.com/tkipf/pygcn/tree/master](https://github.com/tkipf/pygcn/tree/master)
  - [https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html](https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html)

## References:
```bibtex
@misc{kipf2017semisupervised,
    title         = {Semi-Supervised Classification with Graph Convolutional Networks},
    author        = {Thomas N. Kipf and Max Welling},
    year          = {2017},
    eprint        = {1609.02907},
    archivePrefix = {arXiv},
    primaryClass  = {cs.LG}
}
```

## Prepare libraries and data

```bash
!pip install torch-geometric
```

```python
# Standard libraries
import math
import time
import numpy as np
import pandas as pd

# Plotting libraries
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# PyTorch geometric
import torch_geometric
from torch_geometric.datasets import StochasticBlockModelDataset
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_dense_adj
from torch_geometric.transforms import RandomNodeSplit
```

```python
num_nodes_per_class = 100
num_nodes = [num_nodes_per_class] * 3
edge_probs = [[0.1, 0.05, 0.02],
              [0.05, 0.1, 0.02],
              [0.02, 0.02, 0.1]]
dataset = StochasticBlockModelDataset('/content', num_nodes, edge_probs, num_channels=10)
```

```bash
dataset[0]
```

```console
Data(x=[300, 10], edge_index=[2, 4750], y=[300])
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

![image](https://github.com/user-attachments/assets/f53c5b16-da76-4284-98b7-41f8f905958e)

```python
split = RandomNodeSplit(num_val=0.1, num_test=0.1)
data = split(dataset[0])
data
```

```console
Data(x=[300, 10], edge_index=[2, 4750], y=[300], train_mask=[300], val_mask=[300], test_mask=[300])
```

## GCN
Here is the formula that shows the architecture for GCN with two convolutional layers:

![GCN-001](https://github.com/user-attachments/assets/98f60866-d346-44c6-9820-7f9b1cc51997)

> source: equation (9) by Kipf and Welling (2017)


```python
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
```

```python
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid) # nfeat = 10, nhid = 3
        self.gc2 = GraphConvolution(nhid, nclass) # nhid = 3, nclass = 3
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
```

## Training

```python
# normalized adjacency

adj = to_dense_adj(data.edge_index)[0]
# symmetric adjacency matrix
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = adj + torch.eye(adj.shape[0])
# degree matrix
d = np.zeros((adj.shape[0],adj.shape[0]))
np.fill_diagonal(d, adj.sum(1).numpy())
d = torch.from_numpy(d)
d_inv_sqrt = torch.pow(d, -0.5)
# set inf values to zero
d_inv_sqrt = torch.nan_to_num(d_inv_sqrt, posinf=0.0)
d_inv_sqrt = d_inv_sqrt.to(dtype=torch.float32)

adj = (d_inv_sqrt @ adj) @ d_inv_sqrt
```

```python
features = data.x
features = nn.functional.normalize(features)
labels = data.y
idx_train = data.train_mask
idx_val = data.val_mask
idx_test = data.test_mask
```

```python
model = GCN(nfeat=features.shape[1],
            nhid=3, # Number of hidden units
            nclass=labels.max().item() + 1,
            dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
```

```python
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
```

```python
loss_train_list = []
loss_val_list = []

for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train_list.append(loss_train.item())
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # Evaluate validation set performance separately,
    # deactivates dropout during validation run.
    model.eval()
    output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    loss_val_list.append(loss_val.item())
    acc_val = accuracy(output[idx_val], labels[idx_val])

    # Print evaluation metrics every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch:>3} | Train Acc: {acc_train.item()*100:.2f}% | Validation Acc: {acc_val.item()*100:.2f}%')
```

```console
Epoch   0 | Train Acc: 32.92% | Validation Acc: 40.00%
Epoch  10 | Train Acc: 32.92% | Validation Acc: 40.00%
Epoch  20 | Train Acc: 32.92% | Validation Acc: 40.00%
Epoch  30 | Train Acc: 45.42% | Validation Acc: 46.67%
Epoch  40 | Train Acc: 48.75% | Validation Acc: 50.00%
Epoch  50 | Train Acc: 55.00% | Validation Acc: 70.00%
Epoch  60 | Train Acc: 62.08% | Validation Acc: 70.00%
Epoch  70 | Train Acc: 62.08% | Validation Acc: 73.33%
Epoch  80 | Train Acc: 60.83% | Validation Acc: 73.33%
Epoch  90 | Train Acc: 60.83% | Validation Acc: 70.00%
Epoch 100 | Train Acc: 72.08% | Validation Acc: 70.00%
...
...
...
Epoch 1910 | Train Acc: 75.00% | Validation Acc: 86.67%
Epoch 1920 | Train Acc: 78.33% | Validation Acc: 76.67%
Epoch 1930 | Train Acc: 77.08% | Validation Acc: 76.67%
Epoch 1940 | Train Acc: 74.58% | Validation Acc: 80.00%
Epoch 1950 | Train Acc: 78.75% | Validation Acc: 76.67%
Epoch 1960 | Train Acc: 76.25% | Validation Acc: 80.00%
Epoch 1970 | Train Acc: 74.17% | Validation Acc: 76.67%
Epoch 1980 | Train Acc: 77.50% | Validation Acc: 76.67%
Epoch 1990 | Train Acc: 74.58% | Validation Acc: 76.67%
```

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(loss_train_list)
ax1.set_title('Training loss')
ax2.plot(loss_val_list)
ax2.set_title('Validation loss')
plt.show()
```

![image](https://github.com/user-attachments/assets/bf81d1a1-e98a-47bc-96f8-af518f254226)

## Testing

```python
model.eval()
output = model(features, adj)
loss_test = F.nll_loss(output[idx_test], labels[idx_test])
idx_test_preds = output[idx_test].max(1)[1].type_as(labels)
acc_test = accuracy(output[idx_test], labels[idx_test])
print(f'Test Acc: {acc_test.item()*100:.2f}% | Test loss: {loss_test.item():.2f}')
```

```console
Test Acc: 90.00% | Test loss: 0.46
```

```python
torch.sum(labels[idx_test]!=idx_test_preds).item()
```

```console
3
```

```python
G1 = nx.from_numpy_array(adj.numpy())
node_pos=nx.spring_layout(G, seed=0)

idx_train_nodes = np.nonzero(idx_train.numpy())[0]
idx_val_nodes = np.nonzero(idx_val.numpy())[0]
idx_test_nodes = np.nonzero(idx_test.numpy())[0]

idx_train_graph = G.subgraph(idx_train_nodes.tolist())
idx_val_graph = G.subgraph(idx_val_nodes.tolist())

idx_test_graph = G.subgraph(idx_test_nodes.tolist())
idx_test_graph_colors_true = np.array(colors)[labels[idx_test]]
idx_test_graph_colors_preds = np.array(colors)[idx_test_preds]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.set_title('True labels')
nx.draw_networkx_nodes(idx_train_graph,
                pos=node_pos,
                node_size=200,
                node_color='grey',
                alpha=0.15,
                ax = ax1
                )
nx.draw_networkx_nodes(idx_val_graph,
                pos=node_pos,
                node_size=200,
                node_color='grey',
                alpha=0.15,
                ax = ax1
                )
nx.draw_networkx_nodes(idx_test_graph,
                pos=node_pos,
                node_size=200,
                node_color=idx_test_graph_colors_true,
                alpha=0.9,
                ax = ax1
                )

ax2.set_title('Predicted labels')
nx.draw_networkx_nodes(idx_train_graph,
                pos=node_pos,
                node_size=200,
                node_color='grey',
                alpha=0.15,
                ax = ax2
                )
nx.draw_networkx_nodes(idx_val_graph,
                pos=node_pos,
                node_size=200,
                node_color='grey',
                alpha=0.15,
                ax = ax2
                )
nx.draw_networkx_nodes(idx_test_graph,
                pos=node_pos,
                node_size=200,
                node_color=idx_test_graph_colors_preds,
                alpha=0.9,
                ax = ax2
                )

plt.show()
```

![image](https://github.com/user-attachments/assets/e042e5f5-222e-442d-bbaf-5c0f984655ee)

```bash
model
```

```console
GCN(
  (gc1): GraphConvolution (10 -> 3)
  (gc2): GraphConvolution (3 -> 3)
)
```

```python
embeddings = F.relu(model.gc1(features, adj)).detach().numpy()
embeddings.shape
```

```console
(300, 3)
```

```python
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(embeddings[:,0], embeddings[:,1], embeddings[:,2], color=y_colors)
plt.show()
```

![image](https://github.com/user-attachments/assets/a3c8d25d-458b-43c5-9b08-b22dc09e3d16)


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
plt.savefig('plot-01.png', bbox_inches='tight', dpi=600)
```

![image](https://github.com/user-attachments/assets/57f4c8c3-7eaf-458c-ad99-80368253d066)

```python
plt.figure(figsize=(8,8))
plt.axis('off')
nx.draw_networkx_nodes(idx_train_graph,
                pos=node_pos,
                node_size=200,
                node_color='grey',
                alpha=0.15
                )
nx.draw_networkx_nodes(idx_val_graph,
                pos=node_pos,
                node_size=200,
                node_color='grey',
                alpha=0.15
                )
nx.draw_networkx_nodes(idx_test_graph,
                pos=node_pos,
                node_size=200,
                node_color=idx_test_graph_colors_preds,
                alpha=0.9
                )

plt.savefig('plot-02.png', bbox_inches='tight', dpi=600)
```

![image](https://github.com/user-attachments/assets/0e808344-3df3-4f1b-b445-97a811235a30)




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
