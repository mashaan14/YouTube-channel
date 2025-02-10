# Graph Convolutional Networks (GCNs) and Simple Graph Convolution (SGC)

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/PQT2QblNegY" frameborder="0" allowfullscreen></iframe>
</div>


I borrowed some code from these resources:
  - [https://github.com/tkipf/pygcn/tree/master](https://github.com/tkipf/pygcn/tree/master)
  - [https://github.com/Tiiiger/SGC](https://github.com/Tiiiger/SGC)
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

```bibtex
@misc{wu2019simplifying,
  title         = {Simplifying Graph Convolutional Networks},
  author        = {Felix Wu and Tianyi Zhang and Amauri Holanda de Souza Jr. au2 and Christopher Fifty and Tao Yu and Kilian Q. Weinberger},
  year          = {2019},
  eprint        = {1902.07153},
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
num_nodes = [num_nodes_per_class] * 5
edge_probs = [[.15, .05, .05, .05, .05],
              [.05, .15, .02, .02, .02],
              [.05, .02, .15, .05, .02],
              [.05, .02, .05, .15, .02],
              [.05, .02, .02, .02, .15]]
dataset = StochasticBlockModelDataset('/content', num_nodes, edge_probs, num_channels=10)
```

```python
colors = cm.tab10.colors
y_colors = np.array(colors)[dataset[0].y.numpy()]

dataset[0]
```

```console
Data(x=[500, 10], edge_index=[2, 14578], y=[500])
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

![image](https://github.com/user-attachments/assets/ed28cf5d-c82e-4a52-8a8c-85ef83f409f5)

```python
split = RandomNodeSplit(num_val=0.1, num_test=0.1)
data = split(dataset[0])
data
```

```console
Data(x=[500, 10], edge_index=[2, 16862], y=[500], train_mask=[500], val_mask=[500], test_mask=[500])
```

## GCN

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

## SGC

```python
def sgc_precompute(features, adj, degree):
  for i in range(degree):
    features = torch.spmm(adj, features)
  return features

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)
```

## Training

### Prepare data

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
# normalized features

features = data.x
features = nn.functional.normalize(features)
labels = data.y
idx_train = data.train_mask
idx_val = data.val_mask
idx_test = data.test_mask
```

### GCN training

```python
model_GCN = GCN(nfeat=features.shape[1],
            nhid=3, # Number of hidden units
            nclass=labels.max().item() + 1,
            dropout=0.5)
optimizer = optim.Adam(model_GCN.parameters(), lr=0.02, weight_decay=5e-4)
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

for epoch in range(1000):
    model_GCN.train()
    optimizer.zero_grad()
    output = model_GCN(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    loss_train_list.append(loss_train.item())
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # Evaluate validation set performance separately,
    # deactivates dropout during validation run.
    with torch.no_grad():
      model_GCN.eval()
      output = model_GCN(features, adj)
      loss_val = F.nll_loss(output[idx_val], labels[idx_val])
      loss_val_list.append(loss_val.item())
      acc_val = accuracy(output[idx_val], labels[idx_val])

    # Print evaluation metrics every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch:>3} | Train Acc: {acc_train.item()*100:.2f}% | Validation Acc: {acc_val.item()*100:.2f}%')
```

```console
Epoch   0 | Train Acc: 18.00% | Validation Acc: 38.00%
Epoch  10 | Train Acc: 37.25% | Validation Acc: 22.00%
Epoch  20 | Train Acc: 22.25% | Validation Acc: 10.00%
Epoch  30 | Train Acc: 32.00% | Validation Acc: 44.00%
Epoch  40 | Train Acc: 40.25% | Validation Acc: 46.00%
Epoch  50 | Train Acc: 60.50% | Validation Acc: 70.00%
Epoch  60 | Train Acc: 58.50% | Validation Acc: 72.00%
Epoch  70 | Train Acc: 57.00% | Validation Acc: 74.00%
Epoch  80 | Train Acc: 58.25% | Validation Acc: 74.00%
Epoch  90 | Train Acc: 66.00% | Validation Acc: 76.00%
Epoch 100 | Train Acc: 59.50% | Validation Acc: 72.00%
...
...
...
Epoch 910 | Train Acc: 84.00% | Validation Acc: 94.00%
Epoch 920 | Train Acc: 84.00% | Validation Acc: 96.00%
Epoch 930 | Train Acc: 83.25% | Validation Acc: 98.00%
Epoch 940 | Train Acc: 84.25% | Validation Acc: 96.00%
Epoch 950 | Train Acc: 87.75% | Validation Acc: 98.00%
Epoch 960 | Train Acc: 85.25% | Validation Acc: 96.00%
Epoch 970 | Train Acc: 90.75% | Validation Acc: 96.00%
Epoch 980 | Train Acc: 85.00% | Validation Acc: 96.00%
Epoch 990 | Train Acc: 87.50% | Validation Acc: 96.00%
```

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(loss_train_list)
ax1.set_title('GCN Training loss')
ax2.plot(loss_val_list)
ax2.set_title('GCN Validation loss')
plt.show()
```

![image](https://github.com/user-attachments/assets/bc1a962c-25df-47d3-82f7-821902b94804)

### SGC training

```python
model_SGC = SGC(nfeat=features.shape[1],
            nclass=labels.max().item() + 1)
optimizer = optim.Adam(model_SGC.parameters(), lr=1, weight_decay=0)
```

```python
# degree of the approximation was set to 2 by default
features = sgc_precompute(features, adj, 2)
```

```python
loss_train_list = []
loss_val_list = []

for epoch in range(1000):
    model_SGC.train()
    optimizer.zero_grad()
    output = model_SGC(features[idx_train])
    loss_train = F.cross_entropy(output, labels[idx_train])
    loss_train_list.append(loss_train.item())
    acc_train = accuracy(output, labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # Evaluate validation set performance separately,
    # deactivates dropout during validation run.
    with torch.no_grad():
      model_SGC.eval()
      output = model_SGC(features[idx_val])
      loss_val = F.cross_entropy(output, labels[idx_val])
      loss_val_list.append(loss_val.item())
      acc_val = accuracy(output, labels[idx_val])

    # Print evaluation metrics every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch:>3} | Train Acc: {acc_train.item()*100:.2f}% | Validation Acc: {acc_val.item()*100:.2f}%')
```

```console
Epoch   0 | Train Acc: 20.00% | Validation Acc: 30.00%
Epoch  10 | Train Acc: 87.00% | Validation Acc: 100.00%
Epoch  20 | Train Acc: 97.50% | Validation Acc: 100.00%
Epoch  30 | Train Acc: 97.75% | Validation Acc: 100.00%
Epoch  40 | Train Acc: 97.50% | Validation Acc: 100.00%
Epoch  50 | Train Acc: 97.75% | Validation Acc: 100.00%
Epoch  60 | Train Acc: 97.75% | Validation Acc: 100.00%
Epoch  70 | Train Acc: 98.25% | Validation Acc: 100.00%
Epoch  80 | Train Acc: 98.25% | Validation Acc: 100.00%
Epoch  90 | Train Acc: 98.50% | Validation Acc: 100.00%
Epoch 100 | Train Acc: 98.50% | Validation Acc: 100.00%
...
...
...
Epoch 910 | Train Acc: 98.25% | Validation Acc: 100.00%
Epoch 920 | Train Acc: 98.25% | Validation Acc: 100.00%
Epoch 930 | Train Acc: 98.25% | Validation Acc: 100.00%
Epoch 940 | Train Acc: 98.25% | Validation Acc: 100.00%
Epoch 950 | Train Acc: 98.25% | Validation Acc: 100.00%
Epoch 960 | Train Acc: 98.25% | Validation Acc: 100.00%
Epoch 970 | Train Acc: 98.25% | Validation Acc: 100.00%
Epoch 980 | Train Acc: 98.25% | Validation Acc: 100.00%
Epoch 990 | Train Acc: 98.25% | Validation Acc: 100.00%
```

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(loss_train_list)
ax1.set_title('SGC Training loss')
ax2.plot(loss_val_list)
ax2.set_title('SGC Validation loss')
plt.show()
```

![image](https://github.com/user-attachments/assets/6cc73a95-fcca-4845-a071-aa6bc61f0c75)

## Testing

### GCN testing

```python
model_GCN.eval()
output = model_GCN(features, adj)
loss_test = F.nll_loss(output[idx_test], labels[idx_test])
idx_test_preds = output[idx_test].max(1)[1].type_as(labels)
acc_test = accuracy(output[idx_test], labels[idx_test])
print(f'Test Acc: {acc_test.item()*100:.2f}% | Test loss: {loss_test.item():.2f}')
```

```console
Test Acc: 96.00% | Test loss: 0.36
```

```python
torch.sum(labels[idx_test]!=idx_test_preds).item()
```

```console
2
```

### SGC testing

```python
model_SGC.eval()
output = model_SGC(features[idx_test])
loss_test = F.cross_entropy(output, labels[idx_test])
idx_test_preds = output.max(1)[1].type_as(labels)
acc_test = accuracy(output, labels[idx_test])
print(f'Test Acc: {acc_test.item()*100:.2f}% | Test loss: {loss_test.item():.2f}')
```

```console
Test Acc: 100.00% | Test loss: 0.01
```

```python
torch.sum(labels[idx_test]!=idx_test_preds).item()
```

```console
0
```

## Plot

```python
model_GCN
```

```console
GCN(
  (gc1): GraphConvolution (10 -> 3)
  (gc2): GraphConvolution (3 -> 5)
)
```

```python
embeddings = F.relu(model_GCN.gc1(features, adj)).detach().numpy()
embeddings.shape
```

```console
(500, 3)
```

```python
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(embeddings[:,0], embeddings[:,1], embeddings[:,2], color=y_colors)
plt.show()
```

![image](https://github.com/user-attachments/assets/d13694a9-cbc8-4718-85ce-3b5303a7931d)




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
