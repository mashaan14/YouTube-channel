# Spectral Clustering

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/k7M1TMYac-Y" frameborder="0" allowfullscreen></iframe>
</div>


## References:
```bibtex
@inproceedings{NIPS2001_801272ee,
  author    = {Ng, Andrew and Jordan, Michael and Weiss, Yair},
  booktitle = {Advances in Neural Information Processing Systems},
  title     = {On Spectral Clustering: Analysis and an algorithm},
  year      = {2001}
}
```

```bibtex
@misc{vonluxburg2007tutorial,
  title         = {A Tutorial on Spectral Clustering},
  author        = {Ulrike von Luxburg},
  year          = {2007},
  eprint        = {0711.0189},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DS}
}
```

## Prepare libraries and data

```python
# Standard libraries
import numpy as np
import pandas as pd
import seaborn as sns

# Plotting libraries
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm

# scikit-learn
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```

```python
random_seed = 42
plt.style.use('dark_background')
```

```python
X, y = make_blobs(n_samples=200, n_features=2, centers=1, cluster_std=0.15, random_state=random_seed)
# center at the origin
X = X - np.mean(X, axis=0)

X1, y1 = make_circles(n_samples=[600, 200], noise=0.04, factor=0.5, random_state=random_seed)
# add 1 to (make_circles) labels to account for (make_blobs) label
y1 = y1 + 1
# increase the radius
X1 = X1*3

X = np.concatenate((X, X1), axis=0)
y = np.concatenate((y, y1), axis=0)
```

```python
plot_colors = cm.tab10.colors
y_colors = np.array(plot_colors)[y]
```

```python
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X[:,0], X[:,1], marker='o', s=40, color=y_colors)

ax.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,
            labelbottom=False,labeltop=False,labelleft=False,labelright=False);
ax.set(xlabel=None, ylabel=None)
plt.show()
# plt.axis('off')
# plt.savefig('plot.png', bbox_inches='tight', dpi=600, transparent=True)
```

![image](https://github.com/user-attachments/assets/beca7d7f-2f1a-4722-9e5a-fd55f5a5ed81)

## Adjacency matrix $A$
An $n \times n$ matrix, where each element represents the disance between a pair of samples. This is th formula to construct the adjacency matrix $A$:

$A_{ij}=exp(\frac{-\big\|s_i-s_j\big\|^2}{2{\sigma}^2})$

```python
# An n by n matrix, where each element represents the similarity between a pair of samples
sigma = 1
A = -1 * np.square(X[:, None, :] - X[None, :, :]).sum(axis=-1)
A = np.exp(A / (2* sigma**2))
np.fill_diagonal(A, 0)
```

![image](https://github.com/user-attachments/assets/e0bad839-c6a9-4ad5-b9fa-a435a16044c3)

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

bin_values, _, _ = ax1.hist(A.flatten(), bins='auto')

# limit the y-axis to the value of the 10th bin
ax2.hist(A.flatten(), bins='auto')
ax2.set_ylim([0, bin_values[9]])

plt.show()
```

![image](https://github.com/user-attachments/assets/7c05567c-5722-4a1e-a724-c2adb023ea69)

```python
A1 = np.copy(A)
A1[A1 < 0.9] = 0
G = nx.from_numpy_array(A1)

plt.figure(figsize=(6,6))
plt.axis('off')
nx.draw_networkx_nodes(G, pos=X, node_size=20, node_color=y_colors, alpha=0.9)
nx.draw_networkx_edges(G, pos=X, edge_color="white", alpha=0.3)
plt.show()
# plt.savefig('plot-G.png', bbox_inches='tight', dpi=600, transparent=True)
```

![image](https://github.com/user-attachments/assets/88ecaa31-ed84-4bc0-a868-77f527e8805e)

## Graph Laplacian

```python
# identity matrix
I = np.zeros_like(A)
np.fill_diagonal(I, 1)

# degree matrix
D = np.zeros_like(A)
np.fill_diagonal(D, np.sum(A,axis=1))
D_inv_sqrt = np.linalg.inv(np.sqrt(D))

L = I - np.dot(D_inv_sqrt, A).dot(D_inv_sqrt)
```

## Eigenvalues and Eigenvectors

```python
eigenvalues, eigenvectors = np.linalg.eig(L)
eigenvalues = eigenvalues.real
eigenvectors = eigenvectors.real

# Order the eigenvalues in an increasing order
ind = np.argsort(eigenvalues, axis=0)
eigenvalues_sorted = np.take_along_axis(eigenvalues, ind, axis=0)

# Order the eigenvectors based on the magnitude of their corresponding eigenvalues
eigenvectors_sorted = eigenvectors.take(ind, axis=1)
```

```python
fig, axs = plt.subplots(3, 3, figsize=(16, 12))
eigen_v_x = np.linspace(0, eigenvectors_sorted.shape[0], eigenvectors_sorted.shape[0])

for j, ax in enumerate(fig.axes):
  eigen_v_y = eigenvectors_sorted[:,j]
  ax.scatter(eigen_v_x, eigen_v_y, marker='o', color=y_colors)
  ax.set_title(f'eigenvector {j} | eigenvalue: {eigenvalues_sorted[j]:.4f}')

plt.show()
```

![image](https://github.com/user-attachments/assets/9a57b8ce-e310-4ae6-8081-531f5464b13e)

```python
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(eigenvectors_sorted[:,0], eigenvectors_sorted[:,3], marker='o', s=40, color=y_colors)

ax.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,
            labelbottom=False,labeltop=False,labelleft=False,labelright=False);
ax.set(xlabel=None, ylabel=None)
plt.show()
```

![image](https://github.com/user-attachments/assets/0b969100-dfca-4f03-97c5-de9227457182)

```python
X_transformed = eigenvectors_sorted[:,[0,3]]

scaler = StandardScaler()
scaler.fit(X_transformed)
X_transformed_scaled = scaler.transform(X_transformed)


kmeans = KMeans(n_clusters = 3, random_state = random_seed, n_init='auto')
kmeans.fit(X_transformed_scaled)
```

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:,0], X[:,1], marker='o', s=40, color=y_colors)
ax1.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,
            labelbottom=False,labeltop=False,labelleft=False,labelright=False);
ax1.set(xlabel=None, ylabel=None)
ax1.set_title('True labels')

ax2.scatter(X[:,0], X[:,1], marker='o', s=40, color=np.array(plot_colors)[kmeans.labels_])
ax2.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,
            labelbottom=False,labeltop=False,labelleft=False,labelright=False);
ax2.set(xlabel=None, ylabel=None)
ax2.set_title('Predicted labels')

plt.show()
```

![image](https://github.com/user-attachments/assets/e59b8216-d6d4-4fa7-8302-33aa38801deb)


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
