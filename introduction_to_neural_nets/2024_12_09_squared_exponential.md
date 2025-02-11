# Squared Exponential

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/H4S3QAoMEEo" frameborder="0" allowfullscreen></iframe>
</div>


## References
```bibtex
@book{williams2006gaussian,
 title      = {Gaussian processes for machine learning},
 author     = {Williams, Christopher KI and Rasmussen, Carl Edward},
 volume     = {2},
 number     = {3},
 year       = {2006},
 publisher  = {MIT press Cambridge, MA}
}
```

## Squared exponential formula

The squared exponential is written as $e^{\frac{\left\Vert x_i - x_j \right\Vert^2}{2\sigma ^2}}$. The inline formula is usually expressed as $exp{(\frac{\left\Vert x_i - x_j \right\Vert^2}{2\sigma ^2})}$

## Plotting the exponent function

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
```

```python
random_seed = 42
plt.style.use('dark_background')
```

### The exponent function with positive input

```python
x = np.arange(0, 10, 1, dtype=int)
plt.plot(x, np.exp(x))
plt.xlabel('$x$')
plt.ylabel('$e^x$')
plt.show()
```

![image](https://github.com/user-attachments/assets/ef0e4eb6-6229-4def-bec3-22db4cceca83)

### The exponent function with negative input

```python
x = np.arange(0, 10, 1, dtype=int)
plt.plot(x, np.exp(-x))
plt.xlabel('$x$')
plt.ylabel('$e^{-x}$')
plt.show()
```

![image](https://github.com/user-attachments/assets/ea95ec98-8e92-4916-9f2e-914e534276b3)

### The exponent function with negative input and different scales

```python
x = np.arange(0, 100, 1, dtype=int)
for i in [1,2,3]:
  y = np.exp(-x/(2*(i**2)))
  plt.plot(x, y, label='$\sigma=$'+str(i))

plt.legend(fontsize = 18)
plt.xlabel('$x$')
plt.ylabel('$e^{x/{(2\sigma ^2)}}$')
plt.show()
```

![image](https://github.com/user-attachments/assets/54d2d5c6-1824-4920-8eee-74cc1233fbb0)

## The adjacency matrix using squared exponential function

### Generating the data

```python
X, y = make_blobs(n_samples=100, n_features=2, centers=1, cluster_std=0.15, random_state=random_seed)
# center at the origin
X = X - np.mean(X, axis=0)

X1, y1 = make_circles(n_samples=(300, 100), noise=0.04, factor=0.5, random_state=random_seed)
# add 1 to (make_circles) labels to account for (make_blobs) label
y1 = y1 + 1
# increase the radius
X1 = X1*3

X = np.concatenate((X, X1), axis=0)
y = np.concatenate((y, y1), axis=0)

fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(X[:,0], X[:,1], marker='o', s=40)
ax.set_xticks([])
ax.set_yticks([])
plt.show()
```

![image](https://github.com/user-attachments/assets/9cd25acc-d418-4c3a-8a45-70ba1e4d118a)

```python
A = np.square(X[:, None, :] - X[None, :, :]).sum(axis=-1)
A_exp = np.exp(-1*A)
print(A.shape)
```

```console
(500, 500)
```

### The adjacency with squared exponential ***vs*** The adjacency without squared exponential

```python
fig, axs = plt.subplots(1, 2, figsize=(12,5))
axs[0].hist(A.flatten())
axs[0].set_title('$||x_i - x_j||^2$')
axs[1].hist(A_exp.flatten())
axs[1].set_title('$e^{-||x_i - x_j||^2}$')
plt.show()
```

![image](https://github.com/user-attachments/assets/df5ac391-5b3f-4a57-adc6-87ccf72d0f6a)

```python
fig, axs = plt.subplots(1, 2, figsize=(12,5))
axs[0].hist(A[0,:])
axs[0].set_title('$||x_i - x_j||^2$')
axs[1].hist(A_exp[0,:])
axs[1].set_title('$e^{-||x_i - x_j||^2}$')
plt.show()
```

![image](https://github.com/user-attachments/assets/26b2674f-3809-4a70-875a-075f2582fc04)

## Softmax function

### Generating random numbers

```python
np.random.seed(random_seed)
x = np.random.randint(10, size=(6, 10))

fig, axs = plt.subplots(2, 3, figsize=(10, 5), layout="constrained")
for i, ax in enumerate(axs.ravel()):
    ax.bar(np.arange(10), x[i,:])

plt.show()
```

![image](https://github.com/user-attachments/assets/8d35ebb2-c8a2-4379-980b-5c605b44726c)

### Normalizing using softmax

```python
x_exp = np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=1), axis=1)

fig, axs = plt.subplots(2, 3, figsize=(10, 5), layout="constrained")
for i, ax in enumerate(axs.ravel()):
    ax.bar(np.arange(10), x_exp[i,:])

plt.show()
```

![image](https://github.com/user-attachments/assets/4fb9ebf5-856a-476e-9d9e-31358feb1428)

```python
print(x_exp)

import torch
from torch import nn

m = nn.Softmax(dim=1)
print(m(torch.from_numpy(x).float()))
```

```console
[[3.46489855e-02 1.72507141e-03 9.41857077e-02 4.68923027e-03
  3.46489855e-02 6.95943478e-01 6.34618306e-04 3.46489855e-02
  9.41857077e-02 4.68923027e-03]
 [5.46659002e-03 2.98465702e-01 2.98465702e-01 2.01104608e-03
  4.03929403e-02 1.48597323e-02 7.39822508e-04 2.98465702e-01
  4.03929403e-02 7.39822508e-04]
 [2.75427727e-03 5.04463479e-05 4.08770991e-01 7.48690186e-03
  1.50378444e-01 5.04463479e-05 4.08770991e-01 3.72750895e-04
  2.03515093e-02 1.01324198e-03]
 [4.31051357e-01 1.06846949e-03 7.89498100e-03 1.06846949e-03
  5.83364575e-02 7.89498100e-03 4.31051357e-01 5.83364575e-02
  3.93067959e-04 2.90440120e-03]
 [1.25523185e-01 1.14462329e-04 3.41207392e-01 1.25523185e-01
  3.41207392e-01 2.29903733e-03 1.14462329e-04 8.45768567e-04
  1.69877158e-02 4.61773991e-02]
 [7.73679396e-04 1.04706120e-04 2.10307864e-03 2.84620744e-04
  1.14824203e-01 2.10307864e-03 2.84620744e-04 1.55397661e-02
  1.55397661e-02 8.48442480e-01]]
tensor([[3.4649e-02, 1.7251e-03, 9.4186e-02, 4.6892e-03, 3.4649e-02, 6.9594e-01,
         6.3462e-04, 3.4649e-02, 9.4186e-02, 4.6892e-03],
        [5.4666e-03, 2.9847e-01, 2.9847e-01, 2.0110e-03, 4.0393e-02, 1.4860e-02,
         7.3982e-04, 2.9847e-01, 4.0393e-02, 7.3982e-04],
        [2.7543e-03, 5.0446e-05, 4.0877e-01, 7.4869e-03, 1.5038e-01, 5.0446e-05,
         4.0877e-01, 3.7275e-04, 2.0352e-02, 1.0132e-03],
        [4.3105e-01, 1.0685e-03, 7.8950e-03, 1.0685e-03, 5.8336e-02, 7.8950e-03,
         4.3105e-01, 5.8336e-02, 3.9307e-04, 2.9044e-03],
        [1.2552e-01, 1.1446e-04, 3.4121e-01, 1.2552e-01, 3.4121e-01, 2.2990e-03,
         1.1446e-04, 8.4577e-04, 1.6988e-02, 4.6177e-02],
        [7.7368e-04, 1.0471e-04, 2.1031e-03, 2.8462e-04, 1.1482e-01, 2.1031e-03,
         2.8462e-04, 1.5540e-02, 1.5540e-02, 8.4844e-01]])
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
