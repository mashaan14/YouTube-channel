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
