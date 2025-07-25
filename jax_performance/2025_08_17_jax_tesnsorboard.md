# Profiling JAX with TesnorBoard 

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/znxzsPzNK6c" frameborder="0" allowfullscreen></iframe>
</div>

## Contents

* [Acknowledgment](#acknowledgment)

## Acknowledgment
These resources were helpful in preparing this post:
  - [How to Profile TPU Programs](https://jax-ml.github.io/scaling-book/profiling/)
  - [EQuARX: Efficient Quantized All Reduce in XLA for Distributed Machine Learning Acceleration](https://arxiv.org/abs/2506.17615)


## What the machine is actually doing?

In a previous post I wrote about running a vision transformer (ViT) using JAX device mesh. I tested different mesh setups and batch sizes. I was only looking at the runtime and memory consumption. But in this post I want to dive deeper into the time consumed by certain HLO Ops and see whether the machine is spending time in doing computations or communicating between devices.

So, here's a plot of Average Step Time (i.e., runtime time)

![runtime](https://github.com/user-attachments/assets/2e99d157-28d6-4ff2-b6fb-555eb785e0ef)

If we go to (HLO Op Stats) in TensorBoard we can see a breakdown of which HLO Ops taking time

![drawings-01 001](https://github.com/user-attachments/assets/d41a00fa-4c0b-472c-baf5-447f2a69d571)

![Screenshot 2025-07-23 at 5 08 56 PM](https://github.com/user-attachments/assets/d5ff569b-dd84-4ca3-8cf3-205434b475d4)


