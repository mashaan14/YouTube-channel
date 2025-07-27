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

In a [previous post](https://mashaan14.github.io/YouTube-channel/jax_performance/2025_07_14_jax_device_mesh) I wrote about running a vision transformer (ViT) using JAX device mesh. I tested different mesh setups and batch sizes. I was only looking at the runtime and memory consumption. But in this post I want to dive deeper into the time consumed by certain HLO Ops and see whether the machine is spending time in doing computations or communicating between devices. This figure XLA compilation workflow and optimization steps: 

![XLA-compilation](https://github.com/user-attachments/assets/08239147-e5e5-427c-9356-728c9341ac16)
> source: [Intel® Extension for TensorFlow](https://intel.github.io/intel-extension-for-tensorflow/latest/docs/guide/OpenXLA.html)

Here's a plot of Average Step Time (i.e., runtime time) of a parallel vision transformer using an 8 by 1 device mesh on JAX:

![runtime](https://github.com/user-attachments/assets/2e99d157-28d6-4ff2-b6fb-555eb785e0ef)

If we go to (HLO Op Stats) in TensorBoard we can see a breakdown of which HLO Ops taking time:

![HLO Op Stats pie chart](https://github.com/user-attachments/assets/8044aa2d-46c6-4f2b-8256-b5d2b443b4e5)

With batch_size=128, 7.1% of the time was spent performing (%all-reduce.104) HLO op. But with batch_size=4096, 3.1% of the time was spent performing (%fusion.253) HLO op.

![Trace Viewer batch_size 128](https://github.com/user-attachments/assets/79d9f654-033c-4f94-95a9-a0259164ef1a)

![Trace Viewer batch_size 4096](https://github.com/user-attachments/assets/b42bbf21-fe72-4407-ba1b-768160184f58)
