# Fine-tuning a Vision Transformer

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/RjG6_FP_DgU" frameborder="0" allowfullscreen></iframe>
</div>


## Acknowledgement
Most of the instruction in this tutorial were taken from [Vision Transformer and MLP-Mixer Architectures](https://github.com/google-research/vision_transformer/tree/main). Thanks to Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) for making cloud TPUs available for my use.

## References

```bibtex
@article{dosovitskiy2020vit,
title   = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
author  = {Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
journal = {ICLR},
year    = {2021}
}
```

```bibtex
@software{jax2018github,
author  = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
title   = {JAX: composable transformations of Python+NumPy programs},
url     = {http://github.com/jax-ml/jax},
version = {0.3.13},
year    = {2018},
}
```

## ViT Architecture

![image](https://github.com/user-attachments/assets/be3e5759-a956-4804-be51-e0acbabeabed)


There are 4 classes in [`models_vit.py`](https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py) that represent the architecture of a vision transformer:
* `AddPositionEmbs`: Adds learned positional embeddings to the inputs. It's called at the beginning of class `Encoder` to add positional embeddings to patches. 
* `MlpBlock`: Transformer MLP which got called at the end of `Encoder1DBlock` after perfroming attention.
```python
class MlpBlock(nn.Module):
    x = nn.Dense(inputs)    # dense layer
    x = nn.gelu(x)          # nonlinearity
    output = nn.Dense(x)    # dense layer
    return output
```
* `Encoder1DBlock`: Transformer encoder layer which got called iteratively inside `Encoder` class, each iteration represents a transformer layer. `Encoder1DBlock` performs the following tasks:
```python
class Encoder1DBlock(nn.Module):
    x = nn.LayerNorm(inputs)        # LayerNorm
    x = nn.MultiHeadAttention(x)    # attention
    x = x + inputs                  # residuals
    y = nn.LayerNorm(x)             # LayerNorm
    y = MlpBlock(y)                 # mlp
    return x + y                    # residuals
```

* `Encoder`: It adds the positional encoding then runs a number of `Encoder1DBlock` layers. The following pseudocode summarises the class `Encoder` tasks:
```python
class Encoder(nn.Module):
    x = AddPositionEmbs(x)          # positional encoding
    for lyr in range(num_layers):   # looping over transformer layers   
        x = Encoder1DBlock(x)       # a single transformer layer
    encoded = nn.LayerNorm(x)       # LayerNorm
    return encoded
```

* `VisionTransformer`: It splits the images into patches, then runs the transformer layers. At the end, it runs an mlp to get the preditions. Here is a pseudocode for `VisionTransformer` class:
```python
class VisionTransformer(nn.Module):
    x = nn.Conv(x)                      # split an image into patches using nn.conv layer
    x = jnp.reshape(x, [n, h * w, c])   # flatten height and width
    x = encoder(x)                      # run the patches through class Encoder
    x = nn.Dense(x)                     # mlp head
    return x
```

## Datasets

### imagenet21k
ImageNet21k contains 14,197,122 images divided into 21,841 classes. Images are usually resized $224 \times 224$ pixels.

### cifar10
cifar10 contains 60,000 divided into 10 classes, with 6000 images per class. All images are of size $32 \times 32$ pixels.

## How does this repository fine-tune a model?

1. `params` are initialized randomly in `train.py`.
1. The random `params` are passed to `checkpoint.py` along with a path to the pretrained model.
1. All `params` are replaced with the ones in the pretrained model using `inspect_params` function.
1. The parameters of the `head` layer are set back to random.

## Checking VM specs

to check the memory on VM:
```bash
free -h
```

disk check:
```bash
df -h
```

I couldn't find a terminal command to check TPUs. But we can check how many TPUs we have by running the following jax command:
```bash
python -c 'import jax; print(jax.devices())'
```

## Fine-tuning

Check conda environments, then create a new one for ViT fine-tuning.
```bash
conda env list
```

```bash
conda create --name ViT     # create a new conda environment
conda activate ViT
```

Clone the vision transformer repository and change the directory to get inside the folder.
```bash
git clone --depth=1 https://github.com/google-research/vision_transformer
cd vision_transformer
```

Install the requirements specific for a TPU virtual machine.
```bash
pip install -r vit_jax/requirements-tpu.txt
```

We're going to run the fine-tuning script for $2000$ steps to save some time. I already run the fine-tuning script for $10000$ steps, and I got these results:

|step|train loss|test accuracy|
| ---: | ---: | ---: |
|200  |0.9816|0.9567|
|1000 |0.3115|0.9848|
|2000 |0.3293|0.9877|
|5000 |0.3831|0.9891|
|7000 |0.2774|0.9893|
|10000|0.3059|0.9896|

```bash
python -m vit_jax.main --workdir=/tmp/vit-$(date +%s) \
    --config=$(pwd)/vit_jax/configs/vit.py:b16,cifar10 \
    --config.pretrained_dir='gs://vit_models/imagenet21k'\
    --config.total_steps=2000
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
