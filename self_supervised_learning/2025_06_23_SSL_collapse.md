# Collapse in Self-Supervised Learning

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/7NE0NH-PfkA" frameborder="0" allowfullscreen></iframe>
</div>


## Acknowledgment:
Thanks to the authors for making their code available. I borrowed some code from [DINOv2](https://github.com/facebookresearch/dinov2) and [I-JEPA](https://github.com/facebookresearch/ijepa/tree/main) repositories.

## References:

```bibtex
@Article{chen2020simsiam,
  author  = {Xinlei Chen and Kaiming He},
  title   = {Exploring Simple Siamese Representation Learning},
  journal = {arXiv preprint arXiv:2011.10566},
  year    = {2020},
}
```

```bibtex
@article{Jing2021UnderstandingDC,
  title   = {Understanding Dimensional Collapse in Contrastive Self-supervised Learning},
  author  = {Li Jing and Pascal Vincent and Yann LeCun and Yuandong Tian},
  journal = {arXiv preprint arXiv:2110.09348},
  year    = {2021}
}
```

```bibtex
@inproceedings{wang2022asym,
  title     = {On the Importance of Asymmetry for Siamese Representation Learning},
  author    = {Xiao Wang and Haoqi Fan and Yuandong Tian and Daisuke Kihara and Xinlei Chen},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
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
