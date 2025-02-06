# NeRF: Neural Radiance Fields

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/kszswpg7sjs" frameborder="0" allowfullscreen></iframe>
</div>

## Code releases

* Original algorithm:
  * [TF repository](https://github.com/bmild/nerf): The one released with the paper.
  * [JAX repository](https://github.com/google-research/google-research/tree/master/jaxnerf): The most recent, and referred to frequently in this tutorial.

* Followup Works:
  * [Mip-NeRF](https://github.com/google/mipnerf): This implementation is written in JAX, and is a fork of Google's [JaxNeRF implementation](https://github.com/google-research/google-research/tree/master/jaxnerf).
  * [MultiNeRF](https://github.com/google-research/multinerf): The code release for three CVPR 2022 papers: Mip-NeRF 360, Ref-NeRF, and RawNeRF. This implementation is written in JAX, and is a fork of [Mip-NeRF](https://github.com/google/mipnerf).

## References
```bibtex
@inproceedings{mildenhall2020nerf,
title     ={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
author    ={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
year      ={2020},
booktitle ={ECCV},
}
```
```bibtex
@software{jaxnerf2020github,
author  = {Boyang Deng and Jonathan T. Barron and Pratul P. Srinivasan},
title   = {JaxNeRF: an efficient JAX implementation of NeRF},
url     = {https://github.com/google-research/google-research/tree/master/jaxnerf},
version = {0.0},
year    = {2020},
}
```

## Ray casting

**Ray casting** is the process in a ray tracing algorithm that shoots one or more rays from the camera (eye position) through each pixel in an image plane.

![image](https://github.com/user-attachments/assets/cbc49368-df69-43c6-bbc7-67c905228893)

>image source: https://developer.nvidia.com/discover/ray-tracing


## NeRF input

![image](https://github.com/user-attachments/assets/0edcb671-038a-4221-8bde-9b9f7542cf12)


## Volumetric formulation for NeRF

![image](https://github.com/user-attachments/assets/ec7a0c75-87f9-4e44-b8ba-d99f475df891)
>image source: https://graphics.stanford.edu/courses/cs348n-22-winter/LectureSlides/FinalSlides/leo_class_nerf_2022.pdf


## Hierarchical volume sampling

![image](https://github.com/user-attachments/assets/1ad4c90d-e56a-46c1-8722-7549cdb01ffa)
>image source: https://jaminfong.cn/neusample/


## Positional encoding

Fourier features let networks learn high frequency functions in low dimensional domains

![image](https://github.com/user-attachments/assets/893c8a81-670a-423d-af4d-148b176fc728)
>image source: https://bmild.github.io/fourfeat/


## Evaluation Metrics

### Peak signal-to-noise ratio (PSNR)
PSNR is commonly used to quantify reconstruction quality for images and video subject to lossy compression. But in NeRF, PSNR is used to compare a training image with a rendered image of the radiance field. The rendered image is taken from the same angle as the training image.

![Screenshot 2025-02-06 at 10 43 39 AM](https://github.com/user-attachments/assets/fe2082b6-ad2c-441a-b78c-27341103b7e1)

where:

* $I(i,j)$ is the rendered image.

* $K(i,j)$ is the training image.

* ${MAX}_I$ is the maximum possible pixel value of the image.


### Structural similarity index measure (SSIM)

Given two windows $x$ and $y$ of size $N \times N$, SSIM is calculated as:

$SSIM = \frac{(2 \mu_x \mu_y + c_1)(2 \sigma_{xy} + c_2)}{(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}$

where:

* $\mu_x$ the pixel sample mean of $x$
* $\mu_y$ the pixel sample mean of $y$
* $\sigma_x^2$ the variance of $x$
* $\sigma_y^2$ the variance of $y$
* $\sigma_{xy}$ the covariance of $x$ and $y$
* $c_1=(k_1L)^2$ , $c_2=(k_2L)^2$ two variables to stabilize the division with weak denominator
* $L$ the dynamic range of the pixel-values (typically this is $2^\text{bits per pixel} - 1$)
* $k_1=0.01$ and $k_2=0.03$

>source: [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio), [SSIM](https://en.wikipedia.org/wiki/Structural_similarity_index_measure)


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


![image](https://github.com/user-attachments/assets/3c02861e-32dc-45e0-b2d4-49c3bcbc0038)

![Screenshot 2025-01-20 at 6 49 40 AM](https://github.com/user-attachments/assets/7540f66a-52c4-4667-a05c-96e6be11d9e8)

