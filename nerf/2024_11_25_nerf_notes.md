# NeRF: Neural Radiance Fields

![image](https://github.com/user-attachments/assets/3d5f95a6-f7bb-4fc3-b201-e941774ca096)


## YouTube:
I explained this notebook in a [YouTube video](https://youtu.be/kszswpg7sjs).

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
title   = {{JaxNeRF}: an efficient {JAX} implementation of {NeRF}},
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

$MSE = \frac{1}{mn} \sum\limits^{i=0}_{m-1} \sum\limits^{j=0}_{n-1} \big[ I(i,j) - K(i,j) \big]^2$

\\

$PSNR = 10 \cdot {log}_{10} \Big( \frac{{MAX}^2_I}{MSE} \Big)$

$\quad \quad \quad \  = 20 \cdot {log}_{10} \Big( \frac{{MAX}_I}{\sqrt{MSE}} \Big)$

$\quad \quad \quad \  = 20 \cdot {log}_{10} ({MAX}_I) - 10 \cdot {log}_{10} (MSE)$

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

![image](https://github.com/user-attachments/assets/3c02861e-32dc-45e0-b2d4-49c3bcbc0038)

![Screenshot 2025-01-20 at 6 49 40 AM](https://github.com/user-attachments/assets/7540f66a-52c4-4667-a05c-96e6be11d9e8)

