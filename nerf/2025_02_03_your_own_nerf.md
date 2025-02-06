# Your own data with NeRF

<iframe width="560" height="315" src="https://www.youtube.com/embed/6RNE155c7iA" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

![2025_02_03_your_own_NeRF](https://github.com/user-attachments/assets/e6385c40-ca3d-4059-9c0f-2ccfaf0db44f)

## Acknowledgement
I used the [notebook](https://lightning.ai/lightning-ai/studios/structure-from-motion-with-meta-s-vgg-sfm?view=public&section=featured) by Andy McSherry for VGGSfM. I also used Google's [MultiNeRF](https://github.com/google-research/multinerf/tree/main) repository for NeRF training. Thanks to Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) for making cloud TPUs available for my use.

## References

```bibtex
@inproceedings{wang2024vggsfm,
 title      = {VGGSfM: Visual Geometry Grounded Deep Structure From Motion},
 author     = {Wang, Jianyuan and Karaev, Nikita and Rupprecht, Christian and Novotny, David},
 booktitle  = {Proceedings of the IEEE/CVF Conference on Computer Vision and 
               Pattern Recognition},
 pages      = {21686--21697},
 year       = {2024}
}
```

```bibtex
@article{barron2022mipnerf360,
 title      = {Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields},
 author     = {Jonathan T. Barron and Ben Mildenhall and
               Dor Verbin and Pratul P. Srinivasan and Peter Hedman},
 journal    = {CVPR},
 year       = {2022}
}
```

## VGGSfM on lightning ai GPUs
```bash
mkdir -p bike/images
ffmpeg -i bike.mp4 -vf "fps=2" bike/images/%04d.jpg 
python vggsfm/demo.py visualize=True SCENE_DIR=./bike resume_ckpt=vggsfm_v2_0_0.bin
```

You can see the output over the port number `8097`. In lightning ai there is a web server extension, but in other cloud providers you need a tunneling service like ngrok.

```bash
conda install imagemagick
```

The following commands use `imagemagick` tool to resize the images. I took these commands from a [script](https://github.com/google-research/multinerf/blob/main/scripts/local_colmap_and_resize.sh) in multinerf repository.
```bash
DATASET_PATH=bike
cp -r "$DATASET_PATH"/images "$DATASET_PATH"/images_2 
pushd "$DATASET_PATH"/images_2 
ls | xargs -P 8 -I {} mogrify -resize 50% {}
popd
cp -r "$DATASET_PATH"/images "$DATASET_PATH"/images_4
pushd "$DATASET_PATH"/images_4
ls | xargs -P 8 -I {} mogrify -resize 25% {}
popd
cp -r "$DATASET_PATH"/images "$DATASET_PATH"/images_8
pushd "$DATASET_PATH"/images_8
ls | xargs -P 8 -I {} mogrify -resize 12.5% {}
popd
```

```bash
conda install zip
zip -r archive.zip ~/bike
```

## Files for NeRF
NeRF expects the folders to be ordered as follows:

```bash
bike
│
└───sparse # colmap output files
│   │   
│   └───0
│       │   points3D.bin
│       │   images.bin
│       │   cameras.bin
│   
└───images # images with the original resolution
│   │   image01.jpg
│   │   image02.jpg
│   │   ...
│
│
└───images_2 # images with 50% of the original resolution
│   │   image01.jpg
│   │   image02.jpg
│   │   ...
│
│
└───images_4 # images with 25% of the original resolution
│   │   image01.jpg
│   │   image02.jpg
│   │   ...
│
│
└───images_8 # images with 12.5% of the original resolution
│    │   image01.jpg
│    │   image02.jpg
│    │   ...
```

## Mip-NeRF 360 on Google cloud TPUs

Copy `bike.zip` to `~/multinerf/nerf_data/` then unzip it:
```bash
mv bike.zip ~/multinerf/nerf_data/
```

```bash
unzip bike.zip
```

### Training
Prepare terminal variables to run the training script:
```bash
SCENE=bike \
EXPERIMENT=360 \
DATA_DIR=/home/mashaan14/multinerf/nerf_data/ \
CHECKPOINT_DIR=/home/mashaan14/multinerf/nerf_results/"$EXPERIMENT"/"$SCENE"
```

Run the training script:
```bash
python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --gin_bindings="Config.checkpoint_every = 25000" \  
  --logtostderr
```

Here's a snippet from the training output:
```bash
 249100/250000: loss=0.00407, psnr=45.627, lr=2.03e-05 | data=0.00386, dist=3.0e-05, inte=0.00017, 112785 r/s
 249200/250000: loss=0.00407, psnr=45.620, lr=2.03e-05 | data=0.00387, dist=3.0e-05, inte=0.00017, 112759 r/s
 249300/250000: loss=0.00407, psnr=45.633, lr=2.03e-05 | data=0.00386, dist=3.0e-05, inte=0.00017, 112771 r/s
 249400/250000: loss=0.00406, psnr=45.651, lr=2.02e-05 | data=0.00386, dist=3.0e-05, inte=0.00017, 112305 r/s
 249500/250000: loss=0.00407, psnr=45.628, lr=2.02e-05 | data=0.00386, dist=3.0e-05, inte=0.00017, 112297 r/s
 249600/250000: loss=0.00407, psnr=45.625, lr=2.01e-05 | data=0.00386, dist=3.0e-05, inte=0.00017, 112332 r/s
 249700/250000: loss=0.00407, psnr=45.613, lr=2.01e-05 | data=0.00387, dist=3.0e-05, inte=0.00017, 112350 r/s
 249800/250000: loss=0.00407, psnr=45.628, lr=2.01e-05 | data=0.00386, dist=3.0e-05, inte=0.00017, 112341 r/s
 249900/250000: loss=0.00407, psnr=45.623, lr=2.00e-05 | data=0.00387, dist=3.0e-05, inte=0.00017, 112345 r/s
 250000/250000: loss=0.00406, psnr=45.654, lr=2.00e-05 | data=0.00386, dist=3.0e-05, inte=0.00017, 109458 r/s
```

### Rendering

prepare the variables:
```bash
SCENE=bike \
EXPERIMENT=360 \
DATA_DIR=/home/mashaan14/multinerf/nerf_data/ \
CHECKPOINT_DIR=/home/mashaan14/multinerf/nerf_results/"$EXPERIMENT"/"$SCENE"
```

and run the rendering script:
```bash
python -m render \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 200" \
  --gin_bindings="Config.render_dir = '${CHECKPOINT_DIR}/render/'" \
  --gin_bindings="Config.render_video_fps = 25" \
  --logtostderr
```

type these links into the downloading window to download them: 
```bash
/home/mashaan14/multinerf/nerf_results/360/bike/render/bike_360_path_renders_step_250000_color.mp4
/home/mashaan14/multinerf/nerf_results/360/bike/render/bike_360_path_renders_step_250000_acc.mp4
/home/mashaan14/multinerf/nerf_results/360/bike/render/bike_360_path_renders_step_250000_distance_mean.mp4
/home/mashaan14/multinerf/nerf_results/360/bike/render/bike_360_path_renders_step_250000_distance_median.mp4
```
