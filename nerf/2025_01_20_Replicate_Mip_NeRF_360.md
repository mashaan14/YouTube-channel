# Replicate Mip-NeRF 360

![Screenshot 2025-01-20 at 6 36 39 AM](https://github.com/user-attachments/assets/8644b4a5-9585-4a82-a12a-c2470c4fea63)

## YouTube:
I explained these instructions in a [YouTube video](https://youtu.be/5aQpIiNohDA).

## Acknowledgment:
Thanks to Google's TPU Research Cloud (TRC) for making cloud TPUs available for my use. For better explaination, watch [Jon Barron's video](https://youtu.be/zBSH-k9GbV4?si=aZonNYqVJcTLjBFG) talking about Mip-NeRF 360.

## References:
```bibtex
@article{barron2022mipnerf360,
 title    = {Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields},
 author   = {Jonathan T. Barron and Ben Mildenhall and
            Dor Verbin and Pratul P. Srinivasan and Peter Hedman},
 journal  = {CVPR},
 year     = {2022}
}
```

```bibtex
@misc{multinerf2022,
 title  = {MultiNeRF: A Code Release for Mip-NeRF 360, Ref-NeRF, and RawNeRF},
 author = {Ben Mildenhall and Dor Verbin and Pratul P. Srinivasan and Peter Hedman and Ricardo Martin-Brualla and Jonathan T. Barron},
 year   = {2022},
 url    = {https://github.com/google-research/multinerf},
}
```

## Create a TPU VM

Head to google cloud and create a TPU instance. You can apply for [TPU Research Cloud](https://sites.research.google/trc/about/) to have free TPUs for research. Check my video on how to create a TPU instance under TPU Research Cloud:

[TPU VM on Google Cloud](https://youtu.be/PwYHoiB4Fag)

## Install Conda
Download the installer
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Run the installation file and follow the instructions
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

Add the Miniconda directory to the shell PATH environment variable.
```bash
export PATH="/home/mashaan14/miniconda3/bin:$PATH"
```

Make a conda environment.

```bash
conda create --name multinerf python=3.9
```

restart the shell, then run:

```bash
conda activate multinerf
```

## Install jax for TPU
You can install jax by typing in this command in your VM terminal:
```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Verify that JAX can access the TPU and can run basic operations:
```bash
python3
```
```python
import jax
```

Display the number of TPU cores available:
```python
jax.device_count()
```
The number of TPU cores is displayed. If you are using a v4 TPU, this should be `4`. If you are using a v2 or v3 TPU, this should be `8`.

## Clone the repo
```bash
git clone https://github.com/google-research/multinerf.git
cd multinerf
```

## Install requirements
Prepare pip
```bash
conda install pip
pip install --upgrade pip
```

Install requirements
```bash
pip install -r requirements.txt
```

install opencv
```bash
pip3 install opencv-python-headless
```

install ffmpeg
```bash
conda install ffmpeg
```

Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
```bash
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap
```

## Download the dataset
Create a new directory inside multinerf folder:
```bash
mkdir nerf_data
cd nerf_data
```

### Nerf example data
```bash
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
```

```bash
unzip nerf_example_data.zip
```

### Mip-NeRF 360 real dataset
```bash
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
```
```bash
unzip 360_v2.zip
```

## Train on Mip-NeRF 360 real dataset

```bash
SCENE=stump \
EXPERIMENT=360 \
DATA_DIR=/home/mashaan14/multinerf/nerf_data/ \
CHECKPOINT_DIR=/home/mashaan14/multinerf/nerf_results/"$EXPERIMENT"/"$SCENE"
```

```bash
python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --gin_bindings="Config.checkpoint_every = 25000" \  
  --logtostderr
```

## Conversion of out-of-bound Python integers

If you run the training script, you might get this error:

```bash
OverflowError: Python integer -1 out of bounds for uint64
```

![Screenshot 2024-11-13 140902](https://github.com/user-attachments/assets/173c807e-c8ac-43a1-a424-3744c97fb308)

That's because `scene_manager.py` under `internal/pycolmap` uses this statment:

```python
INVALID_POINT3D = np.uint64(-1)
```

Passing a negative number to an unsigned integer gives the maximum value for that datatype. This was [deprecated by numpy](https://numpy.org/devdocs/release/1.24.0-notes.html).

Just replace that command in `scene_manager.py` with:
```python
np.array(-1).astype(np.uint64)
```

You can run both statments below to see colab will throw a warning if you pass a negative number to an unsigned integer.

![Screenshot 2024-11-13 140820](https://github.com/user-attachments/assets/ba76d448-0e42-40ac-be14-c5008234f062)

## Render the scene

```bash
SCENE=stump \
EXPERIMENT=360 \
DATA_DIR=/home/mashaan14/multinerf/nerf_data/ \
CHECKPOINT_DIR=/home/mashaan14/multinerf/nerf_results/"$EXPERIMENT"/"$SCENE"
```

```bash
python -m render \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 50" \
  --gin_bindings="Config.render_dir = '${CHECKPOINT_DIR}/render/'" \
  --gin_bindings="Config.render_video_fps = 5" \
  --logtostderr
```

Now, download the following files:
```bash
/home/mashaan14/multinerf/nerf_results/360/stump/render/stump_360_path_renders_step_250000_color.mp4
/home/mashaan14/multinerf/nerf_results/360/stump/render/stump_360_path_renders_step_250000_acc.mp4
/home/mashaan14/multinerf/nerf_results/360/stump/render/stump_360_path_renders_step_250000_distance_mean.mp4
```
