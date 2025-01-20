# Running nerfacto algorithm on lightning ai GPUs

![image](https://github.com/user-attachments/assets/7dedf2e6-e277-4137-b871-e67dd82fd1e1)


## Basic commands

```bash
pwd              # print the current directory
```
```bash
ls               # list all files and folders
```
```bash
nvidia-smi       # check GPU status
```

## Installing nerfstudio
Most of these commands were taken from nerfstudio [installation page](https://docs.nerf.studio/quickstart/installation.html). I found putting them in this order works on lightning ai GPUs.


```bash
pip uninstall torch torchvision functorch tinycudann
```

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit 
```

```bash
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

```bash
pip install nerfstudio
```

## Download the dataset
This will download a dataset with the following location:
https://drive.google.com/file/d/1FceQ5DX7bbTbHeL26t0x6ku56cwsRs6t/view?usp=sharing

The folders are:

* `poster/images` images with the original resolution
* `poster/images2` images with 50% of the original resolution
* `poster/images4` images with 25% of the original resolution
* `poster/images8` images with 12.5% of the original resolution
* `poster/colmap/sparse/0` colmap output files
* `poster/transforms.json` holds the transform matrices for all images


```bash
ns-download-data nerfstudio --capture-name=poster
```

## Training nerfacto algorithm
This command trains the nerfacto algorithm with default parameters:
```bash
ns-train nerfacto --data data/nerfstudio/poster  
```

The output looks like this:
```bash
Step (% Done)                                                                                        
--------------------                                                                                 
29910 (99.70%)      40.633 ms            3 s, 656.933 ms      103.17 K                               
29920 (99.73%)      40.605 ms            3 s, 248.366 ms      103.20 K                               
29930 (99.77%)      41.887 ms            2 s, 932.074 ms      100.67 K                               
29940 (99.80%)      40.978 ms            2 s, 458.696 ms      102.32 K                               
29950 (99.83%)      41.026 ms            2 s, 51.290 ms       102.18 K                               
29960 (99.87%)      42.086 ms            1 s, 683.453 ms      100.18 K                               
29970 (99.90%)      40.978 ms            1 s, 229.333 ms      102.25 K                               
29980 (99.93%)      41.012 ms            820.242 ms           102.19 K                               
29990 (99.97%)      41.987 ms            419.868 ms           100.42 K                               
29999 (100.00%)                                                                                      
---------------------------------------------------------------------------------------------------- 
Viewer running locally at: http://localhost:7007 (listening on 0.0.0.0)                              
╭─────────────────────────────── 🎉 Training Finished 🎉 ────────────────────────────────╮
│                        ╷                                                               │
│   Config File          │ outputs/poster/nerfacto/2024-12-15_032229/config.yml          │
│   Checkpoint Directory │ outputs/poster/nerfacto/2024-12-15_032229/nerfstudio_models   │
│                        ╵                                                               │
╰────────────────────────────────────────────────────────────────────────────────────────╯
                                                   Use ctrl+c to quit                                
```

## Openning web viewer
The web viewer runs on `http://0.0.0.0:7007`. If you're running nerfstudio locally, it is just a matter of opening the web browser. But if you're on the cloud, that could be tricky. Lightning AI provides a web port with the VM, but other cloud providers don't. For that you need another service, [ngrok](https://ngrok.com/) for example.

Use this command to load a specific checkpoint to the viewer
```bash
ns-viewer --load-config outputs/poster/nerfacto/2024-12-15_032229/config.yml
```

## Render a video
Make sure that you have `ffmpeg` installed. I installed using conda, but it could be different depending on the machine setup.
```bash
conda install ffmpeg
```

You can render a video using `ns-render camera-path` where you manually pick camera poses that will then be saved to `cameras.json`. But if you just want a quick video, just use `ns-render interpolate`:

```bash
ns-render interpolate \
 --load-config outputs/poster/nerfacto/2024-12-15_032229/config.yml \
 --output-path renders/output.mp4
```

Here's a complete list of `ns-render` options:
```bash
usage: ns-render [-h] {camera-path,interpolate,spiral,dataset}

╭─ options ────────────────────────────────────────────────────────────────────────────────────────╮
│ -h, --help        show this help message and exit                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ subcommands ────────────────────────────────────────────────────────────────────────────────────╮
│ {camera-path,interpolate,spiral,dataset}                                                         │
│     camera-path   Render a camera path generated by the viewer or blender add-on.                │
│     interpolate   Render a trajectory that interpolates between training or eval dataset images. │
│     spiral        Render a spiral trajectory (often not great).                                  │
│     dataset       Render all images in the dataset.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```
     
