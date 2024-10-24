```bash
!nvidia-smi
```

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

```bash
ns-download-data nerfstudio --capture-name=poster 
```

```bash
ns-train nerfacto --data data/nerfstudio/poster  
```

```bash
ns-viewer --load-config outputs/poster/nerfacto/2024-10-08_051621/config.yml
```
