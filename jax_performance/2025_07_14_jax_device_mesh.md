# Parallel Vision Transformer using JAX Device Mesh

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/G6c6zk0RhRM" frameborder="0" allowfullscreen></iframe>
</div>



## Acknowledgment
These resources were helpful in preparing this post:
  - [Train a miniGPT language model with JAX](https://docs.jaxstack.ai/en/latest/JAX_for_LLM_pretraining.html)
  - [Train a Vision Transformer (ViT) for image classification with JAX](https://docs.jaxstack.ai/en/latest/JAX_Vision_transformer.html)

## References
```bibtex
@misc{kipf2017semisupervised,
    title         = {Semi-Supervised Classification with Graph Convolutional Networks},
    author        = {Thomas N. Kipf and Max Welling},
    year          = {2017},
    eprint        = {1609.02907},
    archivePrefix = {arXiv},
    primaryClass  = {cs.LG}
}
```

## Epoch run time
![epoch_run_time](https://github.com/user-attachments/assets/3f316f0a-bef7-4d9e-83a2-33dae47c7e06)

## Peak memory allocation
![peak_memory_allocation](https://github.com/user-attachments/assets/d9d881b8-c38c-4878-9897-82d83c914e1d)

## Installing TensorBoard

I had issues visualizing TensorBoard reports in Kaggle notebooks and colab. Also, I was afraid of losing TensorBoard reports when the runtime restarts, which usually happens unexpectedly in colab. So, I had to install TensorBoard loacally and store all reports in one folder and visualize them. Here are the commands to install TensorBoard with a dedicated conda environment:

```bash
conda create -n tf_env
conda activate tf_env
conda install tensorboard
pip install -U tensorboard-plugin-profile
```

```bash
tensorboard --version
```

```console
2.19.0
```

After running `jax.profiler.start_trace(log_dir)` `jax.profiler.stop_trace()`, TensorBoard creates two files `.trace.json.gz` and `.xplane.pb`. For example if `log_dir = "./jax_profile_logs"`, the directory structure will be:

```
jax_profile_logs/
└── plugins/
    └── profile/
        └── <timestamp>/
            ├── .trace.json.gz
            ├── .xplane.pb
```

I downloaded these two files and place them in a similar directory structure in my Downloads folder. I ran 8 experiments, so I got 8 folders in total:

```
/
└── Users/
    └── mashaanalshammari/
        └── Downloads/
            └── plugins/
                └── profile/
                    ├── mesh_4_2_batch_128/
                    │   ├── .trace.json.gz
                    │   └── .xplane.pb
                    ├── mesh_4_2_batch_1024/
                    │   ├── .trace.json.gz
                    │   └── .xplane.pb
                    ├── mesh_4_2_batch_4096/
                    │   ├── .trace.json.gz
                    │   └── .xplane.pb
                    ├── mesh_8_1_batch_128/
                    │   ├── .trace.json.gz
                    │   └── .xplane.pb
                    ├── mesh_8_1_batch_1024/
                    │   ├── .trace.json.gz
                    │   └── .xplane.pb
                    ├── mesh_8_1_batch_4096/
                    │   ├── .trace.json.gz
                    │   └── .xplane.pb
                    ├── mesh_none_batch_128/
                    │   ├── .trace.json.gz
                    │   └── .xplane.pb
                    └── mesh_none_batch_1024/
                        ├── .trace.json.gz
                        └── .xplane.pb
```

Then run this command from the terminal to open TensorBoard:

```bash
tensorboard --logdir=/Users/mashaanalshammari/Downloads/
```

Here is a screenshot of the memory viewer, you can pick a profiler file from the dropdown menu.

![Screenshot 2025-06-21 at 12 44 35 AM](https://github.com/user-attachments/assets/03e2e861-ddc0-46e3-9121-cec6d1cc65d2)




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
