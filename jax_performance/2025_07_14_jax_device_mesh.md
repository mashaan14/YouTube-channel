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

