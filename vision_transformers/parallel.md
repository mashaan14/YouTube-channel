## No Mesh
### `batch_size=128`
```console
Step 20, Loss: 2.8329226970672607, Elapsed Time: 0.09 seconds
Step 40, Loss: 2.125962257385254, Elapsed Time: 0.11 seconds
Step 60, Loss: 2.019329786300659, Elapsed Time: 0.05 seconds
Step 80, Loss: 1.98526132106781, Elapsed Time: 0.04 seconds
Step 100, Loss: 1.9606808423995972, Elapsed Time: 0.04 seconds
Step 120, Loss: 1.8936665058135986, Elapsed Time: 0.06 seconds
Step 140, Loss: 1.8879919052124023, Elapsed Time: 0.08 seconds
Step 160, Loss: 1.868575096130371, Elapsed Time: 0.09 seconds
Step 180, Loss: 1.852310061454773, Elapsed Time: 0.08 seconds
Step 200, Loss: 1.8214492797851562, Elapsed Time: 0.09 seconds
Step 220, Loss: 1.8057546615600586, Elapsed Time: 0.07 seconds
Step 240, Loss: 1.7646404504776, Elapsed Time: 0.05 seconds
Step 260, Loss: 1.75190007686615, Elapsed Time: 0.03 seconds
Step 280, Loss: 1.7592840194702148, Elapsed Time: 0.04 seconds
Step 300, Loss: 1.7529516220092773, Elapsed Time: 0.08 seconds
Step 320, Loss: 1.7421925067901611, Elapsed Time: 0.03 seconds
Step 340, Loss: 1.7424205541610718, Elapsed Time: 0.06 seconds
Step 360, Loss: 1.7304487228393555, Elapsed Time: 0.12 seconds
Step 380, Loss: 1.6941728591918945, Elapsed Time: 0.07 seconds
Epoch 1 completed in 96.38 seconds
```

### `batch_size=1024`
```console
Step 20, Loss: 2.5865652561187744, Elapsed Time: 0.10 seconds
Step 40, Loss: 2.0469934940338135, Elapsed Time: 0.05 seconds
Epoch 1 completed in 86.53 seconds
```

## Mesh 4×2
### `batch_size=128`
```console
Step 20, Loss: 3.453094244003296, Elapsed Time: 0.16 seconds
Step 40, Loss: 2.156808614730835, Elapsed Time: 0.06 seconds
Step 60, Loss: 2.0604236125946045, Elapsed Time: 0.05 seconds
Step 80, Loss: 2.031639337539673, Elapsed Time: 0.08 seconds
Step 100, Loss: 2.0111751556396484, Elapsed Time: 0.08 seconds
Step 120, Loss: 2.0107836723327637, Elapsed Time: 0.07 seconds
Step 140, Loss: 1.969140648841858, Elapsed Time: 0.05 seconds
Step 160, Loss: 1.932132363319397, Elapsed Time: 0.08 seconds
Step 180, Loss: 1.9253463745117188, Elapsed Time: 0.07 seconds
Step 200, Loss: 1.9450167417526245, Elapsed Time: 0.05 seconds
Step 220, Loss: 1.8802388906478882, Elapsed Time: 0.05 seconds
Step 240, Loss: 1.8732833862304688, Elapsed Time: 0.06 seconds
Step 260, Loss: 1.8219003677368164, Elapsed Time: 0.08 seconds
Step 280, Loss: 1.866715669631958, Elapsed Time: 0.07 seconds
Step 300, Loss: 1.8167908191680908, Elapsed Time: 0.08 seconds
Step 320, Loss: 1.8108484745025635, Elapsed Time: 0.08 seconds
Step 340, Loss: 1.846405029296875, Elapsed Time: 0.07 seconds
Step 360, Loss: 1.8250162601470947, Elapsed Time: 0.09 seconds
Step 380, Loss: 1.7758228778839111, Elapsed Time: 0.08 seconds
Epoch 1 completed in 122.90 seconds
```

### `batch_size=1024`
```console
Step 20, Loss: 3.1206886768341064, Elapsed Time: 0.11 seconds
Step 40, Loss: 2.0948102474212646, Elapsed Time: 0.04 seconds
Epoch 1 completed in 112.00 seconds
```

## Mesh 8×1
### `batch_size=128`
```console
Step 20, Loss: 3.4433820247650146, Elapsed Time: 0.13 seconds
Step 40, Loss: 2.1730141639709473, Elapsed Time: 0.08 seconds
Step 60, Loss: 2.0638296604156494, Elapsed Time: 0.08 seconds
Step 80, Loss: 2.033846616744995, Elapsed Time: 0.06 seconds
Step 100, Loss: 2.0194597244262695, Elapsed Time: 0.05 seconds
Step 120, Loss: 1.989200234413147, Elapsed Time: 0.08 seconds
Step 140, Loss: 1.957165002822876, Elapsed Time: 0.08 seconds
Step 160, Loss: 1.9412504434585571, Elapsed Time: 0.09 seconds
Step 180, Loss: 1.9434207677841187, Elapsed Time: 0.06 seconds
Step 200, Loss: 1.9121586084365845, Elapsed Time: 0.08 seconds
Step 220, Loss: 1.8719624280929565, Elapsed Time: 0.08 seconds
Step 240, Loss: 1.8523155450820923, Elapsed Time: 0.07 seconds
Step 260, Loss: 1.8232351541519165, Elapsed Time: 0.10 seconds
Step 280, Loss: 1.8448708057403564, Elapsed Time: 0.07 seconds
Step 300, Loss: 1.8149839639663696, Elapsed Time: 0.05 seconds
Step 320, Loss: 1.7714861631393433, Elapsed Time: 0.12 seconds
Step 340, Loss: 1.8169745206832886, Elapsed Time: 0.08 seconds
Step 360, Loss: 1.838454246520996, Elapsed Time: 0.10 seconds
Step 380, Loss: 1.7932828664779663, Elapsed Time: 0.06 seconds
Epoch 1 completed in 134.89 seconds
```

### `batch_size=1024`
```console
Step 20, Loss: 3.1351163387298584, Elapsed Time: 0.10 seconds
Step 40, Loss: 2.0930397510528564, Elapsed Time: 0.04 seconds
Epoch 1 completed in 118.79 seconds
```

## Error
```console
[libprotobuf ERROR external/com_google_protobuf/src/google/protobuf/message_lite.cc:449] tensorflow.profiler.XSpace exceeded maximum protobuf size of 2GB: 2420964887
```

![peak_memory_allocation](https://github.com/user-attachments/assets/6ddc097e-c181-42e8-a101-21d8c82f7222)
