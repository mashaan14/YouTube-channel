## No Mesh
### `batch_size=128`
```console
Step 20, Loss: 2.8329, Elapsed Time: 0.09 seconds
Step 40, Loss: 2.1259, Elapsed Time: 0.11 seconds
Step 60, Loss: 2.0193, Elapsed Time: 0.05 seconds
Step 80, Loss: 1.9852, Elapsed Time: 0.04 seconds
Step 100, Loss: 1.9606, Elapsed Time: 0.04 seconds
Step 120, Loss: 1.8936, Elapsed Time: 0.06 seconds
Step 140, Loss: 1.8879, Elapsed Time: 0.08 seconds
Step 160, Loss: 1.8685, Elapsed Time: 0.09 seconds
Step 180, Loss: 1.8523, Elapsed Time: 0.08 seconds
Step 200, Loss: 1.8214, Elapsed Time: 0.09 seconds
Step 220, Loss: 1.8057, Elapsed Time: 0.07 seconds
Step 240, Loss: 1.7646, Elapsed Time: 0.05 seconds
Step 260, Loss: 1.7519, Elapsed Time: 0.03 seconds
Step 280, Loss: 1.7592, Elapsed Time: 0.04 seconds
Step 300, Loss: 1.7529, Elapsed Time: 0.08 seconds
Step 320, Loss: 1.7421, Elapsed Time: 0.03 seconds
Step 340, Loss: 1.7424, Elapsed Time: 0.06 seconds
Step 360, Loss: 1.7304, Elapsed Time: 0.12 seconds
Step 380, Loss: 1.6941, Elapsed Time: 0.07 seconds
Epoch 1 completed in 96.38 seconds
```

### `batch_size=1024`
```console
Step 20, Loss: 2.5865, Elapsed Time: 0.10 seconds
Step 40, Loss: 2.0469, Elapsed Time: 0.05 seconds
Epoch 1 completed in 86.53 seconds
```

### `batch_size=4096`
```console
XlaRuntimeError: RESOURCE_EXHAUSTED: XLA:TPU compile permanent error. Ran out of memory in memory space hbm. Used 20.93G of 15.48G hbm. Exceeded hbm capacity by 5.45G.

Total hbm usage >= 21.45G:
    reserved        530.01M 
    program          20.93G 
    arguments            0B 
```

## Mesh 4×2
### `batch_size=128`
```console
Step 20, Loss: 3.4531, Elapsed Time: 0.16 seconds
Step 40, Loss: 2.1568, Elapsed Time: 0.06 seconds
Step 60, Loss: 2.0604, Elapsed Time: 0.05 seconds
Step 80, Loss: 2.0316, Elapsed Time: 0.08 seconds
Step 100, Loss: 2.0112, Elapsed Time: 0.08 seconds
Step 120, Loss: 2.0108, Elapsed Time: 0.07 seconds
Step 140, Loss: 1.9691, Elapsed Time: 0.05 seconds
Step 160, Loss: 1.9321, Elapsed Time: 0.08 seconds
Step 180, Loss: 1.9253, Elapsed Time: 0.07 seconds
Step 200, Loss: 1.9450, Elapsed Time: 0.05 seconds
Step 220, Loss: 1.8802, Elapsed Time: 0.05 seconds
Step 240, Loss: 1.8733, Elapsed Time: 0.06 seconds
Step 260, Loss: 1.8219, Elapsed Time: 0.08 seconds
Step 280, Loss: 1.8667, Elapsed Time: 0.07 seconds
Step 300, Loss: 1.8168, Elapsed Time: 0.08 seconds
Step 320, Loss: 1.8108, Elapsed Time: 0.08 seconds
Step 340, Loss: 1.8464, Elapsed Time: 0.07 seconds
Step 360, Loss: 1.8250, Elapsed Time: 0.09 seconds
Step 380, Loss: 1.7758, Elapsed Time: 0.08 seconds
Epoch 1 completed in 122.90 seconds
```

### `batch_size=1024`
```console
Step 20, Loss: 3.1206, Elapsed Time: 0.11 seconds
Step 40, Loss: 2.0948, Elapsed Time: 0.04 seconds
Epoch 1 completed in 112.00 seconds
```

### `batch_size=4096`
```console
Epoch 1 completed in 97.82 seconds
```

## Mesh 8×1
### `batch_size=128`
```console
Step 20, Loss: 3.4434, Elapsed Time: 0.13 seconds
Step 40, Loss: 2.1730, Elapsed Time: 0.08 seconds
Step 60, Loss: 2.0638, Elapsed Time: 0.08 seconds
Step 80, Loss: 2.0338, Elapsed Time: 0.06 seconds
Step 100, Loss: 2.0195, Elapsed Time: 0.05 seconds
Step 120, Loss: 1.9892, Elapsed Time: 0.08 seconds
Step 140, Loss: 1.9572, Elapsed Time: 0.08 seconds
Step 160, Loss: 1.9413, Elapsed Time: 0.09 seconds
Step 180, Loss: 1.9434, Elapsed Time: 0.06 seconds
Step 200, Loss: 1.9122, Elapsed Time: 0.08 seconds
Step 220, Loss: 1.8720, Elapsed Time: 0.08 seconds
Step 240, Loss: 1.8523, Elapsed Time: 0.07 seconds
Step 260, Loss: 1.8232, Elapsed Time: 0.10 seconds
Step 280, Loss: 1.8449, Elapsed Time: 0.07 seconds
Step 300, Loss: 1.8150, Elapsed Time: 0.05 seconds
Step 320, Loss: 1.7715, Elapsed Time: 0.12 seconds
Step 340, Loss: 1.8170, Elapsed Time: 0.08 seconds
Step 360, Loss: 1.8385, Elapsed Time: 0.10 seconds
Step 380, Loss: 1.7933, Elapsed Time: 0.06 seconds
Epoch 1 completed in 134.89 seconds
```

### `batch_size=1024`
```console
Step 20, Loss: 3.1351, Elapsed Time: 0.10 seconds
Step 40, Loss: 2.0930, Elapsed Time: 0.04 seconds
Epoch 1 completed in 118.79 seconds
```

### `batch_size=4096`
```console
Epoch 1 completed in 104.58 seconds
```

## Error
```console
[libprotobuf ERROR external/com_google_protobuf/src/google/protobuf/message_lite.cc:449] tensorflow.profiler.XSpace exceeded maximum protobuf size of 2GB: 2420964887
```

![peak_memory_allocation](https://github.com/user-attachments/assets/6ddc097e-c181-42e8-a101-21d8c82f7222)
