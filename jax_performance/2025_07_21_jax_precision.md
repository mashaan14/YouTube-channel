```python
dtype=jnp.float32,          # Data type
param_dtype=jnp.float32,    # Parameter data type
```

```console
Total Parameters: 10,670,604 (42.7 MB)
```

---

```python
dtype=jnp.bfloat16,          # Data type
param_dtype=jnp.bfloat16,    # Parameter data type
```

```console
Total Parameters: 10,670,604 (21.4 MB)
```

---

```python
dtype=jnp.float16,          # Data type
param_dtype=jnp.float16,    # Parameter data type
```

```console
Total Parameters: 10,670,604 (21.4 MB)
```

---

```console
XlaRuntimeError: UNIMPLEMENTED: Dot algorithm ALG_DOT_F16_F16_F32 is not supported.
```

---

```python
dtype=jnp.float32,          # Data type
```

```console
Total Parameters: 10,695,564 (42.8 MB) 
```

---

```python
dtype=jnp.bfloat16,          # Data type
```

```console
Total Parameters: 10,695,564 (42.8 MB) 
```

---

```python
dtype=jnp.float16,          # Data type
```

```console
Total Parameters: 10,695,564 (42.8 MB)
```

## bfloat16

```console
Step 20, Loss: 3.958667516708374, Elapsed Time: 0.11 seconds
Step 40, Loss: 2.3608651161193848, Elapsed Time: 0.08 seconds
Step 60, Loss: 2.1720101833343506, Elapsed Time: 0.10 seconds
Step 80, Loss: 2.0653440952301025, Elapsed Time: 0.05 seconds
Step 100, Loss: 1.9823248386383057, Elapsed Time: 0.07 seconds
Step 120, Loss: 1.9301824569702148, Elapsed Time: 0.05 seconds
Step 140, Loss: 1.9318405389785767, Elapsed Time: 0.09 seconds
Step 160, Loss: 1.8989524841308594, Elapsed Time: 0.09 seconds
Step 180, Loss: 1.9052046537399292, Elapsed Time: 0.09 seconds
Step 200, Loss: 1.8983386754989624, Elapsed Time: 0.08 seconds
Step 220, Loss: 1.8348573446273804, Elapsed Time: 0.09 seconds
Step 240, Loss: 1.8322759866714478, Elapsed Time: 0.07 seconds
Step 260, Loss: 1.8048137426376343, Elapsed Time: 0.08 seconds
Step 280, Loss: 1.809252142906189, Elapsed Time: 0.08 seconds
Step 300, Loss: 1.796458125114441, Elapsed Time: 0.09 seconds
Step 320, Loss: 1.7766698598861694, Elapsed Time: 0.10 seconds
Step 340, Loss: 1.745721697807312, Elapsed Time: 0.08 seconds
Step 360, Loss: 1.7272859811782837, Elapsed Time: 0.07 seconds
Step 380, Loss: 1.713021159172058, Elapsed Time: 0.10 seconds
Epoch 1 completed in 157.41 seconds
```
