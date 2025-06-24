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
