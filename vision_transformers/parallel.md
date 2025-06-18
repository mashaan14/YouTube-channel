## no sharding
```console
Step 2, Loss: 3.595409870147705, Elapsed Time: 13.40 seconds
Step 4, Loss: 4.5290117263793945, Elapsed Time: 0.33 seconds
Step 6, Loss: 3.436495542526245, Elapsed Time: 0.33 seconds
Step 8, Loss: 2.7333998680114746, Elapsed Time: 0.33 seconds
Step 10, Loss: 2.441728115081787, Elapsed Time: 0.34 seconds
Step 12, Loss: 2.345740795135498, Elapsed Time: 0.34 seconds
Step 14, Loss: 2.193572759628296, Elapsed Time: 0.33 seconds
Step 16, Loss: 2.16274094581604, Elapsed Time: 0.33 seconds
Step 18, Loss: 2.162346363067627, Elapsed Time: 0.71 seconds
Step 20, Loss: 2.176154613494873, Elapsed Time: 0.33 seconds
Step 22, Loss: 2.1908531188964844, Elapsed Time: 0.33 seconds
Step 24, Loss: 2.1944799423217773, Elapsed Time: 0.33 seconds
Step 26, Loss: 2.1980717182159424, Elapsed Time: 0.33 seconds
Step 28, Loss: 2.143333911895752, Elapsed Time: 0.33 seconds
```

## sharding
```console
Epoch 1 completed in 110.53 seconds
Epoch 2 completed in 73.25 seconds
Epoch 3 completed in 73.60 seconds
Epoch 4 completed in 72.83 seconds
Epoch 5 completed in 73.35 seconds
Epoch 6 completed in 73.46 seconds
Epoch 7 completed in 72.53 seconds
Epoch 8 completed in 74.09 seconds
Epoch 9 completed in 73.96 seconds

```

```console
Step 2, Loss: 4.734677314758301, Elapsed Time: 24.84 seconds
Step 4, Loss: 6.122281074523926, Elapsed Time: 11.69 seconds
Step 6, Loss: 5.498552322387695, Elapsed Time: 0.36 seconds
Step 8, Loss: 3.5090627670288086, Elapsed Time: 0.37 seconds
Step 10, Loss: 2.668524742126465, Elapsed Time: 0.37 seconds
Step 12, Loss: 2.560586929321289, Elapsed Time: 0.38 seconds
Step 14, Loss: 2.3328449726104736, Elapsed Time: 0.37 seconds
Step 16, Loss: 2.3895015716552734, Elapsed Time: 0.36 seconds
Step 18, Loss: 2.3689160346984863, Elapsed Time: 0.35 seconds
Step 20, Loss: 2.2298121452331543, Elapsed Time: 0.36 seconds
Step 22, Loss: 2.21760892868042, Elapsed Time: 0.35 seconds
Step 24, Loss: 2.238011121749878, Elapsed Time: 0.35 seconds
Step 26, Loss: 2.220581531524658, Elapsed Time: 0.34 seconds
Step 28, Loss: 2.1452765464782715, Elapsed Time: 0.35 seconds
```

```python
class PatchEmbedding(nnx.Module):
    def __init__(self, img_size, patch_size, embed_dim, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # Specify dtype for the linear projection weights
        self.proj = nnx.Linear(patch_size * patch_size * 3, embed_dim, rngs=rngs, dtype=dtype)

    def __call__(self, x):
        # Ensure input to proj is also the desired dtype if not already
        x = x.reshape(x.shape[0], self.num_patches, self.patch_size * self.patch_size * x.shape[-1])
        return self.proj(x)

class MultiHeadSelfAttention(nnx.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.0, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Specify dtype for QKV and output projections
        self.qkv_proj = nnx.Linear(embed_dim, embed_dim * 3, rngs=rngs, dtype=dtype)
        self.out_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs, dtype=dtype)
        self.dropout = nnx.Dropout(rate=dropout_rate)
        self.dtype = dtype # Store dtype for potential internal casting if needed (e.g., attention scores)

    def __call__(self, x, training: bool):
        batch_size, seq_len, _ = x.shape
        # Input to qkv_proj should also be cast to self.dtype if necessary
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        queries, keys, values = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        queries = jnp.transpose(queries, (0, 2, 1, 3))
        keys = jnp.transpose(keys, (0, 2, 1, 3))
        values = jnp.transpose(values, (0, 2, 1, 3))

        # Explicit casting for attention score computation can be helpful for mixed precision
        attention_scores = (queries @ jnp.swapaxes(keys, -1, -2)) / jnp.sqrt(self.head_dim).astype(self.dtype)
        attention_weights = nnx.softmax(attention_scores.astype(self.dtype), axis=-1) # Ensure softmax operates on correct dtype
        attention_weights = self.dropout(attention_weights, training=training)

        output = (attention_weights @ values).swapaxes(1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(output)

class MLPBlock(nnx.Module):
    def __init__(self, embed_dim, mlp_dim, dropout_rate=0.0, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.fc1 = nnx.Linear(embed_dim, mlp_dim, rngs=rngs, dtype=dtype)
        self.gelu = nnx.activations.gelu
        self.dropout1 = nnx.Dropout(rate=dropout_rate)
        self.fc2 = nnx.Linear(mlp_dim, embed_dim, rngs=rngs, dtype=dtype)
        self.dropout2 = nnx.Dropout(rate=dropout_rate)

    def __call__(self, x, training: bool):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        return x

class EncoderBlock(nnx.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate=0.0, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        # LayerNorm often doesn't need explicit dtype unless for very specific cases
        # It typically operates on the input's dtype.
        self.norm1 = nnx.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate, rngs=rngs, dtype=dtype)
        self.norm2 = nnx.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_dim, dropout_rate, rngs=rngs, dtype=dtype)

    def __call__(self, x, training: bool):
        # LayerNorm outputs typically match input dtype, but you can explicitly cast before attn/mlp
        x = x + self.attn(self.norm1(x), training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        return x

class VisionTransformer(nnx.Module):
    def __init__(self, img_size, patch_size, num_classes, embed_dim, num_layers, num_heads, mlp_dim, dropout_rate=0.0, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32):
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim, rngs=rngs, dtype=dtype)
        num_patches = (img_size // patch_size) ** 2
        
        # Specify dtype for learnable embeddings
        self.cls_token = nnx.Param(jax.random.normal(rngs.params, (1, 1, embed_dim), dtype=dtype))
        self.pos_embed = nnx.Param(jax.random.normal(rngs.params, (1, num_patches + 1, embed_dim), dtype=dtype))

        self.encoder_blocks = [
            EncoderBlock(embed_dim, num_heads, mlp_dim, dropout_rate, rngs=rngs, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.norm = nnx.LayerNorm(embed_dim) # LayerNorm also has a dtype argument but often not needed explicitly
        self.head = nnx.Linear(embed_dim, num_classes, rngs=rngs, dtype=dtype)
        self.dtype = dtype # Store the global dtype for the model

    def __call__(self, x, training: bool):
        # It's crucial to cast your input data to the desired dtype
        x = x.astype(self.dtype)

        x = self.patch_embed(x)
        batch_size = x.shape[0]
        
        cls_tokens = jnp.tile(self.cls_token.value, (batch_size, 1, 1))
        x = jnp.concatenate((cls_tokens, x), axis=1)
        x = x + self.pos_embed.value # Positional embeddings are already the correct dtype

        for block in self.encoder_blocks:
            x = block(x, training=training)

        cls_output = self.norm(x[:, 0])
        return self.head(cls_output)
```
