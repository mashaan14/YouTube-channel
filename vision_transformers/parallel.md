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

## sharding (4,2)
```console
Step 50, Loss: 2.0222089290618896, Elapsed Time: 0.10 seconds
Step 100, Loss: 1.9536722898483276, Elapsed Time: 0.12 seconds
Step 150, Loss: 1.8822193145751953, Elapsed Time: 0.06 seconds
Step 200, Loss: 1.860122799873352, Elapsed Time: 0.10 seconds
Step 250, Loss: 1.7819035053253174, Elapsed Time: 0.07 seconds
Step 300, Loss: 1.8086696863174438, Elapsed Time: 0.08 seconds
Step 350, Loss: 1.775278925895691, Elapsed Time: 0.07 seconds
Epoch 1 completed in 87.38 seconds
Step 50, Loss: 1.7539153099060059, Elapsed Time: 0.05 seconds
Step 100, Loss: 1.694887638092041, Elapsed Time: 0.08 seconds
Step 150, Loss: 1.7047514915466309, Elapsed Time: 0.10 seconds
Step 200, Loss: 1.675998330116272, Elapsed Time: 0.07 seconds
Step 250, Loss: 1.6657073497772217, Elapsed Time: 0.06 seconds
Step 300, Loss: 1.6906852722167969, Elapsed Time: 0.06 seconds
Step 350, Loss: 1.6561832427978516, Elapsed Time: 0.09 seconds
Epoch 2 completed in 87.30 seconds
Step 50, Loss: 1.657387137413025, Elapsed Time: 0.07 seconds
Step 100, Loss: 1.6667324304580688, Elapsed Time: 0.06 seconds
Step 150, Loss: 1.656468152999878, Elapsed Time: 0.07 seconds
Step 200, Loss: 1.679901123046875, Elapsed Time: 0.09 seconds
Step 250, Loss: 1.652742862701416, Elapsed Time: 0.06 seconds
Step 300, Loss: 1.674971103668213, Elapsed Time: 0.07 seconds
Step 350, Loss: 1.686550259590149, Elapsed Time: 0.08 seconds
Epoch 3 completed in 86.83 seconds
Step 50, Loss: 1.6657437086105347, Elapsed Time: 0.09 seconds
Step 100, Loss: 1.6324213743209839, Elapsed Time: 0.07 seconds
Step 150, Loss: 1.6472724676132202, Elapsed Time: 0.10 seconds
Step 200, Loss: 1.6728358268737793, Elapsed Time: 0.08 seconds
Step 250, Loss: 1.6434122323989868, Elapsed Time: 0.05 seconds
Step 300, Loss: 1.642844557762146, Elapsed Time: 0.07 seconds
Step 350, Loss: 1.6884466409683228, Elapsed Time: 0.09 seconds
Epoch 4 completed in 87.20 seconds
Step 50, Loss: 1.7139999866485596, Elapsed Time: 0.10 seconds
Step 100, Loss: 1.6662925481796265, Elapsed Time: 0.07 seconds
Step 150, Loss: 1.6493831872940063, Elapsed Time: 0.07 seconds
Step 200, Loss: 1.7121976613998413, Elapsed Time: 0.08 seconds
Step 250, Loss: 1.7038582563400269, Elapsed Time: 0.07 seconds
Step 300, Loss: 1.6735692024230957, Elapsed Time: 0.08 seconds
Step 350, Loss: 1.6960369348526, Elapsed Time: 0.09 seconds
Epoch 5 completed in 87.34 seconds
```

## Error
```console
[libprotobuf ERROR external/com_google_protobuf/src/google/protobuf/message_lite.cc:449] tensorflow.profiler.XSpace exceeded maximum protobuf size of 2GB: 2420964887
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
