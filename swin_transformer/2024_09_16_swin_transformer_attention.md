# Swin Transformer Attention

<head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
</head>

<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">
  <iframe style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" src="https://www.youtube.com/embed/mtCTIGgzfbc" frameborder="0" allowfullscreen></iframe>
</div>

## Acknowledgment:
I borrowed some code from [Swin Transformer github repository](https://github.com/microsoft/Swin-Transformer/tree/main).

## References:
```bibtex
@inproceedings{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

## Swin Transformer classes and helper functions
To run the notebook, I copied classes and functions from:

*   [https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py)
*   [https://github.com/microsoft/Swin-Transformer/blob/main/models/build.py](https://github.com/microsoft/Swin-Transformer/blob/main/models/build.py)
*   [https://github.com/microsoft/Swin-Transformer/blob/main/main.py](https://github.com/microsoft/Swin-Transformer/blob/main/main.py)

To save space in this post, I'm only showing the function/class name. 

```python
class Mlp(nn.Module):

def window_partition(x, window_size):

def window_reverse(windows, window_size, H, W):

class WindowAttention(nn.Module):

class SwinTransformerBlock(nn.Module):

class PatchMerging(nn.Module):

class BasicLayer(nn.Module):

class PatchEmbed(nn.Module):

class SwinTransformer(nn.Module):
```

## Initialize the model

```python
model = SwinTransformer(img_size=96,
                          patch_size=4,
                          in_chans=3,
                          num_classes=10,
                          embed_dim=48,
                          depths=[2, 2, 6, 2],
                          num_heads=[3, 6, 12, 24],
                          window_size=6,
                          mlp_ratio=4,
                          qkv_bias=True,
                          qk_scale=None,
                          drop_rate=0.0,
                          drop_path_rate=0.1,
                          ape=False,
                          norm_layer=nn.LayerNorm,
                          patch_norm=True,
                          use_checkpoint=False,
                          fused_window_process=False)

# Transfer to GPU
model.to(device)
# setup the loss function
criterion = nn.CrossEntropyLoss()
# setup the optimizer with the learning rate
model_optimizer = optim.AdamW(model.parameters(), lr=5e-4)
```

```bash
model
```

```console
SwinTransformer(
  (patch_embed): PatchEmbed(
    (proj): Conv2d(3, 48, kernel_size=(4, 4), stride=(4, 4))
    (norm): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
  )
  (pos_drop): Dropout(p=0.0, inplace=False)
  (layers): ModuleList(
    (0): BasicLayer(
      dim=48, input_resolution=(24, 24), depth=2
      (blocks): ModuleList(
        (0): SwinTransformerBlock(
          dim=48, input_resolution=(24, 24), num_heads=3, window_size=6, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=48, window_size=(6, 6), num_heads=3
            (qkv): Linear(in_features=48, out_features=144, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=48, out_features=48, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): Identity()
          (norm2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=48, out_features=192, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=192, out_features=48, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (1): SwinTransformerBlock(
          dim=48, input_resolution=(24, 24), num_heads=3, window_size=6, shift_size=3, mlp_ratio=4
          (norm1): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=48, window_size=(6, 6), num_heads=3
            (qkv): Linear(in_features=48, out_features=144, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=48, out_features=48, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.009)
          (norm2): LayerNorm((48,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=48, out_features=192, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=192, out_features=48, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (downsample): PatchMerging(
        input_resolution=(24, 24), dim=48
        (reduction): Linear(in_features=192, out_features=96, bias=False)
        (norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
      )
    )
    (1): BasicLayer(
      dim=96, input_resolution=(12, 12), depth=2
      (blocks): ModuleList(
        (0): SwinTransformerBlock(
          dim=96, input_resolution=(12, 12), num_heads=6, window_size=6, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=96, window_size=(6, 6), num_heads=6
            (qkv): Linear(in_features=96, out_features=288, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=96, out_features=96, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.018)
          (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=96, out_features=384, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=384, out_features=96, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (1): SwinTransformerBlock(
          dim=96, input_resolution=(12, 12), num_heads=6, window_size=6, shift_size=3, mlp_ratio=4
          (norm1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=96, window_size=(6, 6), num_heads=6
            (qkv): Linear(in_features=96, out_features=288, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=96, out_features=96, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.027)
          (norm2): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=96, out_features=384, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=384, out_features=96, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (downsample): PatchMerging(
        input_resolution=(12, 12), dim=96
        (reduction): Linear(in_features=384, out_features=192, bias=False)
        (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
      )
    )
    (2): BasicLayer(
      dim=192, input_resolution=(6, 6), depth=6
      (blocks): ModuleList(
        (0): SwinTransformerBlock(
          dim=192, input_resolution=(6, 6), num_heads=12, window_size=6, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=192, window_size=(6, 6), num_heads=12
            (qkv): Linear(in_features=192, out_features=576, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=192, out_features=192, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.036)
          (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=192, out_features=768, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=768, out_features=192, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (1): SwinTransformerBlock(
          dim=192, input_resolution=(6, 6), num_heads=12, window_size=6, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=192, window_size=(6, 6), num_heads=12
            (qkv): Linear(in_features=192, out_features=576, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=192, out_features=192, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.045)
          (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=192, out_features=768, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=768, out_features=192, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (2): SwinTransformerBlock(
          dim=192, input_resolution=(6, 6), num_heads=12, window_size=6, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=192, window_size=(6, 6), num_heads=12
            (qkv): Linear(in_features=192, out_features=576, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=192, out_features=192, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.055)
          (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=192, out_features=768, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=768, out_features=192, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (3): SwinTransformerBlock(
          dim=192, input_resolution=(6, 6), num_heads=12, window_size=6, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=192, window_size=(6, 6), num_heads=12
            (qkv): Linear(in_features=192, out_features=576, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=192, out_features=192, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.064)
          (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=192, out_features=768, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=768, out_features=192, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (4): SwinTransformerBlock(
          dim=192, input_resolution=(6, 6), num_heads=12, window_size=6, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=192, window_size=(6, 6), num_heads=12
            (qkv): Linear(in_features=192, out_features=576, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=192, out_features=192, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.073)
          (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=192, out_features=768, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=768, out_features=192, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (5): SwinTransformerBlock(
          dim=192, input_resolution=(6, 6), num_heads=12, window_size=6, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=192, window_size=(6, 6), num_heads=12
            (qkv): Linear(in_features=192, out_features=576, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=192, out_features=192, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.082)
          (norm2): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=192, out_features=768, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=768, out_features=192, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
      )
      (downsample): PatchMerging(
        input_resolution=(6, 6), dim=192
        (reduction): Linear(in_features=768, out_features=384, bias=False)
        (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
    )
    (3): BasicLayer(
      dim=384, input_resolution=(3, 3), depth=2
      (blocks): ModuleList(
        (0): SwinTransformerBlock(
          dim=384, input_resolution=(3, 3), num_heads=24, window_size=3, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=384, window_size=(3, 3), num_heads=24
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.091)
          (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=1536, out_features=384, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
        (1): SwinTransformerBlock(
          dim=384, input_resolution=(3, 3), num_heads=24, window_size=3, shift_size=0, mlp_ratio=4
          (norm1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (attn): WindowAttention(
            dim=384, window_size=(3, 3), num_heads=24
            (qkv): Linear(in_features=384, out_features=1152, bias=True)
            (attn_drop): Dropout(p=0.0, inplace=False)
            (proj): Linear(in_features=384, out_features=384, bias=True)
            (proj_drop): Dropout(p=0.0, inplace=False)
            (softmax): Softmax(dim=-1)
          )
          (drop_path): DropPath(drop_prob=0.100)
          (norm2): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
          (mlp): Mlp(
            (fc1): Linear(in_features=384, out_features=1536, bias=True)
            (act): GELU(approximate='none')
            (fc2): Linear(in_features=1536, out_features=384, bias=True)
            (drop): Dropout(p=0.0, inplace=False)
          )
        )
      )
    )
  )
  (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
  (avgpool): AdaptiveAvgPool1d(output_size=1)
  (head): Linear(in_features=384, out_features=10, bias=True)
)
```

## Import STL10 dataset

```python
# set the preprocess operations to be performed on train/val/test samples
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# download STL10 training set and reserve 50000 for training
train_set = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)

# download STL10 test set
test_set = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)

# define the data loaders using the datasets
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=128, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=128, shuffle=False)
```

```console
Files already downloaded and verified
Files already downloaded and verified
```

## Training

```python
# Make sure gradient tracking is on, and do a pass over the data
model.train(True)
# Training loop
num_of_epochs = 200
for epoch in range(num_of_epochs):
  for imgs, labels in tqdm_notebook(train_loader, desc='epoch '+str(epoch)):
    # Transfer to GPU
    imgs, labels = imgs.to(device), labels.to(device)
    # zero the parameter gradients
    model_optimizer.zero_grad()
    # Make predictions for this batch
    preds = model(imgs)
    # Compute the loss and its gradients
    loss = criterion(preds, labels)
    # backpropagate the loss
    loss.backward()
    # adjust parameters based on the calculated gradients
    model_optimizer.step()

torch.save(model.state_dict(), 'model_'+str(num_of_epochs)+'.pth')
```

## Importing a pretrained model

Uncomment this line if you're using a pretrained model.

```python
# model.load_state_dict(torch.load('model_STL10_200_embed_dim_48.pth', map_location=torch.device('cpu')))
```

```console
<All keys matched successfully>
```

## Testing

```python
all_labels, all_pred_labels = [], []
model.eval()
acc_total = 0
with torch.inference_mode():
  for imgs, labels in test_loader:
    imgs, labels = imgs.to(device), labels.to(device)
    preds = model(imgs)
    pred_cls = preds.data.max(1)[1]
    all_labels.append(labels.data.tolist())
    all_pred_labels.append(pred_cls.data.tolist())
    acc_total += pred_cls.eq(labels.data).cpu().sum()

all_labels_flat = list(itertools.chain.from_iterable(all_labels))
all_pred_labels_flat = list(itertools.chain.from_iterable(all_pred_labels))

acc = acc_total.item()/len(test_loader.dataset)
print(f'Accuracy on test set = {acc*100:.2f}%')
```

```console
Accuracy on test set = 47.46%
```

## Feeding one image to the model

```python
global global_attention
global_attention = []

img = train_loader.dataset.data[2,:,:,:]
print(img.shape)
img_plot = np.transpose(img, (1, 2, 0))
plt.imshow(img_plot)

print(img.shape)
img = torch.Tensor(img)
img = img.unsqueeze(0)
img = img.to(device)
print(img.shape)
pred = model(img)
```

```console
(3, 96, 96)
(3, 96, 96)
torch.Size([1, 3, 96, 96])
```

![image](https://github.com/user-attachments/assets/e611b7fd-1973-448f-b556-f34b53804c2f)

## Attention matrices

```python
for i in global_attention:
  print(i.shape)
```

```console
torch.Size([16, 3, 36, 36])
torch.Size([16, 3, 36, 36])
torch.Size([4, 6, 36, 36])
torch.Size([4, 6, 36, 36])
torch.Size([1, 12, 36, 36])
torch.Size([1, 12, 36, 36])
torch.Size([1, 12, 36, 36])
torch.Size([1, 12, 36, 36])
torch.Size([1, 12, 36, 36])
torch.Size([1, 12, 36, 36])
torch.Size([1, 24, 9, 9])
torch.Size([1, 24, 9, 9])
```

<img width="1920" height="4018" alt="drawings-02-001" src="https://github.com/user-attachments/assets/fbbb3db0-3053-4370-b1be-179f261293c2" />

<img width="1920" height="2497" alt="drawings-02-002" src="https://github.com/user-attachments/assets/eedeaa9b-3d27-46c6-955c-4ee262cf8fdf" />

<img width="1920" height="2279" alt="drawings-02-003" src="https://github.com/user-attachments/assets/86046108-23f4-41e7-8a14-f420d5cd8d0d" />

```python
img_attn = global_attention[0][:,:,:,:].cpu().detach()
print(img_attn.shape)
print(torch.sum(img_attn[0,0,:,:], dim=1))
```

```console
torch.Size([16, 3, 36, 36])
tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
```

## Attention matrix from the 1st layer and 1st head

```python
plt.imshow(global_attention[0][0,0,:,:].cpu().detach().numpy())
```

![image](https://github.com/user-attachments/assets/c101497c-e40f-46e3-b359-a00e0e8bb099)

## Visualizing attention values as histograms

```python
img_attn = global_attention[0][0,0,:,:].cpu().detach().numpy()

fig, axs = plt.subplots(img_attn.shape[0]//6, 6, figsize=(14, 12), layout="constrained")
for i, ax in enumerate(axs.ravel()):
    ax.hist(img_attn[i,:], bins=10)

plt.show()
```

![image](https://github.com/user-attachments/assets/bae3e17b-98a5-44fe-b0c0-b04c2fceda24)


## Plotting the attention matrices from all layers and all heads
The `plot_attention` function takes an attention matrix of size `(num_windows, window_height, window_width)` and returns an image of size `(num_windows*window_height, num_windows*window_width)`. For every patch in a window, I'm taking its value in attention matrix. Then these patches will be organized to form windows using `window_reverse` function.

```python
def plot_attention(img_plot, img_attn, plot_title):
    window_size = 6
    img_attn,_ = img_attn.max(axis=-1)
    num_windows, num_heads, num_patches = img_attn.shape
    img_attn = img_attn.reshape(num_windows, num_heads, int(num_patches**.5), int(num_patches**.5))

    if num_heads <= 6:
        fig, axs = plt.subplots(num_heads//3, 3, figsize=(12,6))
    else:
        fig, axs = plt.subplots(num_heads//3, 3, figsize=(12,12))
    fig.suptitle(plot_title)
    for i, ax in enumerate(axs.ravel()):
        img_attn_plot = img_attn[:,i,:,:]
        img_attn_plot = img_attn_plot.unsqueeze(-1)
        img_H = int(num_windows**.5) * int(num_patches**.5)
        img_attn_plot = window_reverse(img_attn_plot, window_size, img_H, img_H)
        img_attn_plot = img_attn_plot.squeeze(0).squeeze(-1).numpy()

        ax.imshow(scipy.ndimage.zoom(img_attn_plot, img_plot.shape[0]//img_attn_plot.shape[0]))
        ax.axis("off")

    plt.show()
```

<img width="1920" height="3620" alt="drawings-02-004" src="https://github.com/user-attachments/assets/83ae11ab-63e4-4749-a3cb-084d6c5a6a86" />

<img width="1920" height="2793" alt="drawings-02-005" src="https://github.com/user-attachments/assets/8cb124cb-db73-4cff-9ddc-ed0ae25be982" />

```python
plt.imshow(img_plot)
plt.axis("off")
plot_attention(img_plot, global_attention[0][:,:,:,:].cpu().detach(), 'Layer 00, SwinBlock 00')
plot_attention(img_plot, global_attention[2][:,:,:,:].cpu().detach(), 'Layer 01, SwinBlock 00')
plot_attention(img_plot, global_attention[4][:,:,:,:].cpu().detach(), 'Layer 02, SwinBlock 00')
plot_attention(img_plot, global_attention[6][:,:,:,:].cpu().detach(), 'Layer 02, SwinBlock 02')
plot_attention(img_plot, global_attention[8][:,:,:,:].cpu().detach(), 'Layer 02, SwinBlock 04')
```

![image](https://github.com/user-attachments/assets/a95bd2d3-1770-4de3-a784-3d31fc36caac)

![image](https://github.com/user-attachments/assets/3c6b7b33-94a8-4497-8221-8ffcb3a5aa81)

![image](https://github.com/user-attachments/assets/0d47c4d0-58c0-4519-b520-dafdced1f882)

![image](https://github.com/user-attachments/assets/82e066dd-e0e4-4190-a39c-8d078173db6c)

![image](https://github.com/user-attachments/assets/fb9daf3e-6494-4f81-8b3c-eab0f9d1148c)

![image](https://github.com/user-attachments/assets/5e95ba37-899c-4b7f-9840-365bd8b704f4)


```python
plt.imshow(img_plot)
plt.axis("off")
plot_attention(img_plot, global_attention[1][:,:,:,:].cpu().detach(), 'Layer 00, SwinBlock 01')
plot_attention(img_plot, global_attention[3][:,:,:,:].cpu().detach(), 'Layer 01, SwinBlock 01')
plot_attention(img_plot, global_attention[5][:,:,:,:].cpu().detach(), 'Layer 02, SwinBlock 01')
plot_attention(img_plot, global_attention[7][:,:,:,:].cpu().detach(), 'Layer 02, SwinBlock 03')
plot_attention(img_plot, global_attention[9][:,:,:,:].cpu().detach(), 'Layer 02, SwinBlock 05')
```

![image](https://github.com/user-attachments/assets/b6aa2ece-9e2f-4435-b17f-fed5a474f2ea)

![image](https://github.com/user-attachments/assets/9f65f2d6-3f1e-4a90-91e6-604f54c884fe)

![image](https://github.com/user-attachments/assets/efcb00a3-9541-461b-bb5b-1debbcab4dbe)

![image](https://github.com/user-attachments/assets/5ab74837-bec7-4372-b2bf-ee5627a5510d)

![image](https://github.com/user-attachments/assets/c24bfc2b-e324-461b-9f58-023b68f89ad4)

![image](https://github.com/user-attachments/assets/c0487e63-c290-4c65-9d9a-5e0f61786ca2)



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
