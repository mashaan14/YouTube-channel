```bash
conda create -n docker
conda activate docker
pip install transformers torch torchvision fastapi uvicorn pillow timm python-multipart
```

```json
{
  "detections": [
    {
      "label": "cup",
      "score": 0.979,
      "box": [
        3170.98,
        2198.01,
        3595.93,
        2867.43
      ]
    },
    {
      "label": "mouse",
      "score": 0.999,
      "box": [
        2159.85,
        2887.22,
        2598.22,
        3205.22
      ]
    },
    {
      "label": "keyboard",
      "score": 0.999,
      "box": [
        710.57,
        2329.35,
        1937.02,
        2891.38
      ]
    },
    {
      "label": "tv",
      "score": 1,
      "box": [
        1054.97,
        783.91,
        2811.05,
        2358.62
      ]
    }
  ]
}
```
