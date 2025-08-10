```bash
conda create -n docker
conda activate docker
pip install transformers torch torchvision fastapi uvicorn pillow timm python-multipart
```

```bash
uvicorn app:app --reload
```

```console
INFO:     Will watch for changes in these directories: ['/Users/mashaanalshammari/Downloads/app']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [10182] using StatReload
```

Go to:

[http://localhost:8000/docs](http://localhost:8000/docs)

You'll see /predict with a "Try it out" button.

Click "Try it out", upload an image, and click "Execute".

You'll get a JSON response with detections.

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

[https://docs.docker.com/desktop/setup/install/mac-install/](Install Docker Desktop on Mac)

```bash
docker build -t detr-api .
```

```bash
docker run -p 8000:8000 detr-api
```
