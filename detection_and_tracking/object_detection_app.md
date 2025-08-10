# Deploying An Object Detection App to HuggingFace Spaces

In this post I'll walk you through deploying an object detection app. We're going to test the app locally to make sure everything works well. Then, we'll wrap the Dockerfile and upload it with `App.py` to HuggingFace Spaces.
First, let's play it safe and create a conda environment specifically for this project. We’re going to name this environment `docker`:

```bash
conda create -n docker
conda activate docker
pip install transformers torch torchvision fastapi uvicorn pillow timm python-multipart
```

my_folder/
        ├── App.py
        ├── Dockerfile
        └── requirements.txt

In the python code we’re going to use DETR model for object detection. We’re also going to use FastAPI to serve the model:

```python
import os
os.environ["HF_HOME"] = "/tmp/huggingface"

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import io

app = FastAPI()

# Load model + processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    # Get model predictions
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Draw settings based on image size
    scale = max(width, height) / 512  # scale factor relative to 512x512
    box_thickness = int(3 * scale)  # Slightly thicker boxes for visibility
    font_size = max(12, int(15 * scale))  # Increase base font size for better readability

    # Try to load a Truetype font (with fallback to a more reliable font)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # Ensure arial.ttf is available in your environment
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)  # Fallback to another common font
        except:
            font = ImageFont.load_default(size=font_size)  # Use default with explicit size if possible

    # Draw boxes and labels
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_text = f"{model.config.id2label[label.item()]} {score:.2f}"

        # Draw rectangle
        draw.rectangle(box, outline="red", width=box_thickness)

        # Draw label background
        bbox = font.getbbox(label_text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_bg = [box[0], box[1] - text_height - 2, box[0] + text_width + 4, box[1]]  # Add padding
        draw.rectangle(text_bg, fill="red")

        # Draw label text with slight offset for clarity
        draw.text((box[0] + 2, box[1] - text_height - 2), label_text, fill="white", font=font)

    # Return image as PNG
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return StreamingResponse(img_byte_arr, media_type="image/png")
```

Now, let’s build our python file:

```bash
python app.py
```

You can access the app locally by running:

```bash
uvicorn app:app --reload
```

If everything goes alright, you’ll see something like this on the terminal:

```console
INFO:     Will watch for changes in these directories: ['/Users/mashaanalshammari/Downloads/app']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [10182] using StatReload
```

Go to:

[http://localhost:8000/docs](http://localhost:8000/docs)

![Screenshot 2025-08-07 at 12 08 43 AM](https://github.com/user-attachments/assets/68af126e-4f85-4fa3-bfb7-5f489de6a7b8)

You'll see /predict with a "Try it out" button.

![Screenshot 2025-08-07 at 12 09 02 AM](https://github.com/user-attachments/assets/cfa544ae-99ef-457f-9963-100a90a11ab8)

Click "Try it out", upload an image, and click "Execute".

![Screenshot 2025-08-07 at 12 31 55 AM](https://github.com/user-attachments/assets/e8b6ddae-258a-4a18-80d6-c4b71510b90c)

The image is annotated by the bounding boxes that we specified inside `@app.post(“/predict”)`. You can control the line width, color, or font from that function.

Also, you'll get a JSON response with detections. These numbers represent the model confidence, the label, and the coordinates for the bounding box.

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
