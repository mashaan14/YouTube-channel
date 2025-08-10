# Deploying An Object Detection App to HuggingFace Spaces

In this post I'll walk you through deploying an object detection app. We're going to test the app locally to make sure everything works well. Then, we'll wrap the Dockerfile and upload it with `App.py` to HuggingFace Spaces.
First, let's play it safe and create a conda environment specifically for this project. We’re going to name this environment `docker`:

```bash
conda create -n docker
conda activate docker
pip install transformers torch torchvision fastapi uvicorn pillow timm python-multipart
```

Our folder structure would look like this:

```
my_folder/
        ├── App.py
        ├── Dockerfile
        └── requirements.txt
```

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

Let's add a simple HTML interface to our app. We can do it by modifying `App.py`

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

@app.get("/", response_class=HTMLResponse)
async def homepage():
    return """
    <html>
        <head>
            <title>DETR Drag and Drop</title>
        </head>
        <body>
            <h2>Upload an Image for Object Detection</h2>
            <form id="upload-form" enctype="multipart/form-data">
                <input type="file" name="file" id="fileInput" />
                <br><br>
                <button type="submit">Upload</button>
            </form>
            <br>
            <img id="outputImage" style="max-width: 100%;" />
            <script>
                const form = document.getElementById("upload-form");
                form.addEventListener("submit", async (e) => {
                    e.preventDefault();
                    const fileInput = document.getElementById("fileInput");
                    const formData = new FormData();
                    formData.append("file", fileInput.files[0]);

                    const response = await fetch("/predict", {
                        method: "POST",
                        body: formData
                    });
                    const blob = await response.blob();
                    const imageUrl = URL.createObjectURL(blob);
                    document.getElementById("outputImage").src = imageUrl;
                });
            </script>
        </body>
    </html>
    """
```

Install Docker for personal use. I downloaded Docker from this link [Install Docker Desktop on Mac](https://docs.docker.com/desktop/setup/install/mac-install/).

Once you get Docker up and running, put these commands into the Dockerfile and build it from the terminal.

```docker
# Use a slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for compiling packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
        && apt-get clean && \
        rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a non-root user for security (required for Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose port
EXPOSE 7860

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

Now, build it from the terminal:

```bash
docker build -t detr-api .
```

![Screenshot 2025-08-07 at 12 47 31 AM](https://github.com/user-attachments/assets/0ca63a4b-9377-47ef-9c13-d9147a319b2d)

Go to Hugging Face spaces and select building a space from Dockerfile and upload the Dockerfile and `App.py`.

![Screenshot 2025-08-10 at 3 16 55 PM](https://github.com/user-attachments/assets/919f7a3b-5c5b-442e-84cd-bbb92a90f14a)


Upon successful build on Hugging Face you’ll get an app the can detect objects inside an image:

![Screenshot 2025-08-10 at 3 16 40 PM](https://github.com/user-attachments/assets/7be19d8b-dc19-48c9-a441-7981fbd5771a)
