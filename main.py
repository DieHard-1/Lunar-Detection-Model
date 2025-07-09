
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import uvicorn
import os
import cv2
import base64
import numpy as np
import torch
from torch import nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = YOLO(r"model\best.pt")

@app.get("/")
async def root():
    return {"message": "Lunar detection backend is up!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Save temporarily
    temp_path = "temp_input.jpg"
    with open(temp_path, "wb") as f:
        f.write(image_bytes)

    # Run prediction
    results = model(temp_path, conf=0.2)
    result = results[0]

    # Load image with OpenCV
    original_img = cv2.imread(temp_path)

    # ---------- YOLO Bounding Box Section ----------
    crater_table = []
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        width = x2 - x1
        height = y2 - y1
        cx = int(x1 + width / 2)
        cy = int(y1 + height / 2)
        diameter = (width + height) / 2
        cv2.circle(original_img, (cx, cy), radius=3, color=(255, 0, 0), thickness=-1)
        crater_table.append({
            "ID": i + 1,
            "Center X": round(cx, 2),
            "Center Y": round(cy, 2),
            "Width": round(width, 2),
            "Height": round(height, 2),
            "Diameter": round(diameter, 2),
            "Confidence": round(conf, 3)
        })

    plotted = result.plot(labels=False)

    # === Single Best Landing Zone with Margin and Larger Box ===
    crater_mask = np.zeros((original_img.shape[0], original_img.shape[1]), dtype=np.uint8)
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        cv2.rectangle(crater_mask, (x1, y1), (x2, y2), 255, -1)

    window_size = 120
    step = 15
    margin = 40
    safe_candidates = []

    for y in range(margin, original_img.shape[0] - window_size - margin, step):
        for x in range(margin, original_img.shape[1] - window_size - margin, step):
            roi = crater_mask[y:y+window_size, x:x+window_size]
            crater_pixels = np.count_nonzero(roi)
            if crater_pixels < 10:
                center_dist = abs((x + window_size // 2) - 320) + abs((y + window_size // 2) - 320)
                safe_candidates.append((crater_pixels, center_dist, x, y))

    if safe_candidates:
        safe_candidates.sort(key=lambda t: (t[0], t[1]))  
        _, _, x, y = safe_candidates[0]
        cv2.rectangle(plotted, (x, y), (x+window_size, y+window_size), (0, 255, 0), 2)
        cv2.putText(plotted, "Safest Landing Zone", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    pil_image = Image.fromarray(plotted)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    img_data = f"data:image/png;base64,{img_b64}"

    # === Grad-CAM APPENDED SECTION ===
    gradcam_img = generate_gradcam(temp_path)
    gradcam_pil = Image.fromarray(gradcam_img)
    gradcam_buf = BytesIO()
    gradcam_pil.save(gradcam_buf, format="PNG")
    gradcam_b64 = base64.b64encode(gradcam_buf.getvalue()).decode("utf-8")
    gradcam_data = f"data:image/png;base64,{gradcam_b64}"

    os.remove(temp_path)

    return {
        "image": img_data,
        "table": crater_table,
        "gradcam": gradcam_data
    }

# Grad-CAM generator function
def generate_gradcam(img_path):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
    import cv2
    import torch
    import numpy as np
    from torch import nn

    target_layer_index = 10
    target_layer = model.model.model[target_layer_index]

    class YOLOBackbone(nn.Module):
        def __init__(self, model, index):
            super().__init__()
            self.features = nn.Sequential(*model.model.model[:index+1])
        def forward(self, x):
            x.requires_grad_()
            return self.features(x)

    wrapper = YOLOBackbone(model, target_layer_index).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper.to(device)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = rgb.astype(np.float32) / 255.0
    tensor = preprocess_image(img_float, mean=[0, 0, 0], std=[1, 1, 1]).to(device)

    class DummyTarget:
        def __call__(self, x): return x.sum()

    with GradCAM(model=wrapper, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=[DummyTarget()], eigen_smooth=True)[0]

    heatmap = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
    return heatmap

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
