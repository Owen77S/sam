import numpy as np
import torch

#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model, device="cpu")

# Load an image
image_path = "image.jpg"
image = Image.open(image_path)
inference_state = processor.set_image(image)

# Prompt the model with text
prompt="Segment the car"
output = processor.set_text_prompt(state=inference_state, prompt=prompt)

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# Overlay the masks on the image
def overlay_masks(image, masks, alpha=0.5):
    result = np.array(image.convert("RGB"), dtype=np.float32)
    rng = np.random.default_rng(0)
    for mask in masks:
        m = mask.squeeze().cpu().numpy().astype(bool)
        color = rng.integers(0, 255, 3)
        result[m] = (1 - alpha) * result[m] + alpha * color
    return Image.fromarray(result.astype(np.uint8))

segmented = overlay_masks(image, masks)
segmented.save("img_segmented.jpg")
