import gradio as gr
import PIL.Image as Image
from ultralytics import ASSETS, YOLO,SAM
import os
from dotenv import load_dotenv
load_dotenv()

#---------Load the pre-trained YOLOv11 & model
MODEL_PATH = os.getenv("MODEL_PATH")
if MODEL_PATH is None:
    raise ValueError("MODEL_PATH environment variable not set.")
model = YOLO(MODEL_PATH)
sam_model = SAM("sam2_b.pt")

def resize_image(image, size=(512, 512)):
    """Resize the input image to the given size."""
    return image.resize(size)

def process_image(image_file, conf_threshold, iou_threshold, line_width):
    """Process image through both YOLO and SAM models."""
    if image_file is None:
        return None, None
    image_resized = resize_image(image_file, size=(512, 512))
    
    yolo_results = model.predict(
        source=image_resized,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        line_width=line_width
    )
    
    yolo_output = yolo_results[0].plot() if yolo_results else None
    
    #------SAM segmentation
    sam_output = None
    if yolo_results and len(yolo_results[0].boxes):
        boxes = yolo_results[0].boxes.xyxy
        
        sam_results = sam_model(
            image_resized,
            bboxes=boxes,
            verbose=False,
            device=0
        )
        
        if sam_results and len(sam_results) > 0:
            sam_output = sam_results[0].plot()
    
    return yolo_output, sam_output

# Create Gradio interface
interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
        gr.Slider(minimum=1, maximum=10, value=1, label="Line width")
    ],
    outputs=[
        gr.Image(type="pil", label="YOLO Detection"),
        gr.Image(type="pil", label="SAM Segmentation")
    ],
    title="Object Detection and Segmentation",
    description="Upload images for YOLO detection and SAM segmentation analysis.",
    show_progress=True,
)

interface.launch(
    share=True,
    auth=("username", "password")
)  