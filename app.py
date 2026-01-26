# Ultralytics YOLO 🚀, AGPL-3.0 license

import gradio as gr
import PIL.Image as Image
from ultralytics import YOLO

model = None

def predict_image(img, conf_threshold, iou_threshold, model_name, show_labels, show_conf, imgsz):
    """Predicts objects in an image using a YOLOv8 model with adjustable confidence and IOU thresholds."""
    model = YOLO(model_name)
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=show_labels,
        show_conf=show_conf,
        imgsz=imgsz,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im


iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
        gr.Radio(choices=["yolov8n", "yolov8s", "yolov8m", "yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8n-pose", "yolov8s-pose", "yolov8m-pose", "yolov8n-obb", "yolov8s-obb", "yolov8m-obb"], label="Model Name", value="yolov8n"),
        gr.Checkbox(value=True, label="Show Labels"),
        gr.Checkbox(value=True, label="Show Confidence"),
        gr.Radio(choices=[320, 640, 1024], label="Image Size", value=640),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Ultralytics YOLOv8 Inference 🚀",
    description="Upload images for inference. The Ultralytics YOLOv8n model is used by default.",
    examples=[
        ["https://ultralytics.com/images/bus.jpg", 0.25, 0.45, "yolov8n", True, True, 640],
        ["https://ultralytics.com/images/zidane.jpg", 0.25, 0.45, "yolov8n-seg", True, True, 640],
        ["https://ultralytics.com/images/boats.jpg", 0.25, 0.45, "yolov8n-obb", True, True, 1024],
    ],
)
iface.launch(share=True)
