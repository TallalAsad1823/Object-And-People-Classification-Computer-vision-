import gradio as gr
import PIL.Image as Image
import tempfile
import cv2
from ultralytics import ASSETS, YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

def predict_image(img, conf_threshold, iou_threshold):
    """Predicts objects in an image using a YOLOv8 model with adjustable confidence and IOU thresholds."""
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im

def predict_video(video, conf_threshold, iou_threshold):
    """Predicts objects in a video using a YOLOv8 model with adjustable confidence and IOU thresholds."""
    # Create a temporary file to save the processed video
    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_output.close()

    # Load video
    cap = cv2.VideoCapture(video)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up VideoWriter to save output video
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on each frame
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            show_labels=True,
            show_conf=True,
            imgsz=640,
        )

        # Draw the results on the frame
        for r in results:
            frame = r.plot()

        # Write the frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()

    return temp_output.name

# Create a Gradio interface with support for both images and videos
iface = gr.Interface(
    fn=lambda img, conf_threshold, iou_threshold, is_video: predict_video(img, conf_threshold, iou_threshold) if is_video else predict_image(img, conf_threshold, iou_threshold),
    inputs=[
        gr.Video(type="file", optional=True, label="Upload Video"),
        gr.Image(type="pil", optional=True, label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
        gr.Checkbox(label="Is Video?", default=False),
    ],
    outputs=gr.Image(type="pil", label="Result") if not gr.Checkbox else gr.Video(type="file", label="Result"),
    title="Ultralytics Gradio Application 🚀",
    description="Upload images or videos for inference. The Ultralytics YOLOv8n model is used by default.",
    examples=[
        [ASSETS / "bus.jpg", 0.25, 0.45, False],
        [ASSETS / "zidane.jpg", 0.25, 0.45, False],
    ],
)

iface.launch(share=True)
