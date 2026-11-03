# OBJECT DETECTION USING WEB CAMERA

## Aim

To access the laptop camera, capture an image, and detect objects present in the captured image using the YOLOv8 object detection model.

---

## Software Requirements

- Anaconda
- Jupyter Notebook
- Python
- OpenCV (`cv2`)
- Ultralytics YOLOv8
- NumPy
- Matplotlib
- Laptop Camera

---

## Developed By

- **Name:** SHRIVARSHAN P G
- **Register No:** 212225240146

---

## ⚙️ Algorithm

### Step 1:
Import the necessary libraries, including OpenCV, Ultralytics YOLO, and other required modules.

### Step 2:
Load the pretrained YOLOv8 model using the `YOLO()` function.

### Step 3:
Access the laptop camera using `cv2.VideoCapture(0)`.

### Step 4:
Check whether the laptop camera is successfully opened.

### Step 5:
Wait for 5 seconds to allow the camera to initialize and capture a clear image.

### Step 6:
Capture an image from the laptop camera using `cap.read()`.

### Step 7:
Display the captured image using OpenCV or Matplotlib.

### Step 8:
Pass the captured image to the YOLOv8 model for object detection.

### Step 9:
Identify the objects present in the image and generate bounding boxes with class labels and confidence scores.

### Step 10:
Display the final detected image with bounding boxes and labels.

### Step 11:
Repeat the experiment using three different scenes and compare the detection results.

---

## Algorithm Flowchart

```text
       Laptop Camera
             |
             v
       Capture Image
             |
             v
       Display Image
             |
             v
      YOLOv8 Detection
             |
             v
   Identify Detected Objects
             |
             v
 Display Bounding Boxes
       and Labels
             |
             v
       Final Output
```

---

## Program

```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Load a more accurate YOLO model
model = YOLO("yolov8m.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")

else:
    video_writer = None

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = model(
                frame,
                conf=0.60,
                verbose=False
            )

            annotated_frame = results[0].plot()

            # Initialize video writer
            if video_writer is None:
                h, w = annotated_frame.shape[:2]

                video_writer = cv2.VideoWriter(
                    "YOLO_Detection_Output.mp4",
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    20,
                    (w, h)
                )

            # Save detected frame
            video_writer.write(annotated_frame)

            # Display output in Jupyter
            rgb_frame = cv2.cvtColor(
                annotated_frame,
                cv2.COLOR_BGR2RGB
            )

            clear_output(wait=True)

            plt.figure(figsize=(10, 8))
            plt.imshow(rgb_frame)
            plt.title("YOLO Real-Time Object Detection")
            plt.axis("off")
            plt.show()

    except KeyboardInterrupt:
        print("Detection stopped.")

    finally:
    # Release webcam
        cap.release()

    # Release video writer
    if video_writer is not None:
        video_writer.release()

    print("Webcam released.")
    print("Video saved as YOLO_Detection_Output.mp4")
```

---

## Student Task

1. Capture your own image using the laptop camera.
2. Detect the objects present in the captured image.
3. Display the final output with bounding boxes and labels.
4. Perform the experiment using **3 different scenes**.
5. Observe and compare the detected objects in each scene.

---

## Expected Output and Original Captured Image

<img width="1067" height="777" alt="image" src="https://github.com/user-attachments/assets/01980ddf-c659-43de-96b2-e2edc3231f62" />


---

## Submission Requirements

- Jupyter Notebook (`.ipynb`)
- Three original captured images
- Three YOLOv8 output images
- Screenshot of the final results

---

## GitHub Reference

[Ultralytics YOLOv8 – Official GitHub Repository](https://github.com/ultralytics/ultralytics)

---

## Result

Thus, object detection was successfully performed using the YOLOv8 model by accessing the laptop camera, capturing images, and identifying objects with bounding boxes and labels across three different scenes.

---

**Platform:** Anaconda + Jupyter Notebook
