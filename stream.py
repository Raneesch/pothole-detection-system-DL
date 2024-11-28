import cv2
import numpy as np
import threading
from ultralytics import YOLO
import streamlit as st

# Initialize the YOLO model
model = YOLO('best.pt')
class_name = model.names

# Function to play voice alert in a separate thread
def play_voice_alert(message):
    import pyttsx3
    engine = pyttsx3.init()
    def speak():
        engine.say(message)
        engine.runAndWait()
    alert_thread = threading.Thread(target=speak)
    alert_thread.start()

# Streamlit app setup
st.title("Pothole Detection with YOLO")
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    # Open the video for processing
    cap = cv2.VideoCapture(video_path)
    frame_skip = 2  # Process every nth frame
    frame_placeholder = st.empty()
    st.info("Processing video, please wait...")

    count = 0
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        count += 1
        if count % frame_skip != 0:
            continue

        # Resize the frame for display and processing
        img = cv2.resize(img, (640, 360))
        h, w, _ = img.shape

        # Run YOLO model prediction
        results = model.predict(img, verbose=False)

        pothole_detected = False

        for r in results:
            boxes = r.boxes
            masks = r.masks

        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.numpy(), boxes):
                seg = cv2.resize(seg, (w, h))
                contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    d = int(box.cls)
                    c = class_name[d]
                    x, y, x1, y1 = cv2.boundingRect(contour)
                    cv2.rectangle(img, (x, y), (x + x1, y + y1), (255, 0, 0), 2)
                    cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

                    if c.lower() == "pothole":
                        pothole_detected = True

        if pothole_detected:
            play_voice_alert("Pothole detected!")

        # Convert BGR image to RGB for display in Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(img_rgb, channels="RGB", use_column_width=True)

    cap.release()
    st.success("Processing complete!")
else:
    st.info("Please upload a video to start detection.")
