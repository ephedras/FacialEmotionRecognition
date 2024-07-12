'''
Facial detection and expression recognition along with analytics
This is fun project utilising opencv models YuNEt for dection ad facenet for emotion detection
ver. 3.0.9

----------
Lot of flaws could be found; feel free to correct them. Especially safe exit could be implemented.
Have fun.

----------
ephedras
'''
import streamlit as st
import cv2 as cv
import threading
import time
from utils.facial_fer_model import FacialExpressionRecog
from utils.yunet import YuNet
import datetime
import numpy as np


def visualize(image, det_res, fer_res, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    fer_type =''
    print('%s %3d faces detected.' % (datetime.datetime.now(), len(det_res)))

    output = image.copy()
    landmark_color = [
        (255,  0,   0),  # right eye
        (0,    0, 255),  # left eye
        (0,  255,   0),  # nose tip
        (255,  0, 255),  # right mouth corner
        (0,  255, 255)   # left mouth corner
    ]

    for ind, (det, fer_type) in enumerate(zip(det_res, fer_res)):
        bbox = det[0:4].astype(np.int32)
        fer_type = FacialExpressionRecog.getDesc(fer_type)
        print("Face %2d: %d %d %d %d %s." % (ind, bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], fer_type))
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)
        cv.putText(output, fer_type, (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
        
        # expression_counts[fer_type]+=1
        landmarks = det[4:14].astype(np.int32).reshape((5, 2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)
    return output,fer_type


def process(detect_model, fer_model, frame):
    h, w, _ = frame.shape
    detect_model.setInputSize([w, h])
    dets = detect_model.infer(frame)

    if dets is None:
        return False, None, None

    fer_res = np.zeros(0, dtype=np.int8)
    for face_points in dets:
        fer_res = np.concatenate((fer_res, fer_model.infer(frame, face_points[:-1])), axis=0)
    return True, dets, fer_res

st.set_page_config(page_title="FDER",page_icon='assets/logo/icon_clear.png')

# Set up the sidebar with a logo, a dropdown, and radio buttons
st.sidebar.image("assets/logo/logo_clear.png", use_column_width=True)

st.title("Facial Detection and Expression Recognition")

# Global variables to store the frame and the camera status
frame = None
camera_active = False
detect_model = None
fer_model = None
capture_thread = None
expression_counts = {'happy': 0, 'sad': 0, 'disgust': 0, 'neutral':0, "angry":0,  "fearful":0, "surprised":0}
total_frames = 0
last_expression = None

# Initialize face detection and facial expression recognition models
backend_target_pairs = {
    'default': (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_TARGET_CPU)
}

# Function to initialize the models
def initialize_models():
    global detect_model, fer_model
    detect_model = YuNet(modelPath=r'models\face_detection_yunet_2023mar.onnx')
    fer_model = FacialExpressionRecog(modelPath=r'models\facial_expression_recognition_mobilefacenet_2022july_int8.onnx',
                                      backendId=backend_target_pairs['default'][0],
                                      targetId=backend_target_pairs['default'][1])

# Function to capture frames from the webcam
def capture_frames():
    global frame, camera_active, expression_counts, total_frames,  last_expression
    cap = cv.VideoCapture(0)  # Capture from the default camera

    while camera_active:
        ret, captured_frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break

        # Get detection and FER results
        status, dets, fer_res = process(detect_model, fer_model, captured_frame)
        if status:
            # Draw results on the input image
            frame,typ = visualize(captured_frame, dets, fer_res)
            if typ!='':
                expression_counts[typ]+=1
                total_frames += 1
                last_expression = typ
        else:
            frame = captured_frame

        time.sleep(0.05)  # Add a small delay to prevent high CPU usage

    cap.release()
# Function to start the camera feed
def start_camera():
    global camera_active, capture_thread, expression_counts, total_frames
    if not camera_active:
        camera_active = True
        initialize_models()
        capture_thread = threading.Thread(target=capture_frames)
        capture_thread.start()
        expression_counts = {'happy': 0, 'sad': 0, 'disgust': 0, 'neutral':0, "angry":0,  "fearful":0, "surprised":0}
        total_frames = 0

# Function to stop the camera feed and offload the models
def stop_camera():
    global camera_active, detect_model, fer_model, capture_thread
    camera_active = False
    if capture_thread is not None:
        capture_thread.join()  # Wait for the capture thread to finish
    detect_model = None
    fer_model = None

# Display buttons to control the camera
if st.sidebar.button("Start Camera"):
    start_camera()

if st.sidebar.button("Stop Camera"):
    stop_camera()

# Create placeholders
left_col, right_col = st.columns(2)
frame_placeholder = left_col.empty()
emoji_placeholder = right_col.empty()

progress_placeholders = {
    'happy': right_col.empty(),
    'sad': right_col.empty(),
    'disgust': right_col.empty(),
    'neutral': right_col.empty(),
    'angry': right_col.empty(),
    'fearful': right_col.empty(),
    'surprised': right_col.empty()
}

# Emojis for each expression
emojis = {
    'happy': '😊',
    'sad': '😢',
    'disgust': '🤢',
    'neutral': '😐',
    'angry': '😠',
    'fearful': '😨',
    'surprised': '😲'
}

# Main loop to display the frame when the camera is active
while True:
    if camera_active and frame is not None:
        frame_placeholder.image(frame, channels="BGR")
        
        # Calculate the percentage of each expression
        if total_frames > 0:
            percentages = {key: (count / total_frames) * 100 for key, count in expression_counts.items()}
        else:
            percentages = {key: 0 for key in expression_counts}

                   

        # Update progress bars
        for expression, percentage in percentages.items():
            progress_placeholders[expression].progress(int(percentage), text=f'{expression.capitalize()}: {percentage:.2f}%')
        
        # Update the emoji for the last detected expression
        if last_expression:
            emoji_placeholder.markdown(f"### {emojis[last_expression]}")

    elif not camera_active:
        frame_placeholder.empty()  # Clear the placeholder when the camera is inactive
        emoji_placeholder.empty()  # Clear the emoji placeholder when the camera is inactive
        for placeholder in progress_placeholders.values():
            placeholder.empty()  # Clear the progress bars when the camera is inactive

    time.sleep(1)  # Update every second
