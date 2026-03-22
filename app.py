import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
import tempfile

FRAMES_PER_VIDEO = 20
IMG_SIZE = 96

model = tf.keras.models.load_model("best_har_model.h5")

with open("class_names.json") as f:
    class_names = json.load(f)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = np.linspace(0, total_frames-1, FRAMES_PER_VIDEO, dtype=int)

    frames=[]
    count=0
    target=0

    while cap.isOpened() and target < len(frame_indices):
        ret, frame = cap.read()
        if not ret:
            break

        if count == frame_indices[target]:
            frame = cv2.resize(frame,(IMG_SIZE,IMG_SIZE))
            frame = frame/255.0
            frames.append(frame)
            target+=1

        count+=1

    cap.release()

    while len(frames)<FRAMES_PER_VIDEO:
        frames.append(np.zeros((IMG_SIZE,IMG_SIZE,3)))

    return np.array(frames)

st.title("Human Activity Recognition")

video = st.file_uploader("Upload Video",type=["mp4","avi","mov"])

if video:
    st.video(video)

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    if st.button("Predict"):
        frames = extract_frames(tfile.name)
        frames = np.expand_dims(frames,0)

        pred = model.predict(frames)
        idx = np.argmax(pred)

        st.success(class_names[idx])
