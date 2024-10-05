

import google.generativeai as genai
import numpy as np
import streamlit as st
from streamlit_webrtc import  WebRtcMode, webrtc_streamer
import cv2
import mediapipe as mp
from PIL import Image

# Configure Google Generative AI
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Display profile info with image
col1, col2 = st.columns(2)
with col1:
    st.title("Hi, I'm Rohan Doshi")
with col2:
    st.image("Images/image (1).jpg")

# Persona details (truncated for brevity)
persona = """I’m Rohan, a Mechanical Engineer with a specialization in Mechatronics, and I’ve transitioned into Operations Management at Amazon..."""

# Chat functionality
st.title("Chat with virtual me")
user_question = st.text_input("Ask anything you would like to know about me?")
if st.button("ASK", use_container_width=400):
    prompt = persona + user_question
    response = model.generate_content(prompt)
    st.write(response.text)

# Projects gallery with videos
st.title("Projects Gallery")
col3, col4 = st.columns([4, 4])
with col3:
    st.subheader("Interactive Gesture Control Map")
    st.video("Videos/Interactive_Map.mp4")
with col4:
    st.subheader("Autonomous Robotic Vehicle")
    st.video("Videos/Robotic_Vehicle.mov")

st.subheader("Interactive AI Content Generator")
st.text_input("This project is an AI-powered content generator where you can draw with your fingers in the air and receive personalized responses.")
st.image("Images/HandSign.jpg", width=350)
st.subheader("Let's try it yourself")

col5, col6 = st.columns([2, 1])
with col5:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col6:
    st.title("Answer")
    output_text_area = st.subheader("")

# Hand detector initialization
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize MediaPipe Hands detector
detector = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Function to get hand information
def getHandInfo(img):
    hands, img = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if hands.multi_hand_landmarks:
        hand_landmarks = hands.multi_hand_landmarks[0]
        lmList = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        fingers = [1 if lm.y < lmList[i - 2][1] else 0 for i, lm in enumerate(hand_landmarks.landmark[8:21:4], start=8)]  # Basic finger detection logic
        return fingers, lmList
    else:
        return None

# Function to draw on the canvas
def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up
        current_pos = lmList[8][:2]  # Index fingertip position
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(np.multiply(current_pos, [640, 480]).astype(int)),
                 tuple(np.multiply(prev_pos, [640, 480]).astype(int)), (255, 0, 255), 10)
    elif fingers == [1, 1, 0, 0, 1]:  # All fingers up (reset canvas)
        canvas = np.zeros_like(canvas)
    
    return current_pos, canvas

# Function to send the canvas to Google Generative AI
def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 1]:  # All fingers up, trigger AI content generation
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["guess the answer.", pil_image])
        return response.text
    return ""

prev_pos = None
canvas = None
output_text = ""

# Streamlit WebRTC streamer to capture video frames
webrtc_ctx = webrtc_streamer(key="gesture-detector", mode=WebRtcMode.SENDRECV)

while run:
    if webrtc_ctx.video_receiver:
        # Capture video frames
        frame = webrtc_ctx.video_receiver.get_frame()
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, flipCode=1)

        if canvas is None:
            canvas = np.zeros_like(img)

        # Get hand info
        info = getHandInfo(img)
        if info:
            fingers, lmList = info
            prev_pos, canvas = draw(info, prev_pos, canvas)
            output_text = sendToAI(model, canvas, fingers)

        # Blend the canvas and original image
        image_combined = cv2.addWeighted(img, 0.80, canvas, 0.20, 0)
        FRAME_WINDOW.image(image_combined, channels="BGR")

        if output_text:
            output_text_area.text(output_text)
# Setup WebRTC Streamer
webrtc_ctx = webrtc_streamer(
    key="gesture-detector",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)
