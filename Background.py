import cvzone
import google.generativeai as genai
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import av
from cvzone.HandTrackingModule import HandDetector  # Import the HandDetector directly

# Configure Google Generative AI
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Persona details
persona = """ I’m Rohan, a Mechanical Engineer with a specialization in Mechatronics, and I’ve transitioned into Operations Management at Amazon. With over 3 years of experience in research, data analytics, operations management, and project leadership, I thrive on cross-functional leadership, process optimization, and innovative problem-solving.
                In my current role as a Manager at Amazon since 2022, I lead operations in an automated fulfillment center, managing a team of over 100 associates and driving operational excellence. I’ve honed my expertise in troubleshooting, commissioning, and optimizing AI-based systems, achieving a consistent 95%+ on-time metric, reducing operational errors by 15%, and improving efficiency by 20%. Additionally, I led a green dunnage airbags recycling project that resulted in $88,000 in annual savings and a reduction of 1.2 tons of carbon emissions. My work is rooted in fostering an inclusive and supportive environment, particularly for immigrant communities, which has led to a 25% increase in employee engagement and retention.
                Prior to Amazon, I had enriching experiences as a Mechatronics Engineering Intern at Hover City Inc. and an Innovation and Business Development Intern at Toronto Hydro. At Hover City, I contributed to the design of a modular water system for a fully autonomous flying home. I led the research, 3D CAD modeling, and Finite Element Analysis (FEA), which reduced manufacturing time by 12% and scrap production by 32%. Collaborating with aerodynamics and systems engineering teams, I integrated the water system seamlessly into the vehicle structure, improving performance by 25%.
                At Toronto Hydro, I played a key role in launching five next-generation EV charging pilot projects in the City of Toronto. I used Alteryx and SQL to analyze charging data, improving utilization by 40%. My involvement in community roadshows helped drive a 40% increase in customer inquiries and a 25% growth in market share for EV infrastructure.
                Academically, I hold a Bachelor of Mechanical Engineering from Toronto Metropolitan University (formerly Ryerson University) with a specialization in Mechatronics in 2021. I was recognized on the Dean’s List and earned an A+ in my Capstone Design Project. I was also a finalist in the Boeing Go Fly Competition, where I showcased an eVTOL aircraft prototype.
                Throughout my career, I’ve developed proficiency in a wide array of tools and platforms, including Microsoft Office Suite, Project, Visio, JIRA, SolidWorks, ANSYS, AutoCAD, Tableau, Alteryx, Python, and SQL. I’ve led several technical projects, such as developing an AI-powered object detection tool using the YOLOv8 algorithm, and a breast cancer classifier using machine learning.
                I am passionate about entrepreneurship, sustainability, and innovation.My diverse experience across operations, technical design, and product development drives my goal to continue growing in fields like AI, robotics, and sustainable technology. You can connect with me via linkedin: www.linkedin.com/in/rd-rohan-doshi. You can also email me at rohannavindoshi@gmail.com. You can call: +1647-982-0448. I am currently located in Vancouver, Canada. Answer all the questions in first person view and if not sure about any of response respond "Its a secret" """

# Display profile info with image
col1, col2 = st.columns(2)
with col1:
    st.title("Hi, I'm Rohan Doshi")

with col2:
    st.image("Images/image (1).jpg")

# Persona description for chat functionality
st.title("Chat with virtual me")
user_question = st.text_input("Ask anything you would like to know about me?")
if st.button("ASK", use_container_width=400):
    prompt = persona + user_question
    response = model.generate_content(prompt)
    st.write(response.text)

# Projects gallery with videos
st.title("Projects Gallery")
col5, col6 = st.columns([4, 4])
with col5:
    st.subheader("Interactive Gesture Control Map")
    st.video("Videos/Interactive_Map.mp4")
with col6:
    st.subheader("Autonomous Robotic Vehicle")
    st.video("Videos/Robotic_Vehicle.mov")

# Interactive AI content generator project
st.subheader("Interactive AI Content Generator")
st.text_input("This project is a content generator powered by large language models that enables users to draw with hand gestures as input.")
st.image("Images/HandSign.jpg", width=350)
st.subheader("Let's try it yourself")

# Webcam input for hand gesture detection
col3, col4 = st.columns([2, 1])
with col3:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col4:
    st.title("Answer")
    output_text_area = st.subheader("")

# Hand detector initialization
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)


# Callback function to process each frame
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")  # Convert frame to numpy array

    # Detect hand landmarks
    hands, img = detector.findHands(img, draw=True, flipType=True)
    
    # Process hand information and gestures
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        
        # Drawing or content generation based on hand gesture
        if fingers == [0, 1, 0, 0, 0]:  # Pointer gesture
            st.write("Pointer gesture detected, ready to draw!")
        
        elif fingers == [1, 1, 0, 0, 1]:  # Reset condition
            st.write("Gesture to reset detected, resetting canvas...")

        # Add gesture for AI content generation
        if fingers == [1, 1, 1, 1, 1]:  # All fingers up
            response = model.generate_content("Generate a creative response")
            st.write(response.text)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Initialize WebRTC streamer to capture webcam input and process
webrtc_ctx = webrtc_streamer(
    key="webcam-stream",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},  # Only video, no audio
)

# Optionally, you can add a message or instructions
#st.markdown("Webcam streaming using Streamlit.")
