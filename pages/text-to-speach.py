import streamlit as st
import numpy as np
import cv2
from gtts import gTTS
from PIL import Image
from moviepy.editor import ImageClip, TextClip, CompositeVideoClip, AudioFileClip
from datetime import datetime, timedelta

# Function to generate audio from text
def generate_audio(text, filename="speech_audio.mp3"):
    tts = gTTS(text)
    tts.save(filename)

# Function to overlay bold text on the image
def overlay_bold_text_on_image(image, text, font, font_scale, color, thickness, position):
    for i in range(3):  # Simulate bold text with slight shifts
        shift_position = (position[0] + i, position[1] + i)
        cv2.putText(image, text, shift_position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

# Streamlit App
st.title("NLP Text-to-Speech, Audio, and Video Generator")

# Description
st.write("""
This project is designed to automatically generate audio and visual content from written text, utilizing cutting-edge AI technologies: Google Text-to-Speech, OpenCV, and MoviePy libraries to convert written text into speech. 
The audio can be embedded in videos or played directly in the app, making it highly versatile for various use cases.
""")

st.write("""
- **Healthcare Training and Communication**: 
  Provides an engaging way to educate doctors about routine preventive care protocols.
  Customizable messages tailored to specific scenarios.

- **Chatbots and Accessibility Tools**:
  Enables voice-driven applications by converting pre-written scripts into audio-visual content.
  Improves accessibility for visually impaired individuals.

- **Educational Videos**:
  Creates narrated presentations for healthcare professionals or patients.

- **Marketing or Awareness Campaigns**:
  Produces promotional material for healthcare campaigns with minimal effort.
""")

# Add Image (replace with your own image path)
st.image("images/text-to-speach1.png", caption="AI-Healthcare Training")
st.image("images/text-to-speach2.png", caption="AI-Healthcare Training")
st.image("images/text-to-speach3.png", caption="AI-Healthcare Training")
# Templates
templates = {
    "Healthcare Reminder": {
        "title_text": "YOU ARE DUE FOR",
        "text_body": (
            "Our records indicate that you are due for a routine diabetes test. "
            "Your primary care physician has already placed lab orders for blood, urine, and other lab tests."
            "Please schedule your lab visit at your earliest convenience."
        ),
        "title_color": "#FFFFFF",
        "font_scale": 1.5,
        "thickness": 2,
        "x_pos": 50,
        "y_pos": 100,
    },
    "Marketing Campaign": {
        "title_text": "SPECIAL OFFER!",
        "text_body": (
            "Enjoy 25% off your next purchase. This limited-time offer is valid until the end of the month. "
            "Visit our website now to claim your discount!"
        ),
        "title_color": "#00FF00",
        "font_scale": 2.0,
        "thickness": 3,
        "x_pos": 100,
        "y_pos": 150,
    },
}

# Select Template
template_name = st.selectbox("Select a template",  list(templates.keys()) +["Custom"])
if template_name != "Healthcare Reminder":
    selected_template = templates[template_name]
else:
    selected_template = None

# Image Upload
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)  # Convert for OpenCV
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Pre-fill form fields with template values or leave empty for custom
    title_text = st.text_input("Enter title text", selected_template["title_text"] if selected_template else "")
    text_body = st.text_area(
        "Enter the body text",
        selected_template["text_body"] if selected_template else ""
    )

    title_color_hex = st.color_picker("Pick a color for the title text", "#FFFFFF")
   
    # Convert Hex to BGR
    # Extract RGB from the hex color code and convert to BGR format for OpenCV
    r = int(title_color_hex[1:3], 16)
    g = int(title_color_hex[3:5], 16)
    b = int(title_color_hex[5:7], 16)
    title_color_bgr = (b, g, r)  # OpenCV uses BGR format

    # Font scale and slider settings only if image is uploaded
      # Font scale and thickness
    font_scale = st.slider("Font scale for title", 0.5, 3.0, 1.5, key="font_scale_slider")
    thickness = st.slider("Text thickness", 1, 10, 2, key="thickness_slider")
    x_pos_title = st.slider("Title X Position", 0, image.shape[1], 50, key="x_pos_slider")
    y_pos_title = st.slider("Title Y Position", 0, image.shape[0], 50, key="y_pos_slider")

  
    overlay_image = image.copy()
    if title_text:
        # Apply the title text with the selected color
        cv2.putText(overlay_image, title_text, (x_pos_title, y_pos_title), cv2.FONT_HERSHEY_SIMPLEX, font_scale, title_color_bgr, thickness, lineType=cv2.LINE_AA)
        st.image(overlay_image, caption="Image with Title Overlay", use_container_width=True)

    # Generate Audio
    if st.button("Generate Audio"):
        audio_filename = "speech_audio.mp3"
        generate_audio(text_body, filename=audio_filename)
        st.audio(audio_filename, format="audio/mp3", start_time=0)
        st.success("Audio generated successfully!")

    # Generate Video
    if st.button("Generate Video"):
        with st.spinner("Processing video... Please wait."):
            # Convert the image to RGB for MoviePy
            overlay_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
            
            # Set a default duration for the background image clip
            bg_clip = ImageClip(overlay_image_rgb).set_duration(15)  # Duration in seconds

            # Load the audio clip
            audio_clip = AudioFileClip("speech_audio.mp3")

            # Define text segments with start times and durations
            text_segments = [
                {"text": "Routine Diabetes Testing", "start_time": 2, "duration": 3},
                {"text": "Blood and Urine Tests and other lab tests", "start_time": 8, "duration": 3},
            ]

            # Create text clips
            text_clips = []
            for segment in text_segments:
                text_clip = (
                    TextClip(
                        segment["text"],
                        fontsize=50,
                        font="Arial",
                        color="white",  # Ensure the text is white in the video
                        size=bg_clip.size,
                        method="caption"
                    )
                    .set_start(segment["start_time"])
                    .set_duration(segment["duration"])
                )
                text_clips.append(text_clip)

            # Combine background image and text clips
            final_video = CompositeVideoClip([bg_clip] + text_clips).set_audio(audio_clip)

            # Write the video file
            video_filename = "output_video.mp4"
            final_video.write_videofile(video_filename, fps=24, codec="libx264", audio_codec="aac")

            # Display the video in Streamlit
            st.video(video_filename)
            st.success("Video generated successfully!")

            # Schedule Button (Link to Google Meet)
            st.markdown(
                """
                <a href="https://meet.google.com/new" target="_blank">
                    <button style="background-color: #0F9D58; color: white; padding: 10px 20px; font-size: 16px; border: none; cursor: pointer;">
                        Schedule a Meeting
                    </button>
                </a>
                """,
                unsafe_allow_html=True
            )
