import streamlit as st

# Title and Introduction
st.title("AI Voice Assistant")

st.write("""
This AI Voice Assistant offers a hands-free, voice-based approach to managing personal and professional tasks, making digital interactions more natural.
""")

# Key Features of the AI Voice Assistant
st.subheader("Key Features of the AI Voice Assistant:")
st.write("""
- **Speech-to-Text (STT):** Converts spoken language into written text, allowing the assistant to understand and process voice input.
- **Text-to-Speech (TTS):** Converts the assistant’s response into natural-sounding speech, making interactions feel human-like.
 **Generative AI:**  Generative AI plays a critical role in this assistant's functionality. When the assistant receives text
input (via STT), it processes that text to understand the user's intent and then generates a relevant response. This
- **Vocal Interaction:** Enables real-time, back-and-forth conversations with users via voice commands.
""")

# Technologies Used
st.subheader("Technologies Used: ")
st.write("""  I leverages tools like Deepgram for real-time transcription, LangChain for language model integration, and Google APIs for authentication and AI services.
 Additionally, libraries such as PyAudio, Streamlit, and Pydantic ensure smooth voice command processing, interactive interfaces, and data validation.""")


# Add Video Link
st.markdown("[Click here to watch the demo video on how this AI Voice Asistant  works.](https://www.loom.com/share/608753e9879e4c4ca716303db3e73037?sid=d11ae961-d6b0-4889-892a-01f17bd09030)")

st.subheader("Screenshots:")
# Add Image (replace with your own image path)
st.image("images/sale_agent_demo1.png", caption="AI-Scheduling meeting")
st.image("images/sale_agent_demo2.png", caption="AI-Scheduling meeting")

