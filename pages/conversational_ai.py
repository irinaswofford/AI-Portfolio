import streamlit as st

# Streamlit app main function
def main():
    st.title("Conversational AI fine-tuned with RAG")  # Set the title of the page

    # Fine-tuning section
    st.write("Fine-tuning was done using  tokenizer to break down the input text into smaller, understandable units (tokens). Then, I utilized Chroma to store the vector embeddings of the documents, which are numerical representations capturing the semantic meaning of the text. During fine-tuning, the model retrieves relevant document embeddings from Chroma based on user queries, allowing it to generate more accurate and contextually relevant responses.")

    st.header("Projects:")
    # Description of the AI-Powered Teaching Assistant Project
    st.write("""
    ### 1. AI-Powered Teaching Assistant
    This project is designed to help educators by automating and simplifying the creation of multiple-choice assessments. 
    This system leverages **Generative AI** to process and convert various input sources (like articles, textbooks, or lectures) 
    into multiple-choice questions, saving educators time and effort in test creation. 

    The system is scalable, meaning it can easily incorporate additional input types, such as personal notes, videos, or other 
    resource formats, in the future.

    This system utilizes **Conversational AI** capabilities and was fine-tuned using **Retrieval-Augmented Generation (RAG)**, 
    allowing it to understand and generate human-like responses to help educators further refine the assessments. 
    By automating the question creation process, educators can focus more on teaching and less on manual test preparation.
    """)

    # Add Video Link for AI-Powered Teaching Assistant
    st.markdown("[Click here to watch the presentation and video on the last slide explaining how the AI-Powered Teaching Assistant works.](https://pitch.com/v/ai-quiz-generative-tool-f84r6n)")

    st.write("""
    ### 2. AI-Powered Flight Management Assistant (AI Agent)
    We designed the AI-powered Flight Management Assistant to scale and improve both performance and customer experience. 
    By integrating **LangGraph**, we ensured enhanced **Human-in-the-Loop (HITL)** customization, a refined agentic framework, 
    and parallelism for multitasking efficiency. The assistant is built with the capacity to handle future tools like car booking 
    and hotel reservations.

    **Key Features**:

    - **Conversational AI with RAG (Retrieval-Augmented Generation)**: Combines real-time data retrieval with generative AI 
      (Gemini Pro LLM) to deliver accurate, context-aware responses. It can pull live data such as flight statuses and provide 
      dynamic answers to questions like policy updates.
    
    - **Enhanced Agentic Framework (ReAct) for Autonomous Actions**: Allows the assistant to autonomously manage workflows, 
      such as booking flights and coordinating accommodations. It makes decisions based on real-time data, ensuring smooth, 
      automated customer interactions.
    
    - **Planning & Execution**: The assistant can autonomously handle tasks like flight bookings, hotel reservations, and other 
      travel logistics, integrating real-time inputs for efficiency.
    """)

    # Add Video Link for AI-Powered Flight Management Assistant
    st.markdown("[Click here to watch the presentation and video on the last slide explaining how the AI-Powered Flight Management Assistant works.](https://pitch.com/v/ai-vistiu)")

    st.write("""
    ### 3. AI-Powered Flashcard and Quiz Generator
    Another project designed for educators and learners is the **AI-Powered Flashcard and Quiz Generator**. This tool automates 
    the creation of flashcards and quizzes based on input materials, making learning more interactive and engaging.

    **Key Features**:

    - **Automated Question Generation**: It converts articles, lecture notes, and other content into multiple-choice questions, 
      flashcards, and quizzes, streamlining the study process.
    
    - **Personalized Content**: It adjusts the complexity and difficulty of the content based on the learner's progress and preferences.
    
    - **Integration with Study Tools**: This system can be integrated with popular learning platforms to further enhance its impact.
    """)

    # Add Video Link for AI-Powered Flashcard Generator
    st.markdown("[Click here to watch the presentation and video on the last slide explaining how the AI-Powered Flashcard and Quiz Generator works.](https://pitch.com/v/ai-youtube-video-flashcard-generator-n9r7vs)")

# Run the Streamlit app
if __name__ == "__main__":
    main()
