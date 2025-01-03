import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Streamlit app main function
def main():
    # Title of the app
    st.title("Sales Forecasting with LSTM for Future Receipts")

    # LSTM Networks Introduction
    st.markdown("""
    **Long Short-Term Memory (LSTM)** networks are a type of Recurrent Neural Network (RNN) designed to learn and remember order dependencies in sequence prediction tasks. 
    I used a Flask app for visualizing the future amount of receipts, incorporating an LSTM model to predict the next month's sales based on the previous 3 months of sales data.
    """)

    # Tools Used
    st.markdown("""
    **Tools Used:**
    - Torch 
    - Python
    - NumPy
    - Docker
    - Pandas
    - Matplotlib
    - Flask
    """)

    # Add Docker Setup Instructions
    st.markdown("""
    ## Docker Setup for Running the Application Locally:
    
    **1. Clone the Repository:**
    ```bash
    git clone https://github.com/irinaswofford/receipe-predictor.git


    ```

    **2. Navigate to the Root Directory:**
    ```bash
    cd ReceiptPredictor
    ```

    **3. Build the Docker Image:**
    Run the following command to build the Docker image:
    ```bash
    docker build -t receipt-predictor .
    ```

    **Note:** If the build fails, make sure Docker is installed and running correctly, and check the Dockerfile for any dependencies that may need to be installed.

    **4. Run the Docker Container:**
    Once the image is built, run the container with the following command:
    ```bash
    docker run -p 5000:5000 receipt-predictor
    ```

    **5. Access the Application:**
    Open your web browser and go to [http://localhost:5000]) to access the app.
    """)

    # Add Video Link (replace with your own video URL)-to do replace with my video 
    st.markdown("Click here to watch the demo video on how to use Docker to run my Receipt Prediction Model. https://www.loom.com/share/5b59674b71f8454aa0607d7c4624b8fa?sid=7033b9b5-ee15-4308-bec8-cd533c74e66e")


    st.subheader("Screenshot:")
    # Add Image (replace with your own image path)
    st.image("images/receipt_predictor.png", caption="AI-Powered Sales Forecasting Interface")

    # Example Workflow
    st.markdown("""
    ## Example Workflow:
    
    1. Input the past three months' receipts (e.g., 250M, 260M, 280M).
    2. Click "Predict" to get the forecasted receipts for the next month.
    
    **Input Example:**
    - **Month 1:** 250M
    - **Month 2:** 260M
    - **Month 3:** 280M
    
    The app will use these inputs to predict the receipts for the next month based on the trained LSTM model.
    """)

# Run the main function
if __name__ == "__main__":
    main()
