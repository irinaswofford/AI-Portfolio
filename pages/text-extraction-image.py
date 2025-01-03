import streamlit as st
import pytesseract
import cv2
from PIL import Image
import numpy as np

# Streamlit app title
st.title("Computer Vision Extract text from an image")

# Header for the project section
st.header("Text Extraction and Object Measurement (OCR)")

# Introduction to the project and technologies used
st.write("""
This application allows you to upload an image and extract text using Optical Character Recognition (OCR). 
It also applies some image processing techniques such as Gaussian blur for noise reduction. 
The OCR is powered by **Tesseract**, a popular OCR engine. The image processing is done using **OpenCV**.
""")

# Function to preprocess the image and extract text
def ocr_process(image):
    # Convert the image to OpenCV format (numpy array)
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Apply thresholding (binary image) for better OCR performance
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Apply some noise reduction using GaussianBlur
    blurred_image = cv2.GaussianBlur(thresh, (5, 5), 0)

    # OCR text extraction using pytesseract
    # Use tesseract configuration to get better results
    custom_config = r'--oem 3 --psm 6'  # OEM 3: Default, PSM 6: Assume a single uniform block of text
    text = pytesseract.image_to_string(blurred_image, config=custom_config)

    return text

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Open the image with Pillow
    image = Image.open(uploaded_image)

    # Process the image using the OCR model
    extracted_text = ocr_process(image)

    # Display the extracted text
    st.subheader("Extracted Text:")
    st.write(extracted_text)

    # Check if the extracted text matches the expected quote
    expected_quote = "LIVE LIFE WITH NO EXCUSES, TRAVEL WITH NO REGRET"
    if expected_quote in extracted_text.upper():  # Case insensitive comparison
        st.subheader("Attribution:")
        st.write("**Author**: Oscar Wilde")

    # Display the original image
    st.image(image, caption="Original Image", use_container_width=True)

    # Image processing example (e.g., applying Gaussian blur for noise reduction)
    st.subheader("Processed Image:")
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    st.image(blurred_image, caption="Processed Image with Gaussian Blur", use_container_width=True)
