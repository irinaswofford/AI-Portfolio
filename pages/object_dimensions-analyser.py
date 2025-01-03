import streamlit as st

# Title of the application
st.title("Object Dimension Analyzer")

# Description
st.subheader("Description:")
st.write("""
The Object Dimension Analyzer is an interactive tool that uses computer vision to measure the dimensions of objects in an image. By uploading an image and specifying the real-world width of a reference object, the application calculates the dimensions of other objects in the image and overlays the measurements directly onto the image. This is useful in scenarios like quality control, object sizing, or educational purposes.
""")

# Key Features
st.subheader("Key Features:")
st.markdown("""
- **Image Uploading:**  
  Users can upload an image containing one or more objects.

- **Reference Measurement Input:**  
  Input the width of a known reference object (e.g., the left-most object) in inches.

- **Dimension Calculation:**  
  Automatically calculates the dimensions (height and width) of other objects based on the reference width.

- **Visualization:**  
  The processed image displays contours and annotated dimensions for each detected object.
""")

# Technologies Used
st.subheader("Technologies Used:")
st.markdown("""
- **Streamlit:** Simplified deployment of the interactive web application.
- **OpenCV:** For image processing and computer vision tasks like contour detection.
- **SciPy:** For precise Euclidean distance calculations between points.
- **Imutils:** Utility library for streamlined image processing.
- **NumPy:** For numerical computations and array manipulations.
""")

# Example Workflow
st.subheader("Example Workflow:")
st.markdown("""
1. Upload an image containing multiple objects.
2. Input the width of the left-most object (e.g., 0.955 inches).
3. The tool calculates and displays the dimensions of the remaining objects in the image.
4. The annotated image is displayed for download or further analysis.
""")
