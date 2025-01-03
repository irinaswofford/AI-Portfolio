import streamlit as st
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2


st.title("Object Dimension Analyzer")
st.write("""
This is an interactive tool that uses computer vision to measure the dimensions of objects in an image. By uploading an image and specifying the real-world width of a reference object, the application calculates the dimensions of other objects in the image and overlays the measurements directly onto the image. This is useful in scenarios like quality control, object sizing, or educational purposes.
""")
# Technologies Used
st.subheader("Technologies Used:")
st.markdown("""
- **OpenCV:** For image processing and computer vision tasks like contour detection.
- **SciPy:** For precise Euclidean distance calculations between points.
- **Imutils:** Utility library for streamlined image processing.
""")

# Example Workflow
st.subheader("Example Workflow:")
st.markdown("""
1. Upload an image containing multiple objects.
2. Input the width of the left-most object (e.g., 0.955 inches).
3. The tool calculates and displays the dimensions of the remaining objects in the image.
""")
import streamlit as st
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

# Function to calculate the midpoint between two points
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Function to process the image and detect object sizes
def process_image(image, width):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) == 0:
        return None, "No contours detected. Please ensure the image contains discernible objects."

    (cnts, _) = contours.sort_contours(cnts)
    
    pixelsPerMetric = None
    results = []
    
    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
        
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / width
        
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric
        
        results.append((box, dimA, dimB))
    
    return results, None

# Function to draw the results on the image
def draw_results(image, results):
    for (box, dimA, dimB) in results:
        cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)
        
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        
        cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        
        cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)
        
        cv2.putText(image, "{:.1f}in".format(dimA), (int(tltrX - 15), int(tltrY - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(image, "{:.1f}in".format(dimB), (int(trbrX + 10), int(trbrY)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    
    return image

# Upload image and specify reference width
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Number input for the reference object width (in inches)
width = st.number_input("Width of the left-most object (in inches)", value=0.955, step=0.001)

# Ensure width is greater than 0
if width <= 0:
    st.error("Please enter a valid width for the left-most object (greater than 0).")

# Processing the image
if uploaded_file is not None:
    try:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    except Exception as e:
        st.error(f"Error loading the image: {e}")
    
    # Process the image with the reference width
    results, error = process_image(image, width)
    
    if error:
        st.error(error)
    else:
        # Display the processed image and results
        output_image = draw_results(image.copy(), results)
        st.image(output_image, channels="BGR", caption="Processed Image")
        
        st.write(f"Number of objects detected: {len(results)}")
        
        for i, (_, dimA, dimB) in enumerate(results, 1):
            st.write(f"Object {i}: {dimA:.1f}in x {dimB:.1f}in")
