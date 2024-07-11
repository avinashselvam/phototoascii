from flask import Flask, request, send_file
from flask_cors import CORS
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

def get_opencv_img_from_buffer(buffer):
    bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
    return cv2.imdecode(bytes_as_np_array, cv2.IMREAD_GRAYSCALE)

def process_image(image):

    height, width = image.shape

    SCALE = 100

    image = cv2.resize(image, ((width*SCALE)//height, SCALE), interpolation = cv2.INTER_LINEAR)

    edges = cv2.Canny(image, 100, 200)

    # Apply thresholding
    _, thresh = cv2.threshold(edges, 128, 255, cv2.THRESH_BINARY)

    # Compute gradients
    grad_x = cv2.Sobel(thresh, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(thresh, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitudes and angles
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

    # Map angles to text characters
    def angle_to_char(angle):
        angle += 90
        if 90-22.5 <= angle < 90+22.5 or 270-22.5 <= angle < 270+22.5:
            return '||'
        elif 45-22.5 <= angle < 45+22.5 or 225-22.5 <= angle < 225+22.5:
            return '/ '
        elif 135-22.5 <= angle < 135+22.5 or 315-22.5 <= angle < 315+22.5:
            return ' \\'
        else:
            return '- '

    # Create text representation
    height, width = angle.shape
    text_image = []

    for y in range(height):
        for x in range(width):
            if magnitude[y, x] > 0:
                text_image.append(angle_to_char(angle[y, x]))
            else:
                text_image.append("  ")
        text_image.append("\n")

    ascii = "".join(text_image)
    
    return ascii

@app.get("/")
def status():
    return "ok", 200

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    image = get_opencv_img_from_buffer(file.stream)
    
    return process_image(image), 200

