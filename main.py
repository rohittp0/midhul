import requests
from picamera import PiCamera
from picamera.array import PiRGBArray
import io
import time

# Initialize the camera
camera = PiCamera()
camera.resolution = (640, 480)
rawCapture = PiRGBArray(camera, size=(640, 480))

# Allow the camera to warm up
time.sleep(2)

# The URL to your Flask server endpoint
url = 'http://<Flask_Server_IP>:8000/'

print("Will talk to server at", url)

def send_for_prediction(image_path):
    """Send the captured image to the Flask server for prediction."""
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    return response

# Continuously capture images from the camera and send them for processing
for frame in camera.capture_continuous(rawCapture, format='jpeg', use_video_port=True):
    # Save the frame as an in-memory file
    stream = io.BytesIO(frame.array)
    image_path = 'temp_image.jpg'  # Temporary image path
    with open(image_path, 'wb') as f:
        f.write(stream.getvalue())
    
    # Send the image for prediction
    response = send_for_prediction(image_path)
    if response.ok:
        print(response.json())
    else:
        print("Failed to get response from server", response.text)
    
    # Clear the stream to make it ready for the next frame
    rawCapture.truncate(0)
