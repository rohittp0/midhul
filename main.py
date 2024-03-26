import requests
from picamera import PiCamera
from picamera.array import PiRGBArray
import io
import time
from PIL import Image

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

time.sleep(2)

# The URL to your Flask server endpoint
url = 'http://192.168.0.135:8000/'

print("Will talk to server at", url)

def send_for_prediction(image_path):
    """Send the captured image to the Flask server for prediction."""
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    return response

# Continuously capture images from the camera and send them for processing
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # Save the frame as an in-memory file
    image = frame.array
    Image.fromarray(image).save("temp_image.png")
    # cv2.save("temp_image.png", image)
  
    # Send the image for prediction
    response = send_for_prediction("temp_image.png")
    if response.ok:
        data = response.json()
        if data["predicted_probability"] > 0.75:
            print(f"\rThe leaf is {data['predicted_class']} with probablity {data['predicted_probability']}", end="")
    else:
        print("Failed to get response from server", response.text)
    
    # Clear the stream to make it ready for the next frame
    rawCapture.truncate(0)
