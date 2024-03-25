import torch
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from picamera import PiCamera
from picamera.array import PiRGBArray
import io

# Class labels
class_labels = ["Bacterial spot", "Healthy"]

# Initialize the GoogleNet model
google_net = models.googlenet(pretrained=True)
google_net.fc = torch.nn.Linear(1024, 2)  # Adjusting the final layer for binary classification

# Load the trained model
model_path = 'Swingog.h5'
model = google_net
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the preprocessing
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize camera
camera = PiCamera()
camera.resolution = (640, 480)
rawCapture = PiRGBArray(camera, size=(640, 480))

# Capture images continuously
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # Convert the image into a PIL Image
    image = Image.open(io.BytesIO(frame.array))
    
    # Preprocess the image
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Make a prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        predicted_class_idx = torch.argmax(probs).item()
        predicted_class = class_labels[predicted_class_idx]
        predicted_probability = probs[predicted_class_idx].item()

    print(f"Predicted class: {predicted_class}, Probability: {predicted_probability}")
    
    # Clear the stream in preparation for the next frame
    rawCapture.truncate(0)
