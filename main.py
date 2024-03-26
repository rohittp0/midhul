from flask import Flask, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
import torchvision.models as models

app = Flask(__name__)

# Define class labels
class_labels = ["Bacterial spot", "Healthy"]

# Initialize the model
google_net = models.googlenet(pretrained=True)
google_net.fc = torch.nn.Linear(1024, 2)  # Corrected for binary classification

# Load the trained model
model_path = 'Swingog.h5'
model = google_net
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        # Handle image upload and prediction
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image = Image.open(uploaded_file)
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                predicted_class_idx = torch.argmax(probs).item()
                predicted_class = class_labels[predicted_class_idx]
                predicted_probability = probs[predicted_class_idx].item()

            response = {
                'predicted_class': predicted_class,
                'predicted_probability': predicted_probability
            }
            return jsonify(response)
    return jsonify({'error': 'No file uploaded'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
