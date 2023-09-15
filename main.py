import torch
from PIL import Image
from flask import Flask, render_template, request
from torchvision import transforms
import torchvision.models as models


app = Flask(__name__)

# Define your class labels here
class_labels = ["Bacterial spot", "Healthy"]

google_net = models.googlenet(pretrained=True)
google_net.fc = torch.nn.Linear(1024, 4)  # Binary classification, 2 output classes

# Load the trained model
model_path = 'Swingog.h5'  # Specify the path to your saved model checkpoint
model = google_net  # Use your actual model instance here

# Load the model state dict
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Move the model to the same device as the input tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle image upload and prediction
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Load and preprocess the uploaded image
            image = Image.open(uploaded_file)
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            # Make a prediction
            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                predicted_class_idx = torch.argmax(probs).item()
                predicted_class = class_labels[predicted_class_idx]
                predicted_probability = probs[predicted_class_idx].item()

            return render_template('index.html', prediction=f'Predicted Class: {predicted_class}',
                                   probability=f'Predicted Probability: {predicted_probability:.4f}',
                                   text=(predicted_class=='Bacterial spot'))
    return render_template('index.html')


app.run(host='localhost', port=8000, debug=True)
