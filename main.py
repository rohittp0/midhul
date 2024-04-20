from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('yolov8x.pt')


@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        # Handle image upload and prediction
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image = Image.open(uploaded_file)
            result = model.predict(source=image)[0]
            names = result.names

            predicted_class = names[int(result.boxes.cls[0])]
            predicted_probability = float(result.boxes.conf[0])

            response = {
                'predicted_class': predicted_class,
                'predicted_probability': predicted_probability
            }
            return jsonify(response)
    return jsonify({'error': 'No file uploaded'}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
