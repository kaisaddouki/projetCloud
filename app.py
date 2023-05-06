from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

app = Flask(__name__)
model = load_model('DS.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    img_data = img_file.read()
    img = image.img_to_array(image.load_img(io.BytesIO(img_data), target_size=(32, 32)))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_idx = np.argmax(prediction)
    class_name = class_names[class_idx]
    print (class_name)
    return render_template('index.html', prediction=f'The image is a {class_name}')

if __name__ == '__main__':
    app.run(debug=True, port=8900)
