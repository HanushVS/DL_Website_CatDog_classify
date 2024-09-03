from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

dic = {0: 'Cat', 1: 'Dog'}

model = load_model('model.h5')

def predict_label(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(100, 100))
    img = image.img_to_array(img) / 255.0
    img = img.reshape(1, 100, 100, 3)
    
    # Predict the class
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    return dic[predicted_class]

# Routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename    
        img.save(img_path)

        # Ensure image is saved successfully
        if not os.path.isfile(img_path):
            return render_template("index.html", prediction="Image could not be saved.", img_path=None)
        
        # Predict the label
        p = predict_label(img_path)

        return render_template("index.html", prediction=p, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
