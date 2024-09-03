###

# Image Classification Web Application

This project is a web application designed for image classification using a deep learning model. Users can upload images, which are then classified by a pre-trained neural network model.

## Features

- **Image Upload**: Allows users to upload images via a web form.
- **Image Classification**: Classifies the uploaded image as either 'Cat' or 'Dog' using a deep learning model.
- **Results Display**: Shows the uploaded image along with the predicted classification result.

## Technologies Used

- **Flask**: A lightweight WSGI web application framework for Python.
- **Keras**: Provides a high-level API for building and training the neural network model.
- **Bootstrap**: Used for creating a responsive and modern web design.
- **HTML/CSS**: For structuring and styling the web interface.

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/HanushVs/image-classification-web-app.git
   cd image-classification-web-app
   
## Install Dependencies

Make sure you have Python 3 installed. Create and activate a virtual environment, then install the required packages:

bash

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

Add the Model File

Ensure that you have the pre-trained model file (model.h5) in the project directory. This model file can be trained separately and saved as model.h5.

Run the Application

bash

python app.py
The application will be available at http://127.0.0.1:5000/.

## Project Structure
app-py: The main Flask application script.
templates/index.html: HTML template for the user interface.
static/styles.css: CSS file for styling the web pages.
model.h5: Pre-trained Keras model for image classification.
requirements.txt: List of Python dependencies.

## Contributors
HanushVs (@HanushVs)
Adivsb (@Adivsb)



Acknowledgments
