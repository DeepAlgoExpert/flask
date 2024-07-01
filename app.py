from flask import Flask, request, send_file
import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import base64
from gradio_client import Client, file
import shutil
from flask_cors import CORS
import requests

app = Flask(__name__)

CORS(app, resources={r"/tryon": {"origins": "https://oasis3d.netlify.app"}},
          allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin", "Access-Control-Allow-Headers", "Access-Control-Allow-Methods"])

CORS(app, resources={r"/virtual-fit": {"origins": "https://vtryandbuy.netlify.app"}},
          allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin", "Access-Control-Allow-Headers", "Access-Control-Allow-Methods"])

CORS(app, resources={r"/virtual-fit": {"origins": "https://vtryandbuy.netlify.app"}},
          allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin", "Access-Control-Allow-Headers", "Access-Control-Allow-Methods"])

CORS(app, resources={r"/virtual-fit-demo": {"origins": "https://virtualfitting.netlify.app"}},
          allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin", "Access-Control-Allow-Headers", "Access-Control-Allow-Methods"])

CORS(app, resources={r"/virtual-fit-demo": {"origins": "https://vtryandbuy.netlify.app"}},
          allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin", "Access-Control-Allow-Headers", "Access-Control-Allow-Methods"])

CORS(app, resources={r"/": {"origins": "https://oasis3d.netlify.app"}},
          allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin", "Access-Control-Allow-Headers", "Access-Control-Allow-Methods"])
# Define the directory where uploaded files are stored

CORS(app, resources={r"/": {"origins": "http://localhost:3000"}},
          allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin", "Access-Control-Allow-Headers", "Access-Control-Allow-Methods"])
# Define the directory where uploaded files are stored

CORS(app, resources={r"/virtual-fit": {"origins": "https://styleshifter.netlify.app/"}},
          allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin", "Access-Control-Allow-Headers", "Access-Control-Allow-Methods"])

CORS(app, resources={r"/": {"origins": "https://styleshifter.netlify.app/"}},
          allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Origin", "Access-Control-Allow-Headers", "Access-Control-Allow-Methods"])
# Define the directory where uploaded files are stored






# Define the model upload folder
MODEL_UPLOAD_FOLDER = 'images/model'
app.config['MODEL_UPLOAD_FOLDER'] = MODEL_UPLOAD_FOLDER

# Ensure the model upload folder exists
if not os.path.exists(MODEL_UPLOAD_FOLDER):
    os.makedirs(MODEL_UPLOAD_FOLDER)


# Define the model upload folder
GARMENT_UPLOAD_FOLDER = 'images/garment'
app.config['GARMENT_UPLOAD_FOLDER'] = GARMENT_UPLOAD_FOLDER

# Ensure the model upload folder exists
if not os.path.exists(GARMENT_UPLOAD_FOLDER):
    os.makedirs(GARMENT_UPLOAD_FOLDER)

RESULT_FOLDER = 'result'
app.config['VTO_RESULT_FOLDER'] = RESULT_FOLDER

# Ensure the model upload folder exists
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'png'}

# Function to check if a filename has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_base64_image(base64_string, filename):
    try:
        # Extract base64 image data
        _, base64_data = base64_string.split(',', 1)
        
        # Decode base64 data
        image_data = base64.b64decode(base64_data)
        
        # Write image data to file
        with open(filename, 'wb') as f:
            f.write(image_data)
        
        print(f"Image saved as {filename}")
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

# Endpoint to serve vto result image
@app.route('/', methods=['GET'])
def uploaded_file():
    return send_from_directory(RESULT_FOLDER, 'image.png')

@app.route('/', methods=['GET', 'POST'])
def try_on():
    
    model = request.json['model']
    model_type = request.json['modelType']
    garment = request.json['garment']
    garment_type = request.json['garmentType']
    print('model:', model)

    model_filename = "model.jpg"
    model_path= os.path.join(app.config['MODEL_UPLOAD_FOLDER'], model_filename)
    save_base64_image(model, model_path)

    garment_filename = "garment.jpg"
    garment_path= os.path.join(app.config['GARMENT_UPLOAD_FOLDER'], garment_filename)
    save_base64_image(garment, garment_path)
    
    app.logger.info("garment_type: %s", garment_type)

    client = Client("https://levihsu-ootdiffusion.hf.space/--replicas/6urx6/")

    if (garment_type=="Upper-body"):
        result = client.predict(
        os.path.join(app.config['MODEL_UPLOAD_FOLDER'], model_filename),	# filepath  in 'Model' Image component
        os.path.join(app.config['GARMENT_UPLOAD_FOLDER'], garment_filename),	# filepath  in 'Garment' Image component
        1,	# float (numeric value between 1 and 4) in 'Images' Slider component
        20,	# float (numeric value between 20 and 40) in 'Steps' Slider component
        2,	# float (numeric value between 1.0 and 5.0) in 'Guidance scale' Slider component
        -1,	# float (numeric value between -1 and 2147483647) in 'Seed' Slider component
        api_name="/process_hd"
        )
    else:
        result = client.predict(
        os.path.join(app.config['MODEL_UPLOAD_FOLDER'], model_filename),	# filepath  in 'Model' Image component
        os.path.join(app.config['GARMENT_UPLOAD_FOLDER'], garment_filename),	# filepath  in 'Garment' Image component
        garment_type,	# Literal['Upper-body', 'Lower-body', 'Dress']  in 'Garment category (important option!!!)' Dropdown component
        1,	# float (numeric value between 1 and 4) in 'Images' Slider component
        20,	# float (numeric value between 20 and 40) in 'Steps' Slider component
        2,	# float (numeric value between 1.0 and 5.0) in 'Guidance scale' Slider component
        -1,	# float (numeric value between -1 and 2147483647) in 'Seed' Slider component
        api_name="/process_dc"
        )

    #vto_url = f"http://localhost:5000/results/{result_filename}"
    source = result[0]["image"]
    print('source:', source)
    destination = app.config['VTO_RESULT_FOLDER']

    result_file_path = 'result/image.png'

    if os.path.exists(result_file_path):
        # Delete the file
        os.remove(result_file_path)
        print(f"File '{result_file_path}' deleted successfully.")
    else:
        print(f"File '{result_file_path}' does not exist.")

    # Move the file
    shutil.move(source, destination)
    #os.remove(source)
    vto_url = f"http://localhost:5000/image.png"

    with open(result_file_path, 'rb') as file:
        image_bytes = file.read()
        # Convert image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
    return base64_image

@app.route('/virtual-fit', methods=['GET', 'POST'])
def try_on_fit():
    
    model = request.json['model']
    print('model:', model)
    model_type = request.json['modelType']
    garment = request.json['garment']
    garment_type = request.json['garmentType']
    subgarment_type = request.json['subgarmentType']
    print('garment_type:', garment_type)
    print('subgarment_type:', subgarment_type)

    model_filename = "model.jpg"
    model_path= os.path.join(app.config['MODEL_UPLOAD_FOLDER'], model_filename)
    save_base64_image(model, model_path)

    garment_filename = "garment.jpg"
    garment_path= os.path.join(app.config['GARMENT_UPLOAD_FOLDER'], garment_filename)
    save_base64_image(garment, garment_path)
    
    app.logger.info("subgarment_type: %s", subgarment_type)

    client = Client("http://3.239.7.20:7860/")

    # Test the endpoint by ensuring the input parameters match those expected by the Gradio app
    result = client.predict(
        model_path,  # for imgs
        garment_path,  # for garm_img
        garment_type,  # for category
        subgarment_type,  # for prompt
        api_name="/virtual-dressing"
    )
    #vto_url = f"http://localhost:5000/results/{result_filename}"
    source = result
    destination = app.config['VTO_RESULT_FOLDER']

    result_file_path = 'result/image.png'

    if os.path.exists(result_file_path):
        # Delete the file
        os.remove(result_file_path)
        print(f"File '{result_file_path}' deleted successfully.")
    else:
        print(f"File '{result_file_path}' does not exist.")

    # Move the file
    shutil.move(source, destination)
    #os.remove(source)
    vto_url = f"http://localhost:5000/image.png"

    with open(result_file_path, 'rb') as file:
        image_bytes = file.read()
        # Convert image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
    return base64_image


def download_image(url, save_path):
    # Send a HTTP request to the image URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open the file in binary write mode and save the image content
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Image successfully downloaded and saved to {save_path}")
    else:
        print(f"Failed to retrieve image. HTTP Status code: {response.status_code}")

@app.route('/virtual-fit-demo', methods=['GET', 'POST'])
def try_on_fit_demo():
    
    model = request.json['model']
    print('model:', model)
    model_type = request.json['modelType']
    garment = request.json['garment']
    garment_type = request.json['garmentType']
    subgarment_type = request.json['subgarmentType']
    print('garment_type:', garment_type)
    print('subgarment_type:', subgarment_type)

    model_filename = "model.jpg"
    model_path= os.path.join(app.config['MODEL_UPLOAD_FOLDER'], model_filename)
    download_image(model, model_path)

    garment_filename = "garment.jpg"
    garment_path= os.path.join(app.config['GARMENT_UPLOAD_FOLDER'], garment_filename)
    download_image(garment, garment_path)
    
    app.logger.info("subgarment_type: %s", subgarment_type)

    client = Client("http://44.211.73.129:7860/")

    # Test the endpoint by ensuring the input parameters match those expected by the Gradio app
    result = client.predict(
        model_path,  # for imgs
        garment_path,  # for garm_img
        garment_type,  # for category
        subgarment_type,  # for prompt
        api_name="/virtual-dressing"
    )
    #vto_url = f"http://localhost:5000/results/{result_filename}"
    source = result
    destination = app.config['VTO_RESULT_FOLDER']

    result_file_path = 'result/image.png'

    if os.path.exists(result_file_path):
        # Delete the file
        os.remove(result_file_path)
        print(f"File '{result_file_path}' deleted successfully.")
    else:
        print(f"File '{result_file_path}' does not exist.")

    # Move the file
    shutil.move(source, destination)
    #os.remove(source)
    vto_url = f"http://localhost:5000/image.png"

    with open(result_file_path, 'rb') as file:
        image_bytes = file.read()
        # Convert image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
    return base64_image
    #return send_from_directory(result_file_path)



if __name__ == "__main__":
    
    #app.run(debug=True, threaded=False)
    app.run(host = "0.0.0.0", port=5000)
    #app.run(debug=True, port=5000, threaded=False)



