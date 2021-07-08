import sys
import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH ='model_resnet50.h5'

model = load_model(MODEL_PATH)

def model_predict(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))

    x= image.img_to_array(img)

    x=x/255
    x = np.expand_dims(x,axis=0)

    pred = model.predict(x)
    pred = np.argmax(pred,axis=1)

    if pred ==0:
        pred = "AUDI"
    elif pred==1:
        pred = "Lamborghini"
    else:
        pred = "Mercedes"
    
    return pred

@app.route('/',methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods =['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        base_path = os.path.dirname(__file__)
        file_path = os.path.join(
            base_path,'uploads',secure_filename(f.filename)
        )
        f.save(file_path)

        pred = model_predict(file_path,model)
        result = pred
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)

