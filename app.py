from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np
import tensorflow as tf
from flask_mail import Mail,Message

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)
app.debug=True
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'ayushompanda107@gmail.com'
app.config['MAIL_PASSWORD'] = 'ecvbdktqmqhennyf'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail=Mail(app)

MODEL_PATH = 'models/trained_model.h5'


model = load_model(MODEL_PATH)
       
print('Model loaded. Start serving...')



result = ""
def model_predict(img_path, model):
    img = tf.keras.utils.load_img(img_path, target_size=(64, 64))

  
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)

   
    preds = model.predict(img)
    return preds


@app.route('/', methods=['GET'])
def index():
   
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        f = request.files['file']
       

        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        
        preds = model_predict(file_path, model)
        os.remove(file_path)

       
        str1 = 'Pneumonia'
        str2 = 'Normal'
        if preds ==1:
           global result 
           result = str1
        else:
           result = str2
        
        if preds == 1:
            return str1
        else:
            return str2
        
    return None

@app.route('/email',methods=["Post"])
def form():
    global result
    name = request.form.get("name")
    email = request.form.get("email")
    msg = Message("Hey",sender='dontdoit@demo.com',recipients=[email])
    msg.body = f"Hey {name}, According to your given lung X-Ray and our model prediction  you have {result} condition "
    app.logger.info(name,email,result)
    mail.send(msg)
    return render_template('index.html')
    
    

    
if __name__ == '__main__':
        app.run()
