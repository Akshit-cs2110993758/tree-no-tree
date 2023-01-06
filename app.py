#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2


path = os.getcwd() 
model = tf.keras.models.load_model(path + r"\\treechecker.model")

class_names = ['Not_Tree_Images', 'Tree_Images']

app = Flask(__name__)

UPLOAD_FOLDER = path + r'/static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     

@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        # img = tf.keras.utils.load_img(
        #     path+r'\static\uploads\\'+filename, target_size=(256, 256)
        # )
        # img_array = tf.keras.utils.img_to_array(img)
        img_array = cv2.imread(path+r'\static\uploads\\'+filename)
        img_array = cv2.resize(img_array,dsize=(256,256),interpolation=cv2.COLOR_BGR2GRAY)
        checker = -1
        for i in img_array:
            for j in i:
                for k in j:
                    if k==0:
                        checker = 0
                    else:
                        checker = 1
        if checker == 1:
            img_array = cv2.bitwise_not(img_array)
        
        img_array = tf.expand_dims(img_array, 0)
  

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        if class_names[np.argmax(score)] == 'Tree_Images':
            flash(f'its detected as a tree with {round(100*np.max(score),2)}% accuracy')
        if class_names[np.argmax(score)] == 'Not_Tree_Images':
            flash(f'its not detected as a tree with {round(100*np.max(score),2)}% accuracy')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)



@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run()