from asyncio.windows_events import NULL
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from PIL import Image

cwd = os.getcwd()
print(cwd)
trees = os.listdir(cwd + r'\Tree_NoTree_Dataset\Tree_Images')

model = tf.keras.models.load_model(cwd + r"\\treechecker.model")

class_names = ['Not_Tree_Images', 'Tree_Images']
for tree in trees:
    img = tf.keras.utils.load_img(
        cwd + r'\\Tree_NoTree_Dataset\\Tree_Images\\' + tree, target_size=(256, 256)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])


    print(f"------------------------------------------------\n-------------{tree}--------------\n")

    print(f'its {class_names[np.argmax(score)]} with {100*np.max(score)} accuracy')