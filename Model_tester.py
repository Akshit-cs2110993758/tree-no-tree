import cv2
import numpy as np
from PIL import Image
from keras import models
import tensorflow as tf
import os
cwd = os.getcwd()
model = models.load_model(cwd + r"\treechecker.model")
video = cv2.VideoCapture(0)
class_names = ['Not_Tree_Images', 'Tree_Images']
while True:
        _, frame = video.read()

        im = Image.fromarray(frame, 'RGB')

        im = im.resize((256,256))
        img_array = np.array(im)

        img_array = np.expand_dims(img_array, axis=0)


        prediction = int(model.predict(img_array)[0][0])
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        frames = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(frames, (11, 11), 0)
        canny = cv2.Canny(blur, 30, 150, 3)
        dilated = cv2.dilate(canny, (1, 1), iterations=0)
        # Add box around detection      **(under construction)**

        (cnt,hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in cnt:
                x,y,w,h = cv2.boundingRect(i)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 4)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)        
        print(f'number of tree {len(cnt)} its {class_names[np.argmax(score)]} with {100*np.max(score)} accuracy')

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()
