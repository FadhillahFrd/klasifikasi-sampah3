# Dependencies
import numpy as np
import keras
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array 
from keras.models import load_model
import tensorflow as tf

# def predict(img_path):

def getPrediction(filename):
     model = load_model("./model/final_model9010.h5")
     img = load_img('static/'+filename, target_size=(180, 180))
     img = img_to_array(img)
     img = img / 255
     img = np.expand_dims(img,axis=0)
     category = model.predict(img)
     classes=np.argmax(category,axis=1)
     answer = classes[0]
     probability = model.predict(img)
     probability_results = 0

     if answer == 1:
          answer = "Recycle"
          probability_results = probability[0][1]
          probability_results = int(probability_results * 100)
     else:
          answer = "Organic"
          probability_results = probability[0][0]
          probability_results = int(probability_results * 100)

     answer = str(answer)
     probability_results=str(probability_results) + str("%")

     values = [answer, probability_results, filename]
     return values[0], values[1], values[2]
