
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import load_img,img_to_array
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from PIL import Image



x_train = []
y_train = np.array([])
x_test = []
y_test = []
y_predicted = []

global celeb
celeb = ["anne_hathaway","arnold_schwarzenegger","ben_afflek","dwayne_johnson","elton_john","jerry_seinfeld","kate_beckinsale",
        "keanu_reeves","lauren_cohan","madonna","mindy_kaling","simon_pegg","sofia_vergara","will_smith"] 



def en(name):
    for i in range(len(celeb)):
        if celeb[i] == name:
            return i+1


main_path="/home/amiryousof/Desktop/andriod/CR_tensorflow_1/images"

for folder in os.listdir(main_path+"/train"):
    for file in os.listdir(main_path+f"/train/{folder}"):
        
        #image = cv2.imread(f"E:\\P\\PYTHON\\CR\\images\\train\\{folder}\\{file}" ,cv2.IMREAD_GRAYSCALE)
        #img = load_img(main_path+f"/train/{folder}/{file}")
        image = cv2.imread(main_path+f"/train/{folder}/{file}",cv2.IMREAD_GRAYSCALE)
        image = keras.utils.normalize(image,axis=1)
        image = cv2.resize(image,(400,400))
        ##image = Image.open(main_path+f"/train/{folder}/{file}").convert('LA')
        #image = img_to_array(img)
        x_train.append(image) 
        
        np.append(y_train,en(folder))



for folder in os.listdir(main_path+"/test"):
    for file in os.listdir(main_path+f"/test/{folder}"):
        
        #image = cv2.imread(f"E:\\P\\PYTHON\\CR\\images\\test\\{folder}\\{file}" ,cv2.IMREAD_GRAYSCALE)
        #img = load_img(main_path+f"/test/{folder}/{file}")
        image = cv2.imread(main_path+f"/test/{folder}/{file}",cv2.IMREAD_GRAYSCALE)
        image = keras.utils.normalize(image,axis=1)
        ##image = Image.open(main_path+f"/test/{folder}/{file}").convert('LA')
        #image = img_to_array(img)
        
        x_test.append(image)
        np.append(y_test,en(folder))

#x_tarin = keras.utils.normalize(x_train,axis=1)
#x_test = keras.utils.normalize(x_test,axis=1)



model = Sequential()
#model.add(Flatten())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(len(celeb), (3, 3), activation='relu'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

y = model.predict(x_test)


