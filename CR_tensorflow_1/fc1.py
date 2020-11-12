
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.preprocessing.image import load_img,img_to_array
import os
import matplotlib.pyplot as plt
import cv2

x_train = []
y_train = []
x_test = []
y_test = []
x_predicted = []
global celeb
celeb = ["anne_hathaway","arnold_schwarzenegger","ben_afflek","dwayne_johnson","elton_john","jerry_seinfeld","kate_beckinsale",
        "keanu_reeves","lauren_cohan","madonna","mindy_kaling","simon_pegg","sofia_vergara","will_smith"] 



def encoder(name):
    for i in range(len(celeb)):
        if celeb[i] == name:
            return i+1




for folder in os.listdir("E:\\P\\PYTHON\\CR\\images\\train"):
    for file in os.listdir(f"E:\\P\\PYTHON\\CR\\images\\train\\{folder}"):
        
        #image = cv2.imread(f"E:\\P\\PYTHON\\CR\\images\\train\\{folder}\\{file}" ,cv2.IMREAD_GRAYSCALE)
        img = load_img(f"E:\\P\\PYTHON\\CR\\images\\train\\{folder}\\{file}")
        image = img_to_array(img)
        x_train.append(image) 
        
        y_train.append(encoder(folder))



for folder in os.listdir("E:\\P\\PYTHON\\CR\\images\\test"):
    for file in os.listdir(f"E:\\P\\PYTHON\\CR\\images\\test\\{folder}"):
        
        #image = cv2.imread(f"E:\\P\\PYTHON\\CR\\images\\test\\{folder}\\{file}" ,cv2.IMREAD_GRAYSCALE)
        img = load_img(f"E:\\P\\PYTHON\\CR\\images\\test\\{folder}\\{file}")
        image = img_to_array(img)
        
        x_test.append(image)
        y_test.append(encoder(folder))



model = Sequential()
model.add(Flatten())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(len(celeb), (3, 3), activation='relu'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

y = model.predict(x_test)


